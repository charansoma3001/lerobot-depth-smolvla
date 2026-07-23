"""
Build the depth supervision targets used to train C2.This runs Depth Anything V2 over every frame of the demonstration dataset,
reshapes each depth map to match what SmolVLA actually sees, shrinks it to a 64x64 grid, normalises the whole dataset together,
and saves the result as a small Hugging Face dataset keyed by (episode_index, frame_index).
With --norm=global the full script produces the same numbers; the extra length there is argparse, batching and QA images.
"""

import numpy as np
import torch
import torch.nn.functional as F
from datasets import Array2D, Dataset, Features, Value
from PIL import Image
from tqdm import tqdm
from transformers import AutoImageProcessor, AutoModelForDepthEstimation
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.smolvla.modeling_smolvla import resize_with_pad

DATASET_REPO = "charansoma3001/depth-smolvla-blue-demos_20260603_154319"
OUTPUT_REPO = "charansoma3001/smolvla-depth-maps"
CAMERA = "observation.images.overhead"
DAV2_MODEL = "depth-anything/Depth-Anything-V2-Small-hf"
DEPTH_SIZE = 64
DEVICE = "cuda"
PUSH_TO_HUB = True

def estimate_depth(frame, processor, model, target_h, target_w):
    """
    Return one 64x64 depth map for a single dataset frame.
    The frame is a (3, H, W) tensor in [0, 1]. DAV2 predicts a raw depth map at its own resolution, so we resize it back to the camera resolution and then
    apply the same aspect-preserving pad SmolVLA uses on the RGB input. This keeps the depth target spatially aligned with the image the policy is trained on.
    """
    # PyTorch stores images channels-first as (channels, height, width) with float values in [0, 1].
    # Depth Anything expects height-width-channels, 8-bit integers in [0, 255]. permute reorders the axes,
    # then we rescale and cast to match that convention.
    rgb = (frame.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    image = Image.fromarray(rgb)

    # The processor resizes the image to the resolution the network was trained on and normalises the pixels,
    # returning PyTorch tensors ("pt"), .to(DEVICE) moves those tensors onto the GPU.
    inputs = processor(images=image, return_tensors="pt").to(DEVICE)
    # no_grad turns off gradient. That is only needed while training a network, here we only run it forward, so switching it off saves memory and time.
    with torch.no_grad():
        # Depth Anything V2 is a monocular depth estimator: from a single RGB image it predicts a depth value for every pixel.
        # The values are relative, not real-world distances, a larger value means nearer to the camera.
        depth = model(**inputs).predicted_depth  # (1, h, w), larger means closer

    # The network emits depth at its own resolution. interpolate resamples that grid back up to the camera's pixel dimensions.
    # "bilinear" estimates each new pixel by blending the four nearest source pixels, which gives a smooth result rather than a jagged one.
    depth = F.interpolate(depth[None], size=(target_h, target_w), mode="bilinear", align_corners=False)
    # SmolVLA does not stretch images to a square. It scales the image to fit and pads the leftover strip. We apply the identical transform to the depth map
    # so each depth cell lines up with the same image region the policy sees, then downsize the whole thing to 64x64.
    depth = resize_with_pad(depth, DEPTH_SIZE, DEPTH_SIZE, pad_value=0)
    # Drop the two leading length-1 axes, copy the tensor off the GPU back into main memory, and return it as a plain NumPy array.
    return depth[0, 0].cpu().numpy()

def normalise(raw_depths):
    """
    Scale every frame to [0, 1] using dataset-wide 2nd/98th percentiles. Normalising the dataset as a whole means a given real-world
    height maps to the same value in every frame, which is what lets the depth target carry a consistent elevation signal.
    """
    # Use the 2nd and 98th percentiles rather than the raw min and max, so that extreme pixels cannot stretch the scale for every other frame.
    lo = np.percentile(raw_depths, 2)
    hi = np.percentile(raw_depths, 98)
    print(f"normalising with lo={lo:.3f}, hi={hi:.3f}")
    # Map the [lo, hi] band onto [0, 1] and clip anything outside it. The tiny 1e-8 term only helps if lo and hi ever turn out equal, which causes divide by zero.
    return np.clip((raw_depths - lo) / (hi - lo + 1e-8), 0.0, 1.0).astype(np.float32)

def main():
    # LeRobotDataset wraps the recorded robot demonstrations behind an interface. from_pretrained downloads the published Depth Anything V2 weights
    # and its matching processor from the Hugging Face model hub, .eval() puts the network in inference mode, disabling training layers.
    dataset = LeRobotDataset(DATASET_REPO)
    processor = AutoImageProcessor.from_pretrained(DAV2_MODEL)
    model = AutoModelForDepthEstimation.from_pretrained(DAV2_MODEL).to(DEVICE).eval()

    n_frames = dataset.num_frames
    orig_h, orig_w = dataset.features[CAMERA]["shape"][:2]
    raw_depths = np.zeros((n_frames, DEPTH_SIZE, DEPTH_SIZE), dtype=np.float32)
    episodes = np.zeros(n_frames, dtype=np.int64)
    frames = np.zeros(n_frames, dtype=np.int64)

    for i in tqdm(range(n_frames), desc="depth"):
        item = dataset[i]
        raw_depths[i] = estimate_depth(item[CAMERA], processor, model, orig_h, orig_w)
        episodes[i] = int(item["episode_index"])
        frames[i] = int(item["frame_index"])

    depths = normalise(raw_depths)

    # Assemble the targets into a Hugging Face dataset. Every row is keyed by (episode_index, frame_index)
    hf_dataset = Dataset.from_dict(
        {
            "episode_index": episodes.tolist(),
            "frame_index": frames.tolist(),
            "depth": [depths[i] for i in range(n_frames)],
        },
        features=Features(
            {
                "episode_index": Value("int64"),
                "frame_index": Value("int64"),
                "depth": Array2D(shape=(DEPTH_SIZE, DEPTH_SIZE), dtype="float32"),
            }
        ),
    )

    if PUSH_TO_HUB:
        hf_dataset.push_to_hub(OUTPUT_REPO, private=True)
        print(f"pushed to {OUTPUT_REPO}")
    else:
        hf_dataset.save_to_disk(OUTPUT_REPO.replace("/", "__") + ".local")
        print("saved locally")

if __name__ == "__main__":
    main()