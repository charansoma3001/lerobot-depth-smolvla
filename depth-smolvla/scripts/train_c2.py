"""
Train the Implicit depth (C2) policy: SmolVLA with an auxiliary depth-reconstruction task.
SmolVLA is a "vision-language-action" policy: a neural network that takes the robot's camera images plus a text instruction and predicts the
next arm actions. Its normal training objective is a flow-matching loss (call it L_flow) that teaches it to reproduce the actions in our 
recorded demonstrations. C2 takes the same internal image features, and a small second network head reconstructs a depth map of the scene. 
That gives a second loss, L_depth, and the network is trained on L = L_flow + depth_loss_weight * L_depth.

We change one thing only: the dataset is wrapped so that, alongside every frame, it also hands over that frame's depth target. 
LeRobot's policy code then finds it and adds L_depth automatically.

Example:
    torchrun --nproc_per_node=3 scripts/train_c2.py \
        --depth_repo_id=charansoma3001/smolvla-depth-maps \
        --dataset.repo_id=charansoma3001/depth-smolvla-blue-demos_20260603_154319 \
        --policy.type=smolvla \
        --policy.device=cuda \
        --policy.freeze_vision_encoder=false \
        --policy.depth_loss_weight=0.05 \
        --batch_size=22 --steps=20000 \
        --policy.scheduler_decay_steps=20000 \
        --save_freq=2000 \
        --output_dir=outputs/train/smolvla_c2_blue \
        --job_name=smolvla_c2_blue \
        --policy.repo_id=charansoma3001/smolvla_c2_blue \
        --wandb.enable=true --wandb.project=depth-smolvla
"""

import logging
import sys
import numpy as np
import torch
from datasets import load_dataset
import lerobot.scripts.lerobot_train as train_module

# Key under which the depth target is attached to each training item. The "observation." prefix matters: LeRobot's input pipeline passes anything with that
# prefix straight through to the policy, which is how the depth map reaches the point where the loss is computed.
OBS_DEPTH = "observation.depth"

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("train_c2")

class DepthAugmentedDataset(torch.utils.data.Dataset):
    """
    A thin shell around the real dataset that adds a depth target to every frame. The training loop only ever calls those two, so we can send our own
    object in front of the real dataset. We forward the count unchanged and, for each item, look up and attach its depth map.
    """

    def __init__(self, inner, depth_repo_id: str):
        self.inner = inner

        log.info("Loading depth targets from %s", depth_repo_id)
        depth_ds = load_dataset(depth_repo_id, split="train")

        # Hold every depth map in one contiguous NumPy array. Alongside it we build a plain dictionary mapping(episode, frame) to that map's row.
        self.depth = np.asarray(depth_ds["depth"], dtype=np.float32)  # (N, S, S)
        eps = depth_ds["episode_index"]
        frs = depth_ds["frame_index"]
        self.key_to_row = {(int(e), int(f)): i for i, (e, f) in enumerate(zip(eps, frs))}
        log.info("Loaded %d depth maps of shape %s", len(self.depth), self.depth.shape[1:])

    def __len__(self):
        return len(self.inner)

    def __getitem__(self, idx):
        # Ask the real dataset for the frame, then find its matching depth map by the (episode, frame) pair that identifies it, and attach it.
        item = self.inner[idx]
        ep = int(item["episode_index"])
        fr = int(item["frame_index"])
        row = self.key_to_row.get((ep, fr))
        if row is None:
            raise KeyError(f"No depth map for (episode={ep}, frame={fr}).")
        item[OBS_DEPTH] = torch.from_numpy(self.depth[row])  # (S, S) float32
        return item

    def __getattr__(self, name):
        # Python calls __getattr__ only for attributes this wrapper does not define itself. We forward those to the wrapped dataset, so any other property 
        # the training code reads (metadata, episode boundaries, and so on) transparently comes from the real dataset and our shell stays invisible to it.
        return getattr(self.inner, name)

def main():
    # LeRobot's argument parser does not know about our extra --depth_repo_id flag, so we pull it out of the command-line arguments and hand the rest along
    # untouched. sys.argv is the raw list of command-line tokens for this process.
    depth_repo_id = None
    remaining = []
    for arg in sys.argv:
        if arg.startswith("--depth_repo_id="):
            depth_repo_id = arg.split("=", 1)[1]
        else:
            remaining.append(arg)
    sys.argv = remaining

    if depth_repo_id is None:
        raise SystemExit("Missing required --depth_repo_id=<hf_repo> argument.")

    # "Monkeypatching": replacing a function in an already-imported module at runtime. LeRobot builds its dataset by calling make_dataset. We swap that function for one
    # that first calls the original and then wraps its result. From then on the training loop receives our depth-augmented dataset instead of the plain one.
    original_make_dataset = train_module.make_dataset

    def make_dataset_with_depth(cfg):
        dataset = original_make_dataset(cfg)
        return DepthAugmentedDataset(dataset, depth_repo_id)

    train_module.make_dataset = make_dataset_with_depth

    # Give control to LeRobot's training entry point. It reads the remaining command-line flags, sets up the model and the multi-GPU
    # distributed run, then trains, checkpoints and logs exactly as it does for C1.
    log.info("Starting C2 training (depth_repo_id=%s)", depth_repo_id)
    train_module.train()

if __name__ == "__main__":
    main()
