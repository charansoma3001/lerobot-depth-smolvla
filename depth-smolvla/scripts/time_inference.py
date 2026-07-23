"""
Measures inference latency of one full forward pass for Baseline (C1) and Implicit depth (C2), on a dummy observation, and prints per-forward ms and the per-chunk Hz.
"""

import time

import torch

from lerobot.policies.smolvla.modeling_smolvla import (
    OBS_LANGUAGE_ATTENTION_MASK,
    OBS_LANGUAGE_TOKENS,
    OBS_STATE,
    SmolVLAPolicy,
)

MODELS = {
    "Baseline (C1)": "charansoma3001/smolvla_posctrl_c1_full2",
    "Implicit depth (C2)": "charansoma3001/smolvla_posctrl_c2_full2",
}
# The first few forward passes are slow: the framework allocates buffers the first time they run. We do WARMUP passes to account for that cost, then time ITERS real passes and average them.
WARMUP, ITERS = 10, 50

# MPS is PyTorch's backend for the GPU built into Apple Silicon Macs; fall back to the CPU if it is unavailable. (This machine has no CUDA GPU.)
device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"device: {device}\n")

for label, path in MODELS.items():
    # Load the trained policy weights from the Hugging Face hub and move them to the device.
    policy = SmolVLAPolicy.from_pretrained(path).to(device)
    policy.eval()
    cfg = policy.config

    # Dummy observation. We are timing raw compute, not behaviour, so random tensors of the correct shape and dtype are sufficient.
    batch = {}
    for key, feat in cfg.image_features.items():
        c, h, w = feat.shape
        batch[key] = torch.rand(1, c, h, w, device=device)
    batch[OBS_STATE] = torch.rand(1, cfg.robot_state_feature.shape[0], device=device)
    batch[OBS_LANGUAGE_TOKENS] = torch.zeros(1, 48, dtype=torch.long, device=device)
    batch[OBS_LANGUAGE_ATTENTION_MASK] = torch.ones(1, 48, dtype=torch.bool, device=device)

    for _ in range(WARMUP):  # warm up
        policy.reset()
        policy.predict_action_chunk(batch)

    # synchronize() blocks until the queued work is done, so we call it just before starting the clock and again before stopping it; otherwise we would not compute timing for 
    # the computation. perf_counter is a high-resolution timer.
    times = []
    for _ in range(ITERS):
        policy.reset()
        if device == "mps":
            torch.mps.synchronize()
        t0 = time.perf_counter()
        policy.predict_action_chunk(batch)
        if device == "mps":
            torch.mps.synchronize()
        times.append((time.perf_counter() - t0) * 1000)

    # predict_action_chunk returns a whole block of future actions from one forward pass, so "Hz/chunk" is how many such blocks the policy can produce per second
    ms = sum(times) / len(times)
    print(f"{label:<22} {ms:7.1f} ms/forward   {1000/ms:.2f} Hz/chunk (chunk={cfg.n_action_steps})")
