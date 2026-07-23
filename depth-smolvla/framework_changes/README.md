# Framework changes: the C2 auxiliary depth head

The C1 baseline runs the SmolVLA policy unchanged from the LeRobot framework. C2 adds an auxiliary depth-reconstruction head that is trained alongside the policy and then discarded at inference, so C1 and C2 have the same deployment cost.

That head lives inside SmolVLA's own model code. This folder holds a diff rather than the whole files. LeRobot's `modeling_smolvla.py` is around 800 lines of framework code; only about 90 of them are mine, and a diff shows exactly which.

## The file

- `c2_depth_head.diff` - the change applied to two files in a fresh fork of LeRobot.
  Base commit `73dbb6f4` (`refactor(smolvla): reuse shared VLA components`, 22 Jul 2026), from `github.com/huggingface/lerobot`. Apply with `git apply c2_depth_head.diff` from the repository root.

The same change is also on GitHub, as a fork with the diff committed: https://github.com/charansoma3001/lerobot-depth-smolvla

## What the diff does

The change is L_total = L_flow + lambda * L_depth: keep SmolVLA's flow-matching action loss, and add a small depth-prediction loss on top, weighted by lambda.

`configuration_smolvla.py`
- Adds two settings: `depth_loss_weight` (the lambda; 0.0 disables the head and recovers C1) and `depth_target_size` (the 64x64 depth-map resolution).

`modeling_smolvla.py`
- `DepthReconstructionHead`: the new module. It reads the 64 image tokens SmolVLA already produces per frame (an 8x8 grid), maps each to one depth value with a single linear layer, and upsamples the 8x8 result to 64x64.
- The head is built only when `depth_loss_weight > 0`, so C1 gains no extra parameters.
- `embed_prefix` now also returns the image tokens so the head can read them.
- `VLAFlowMatching.forward` predicts the depth map and returns it next to the action loss (returns `None` for C1).
- `SmolVLAPolicy.forward` computes the depth loss (mean squared error against the depth target in the batch) and adds `lambda * L_depth` to the flow loss. Both terms are logged separately.
