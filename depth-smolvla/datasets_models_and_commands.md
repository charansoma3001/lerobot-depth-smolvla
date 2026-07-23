# Datasets, models, and commands

A record of the HuggingFace datasets and model checkpoints produced for this project,
and the commands used to record data, train the policies, run the evaluation rollouts,
and analyse them. All under the HuggingFace account `charansoma3001`.

Robot commands ran from `/Users/charan/Repos/lerobot` on the Mac (robot client).

## HuggingFace datasets

| Repo ID | Contents |
|---------|----------|
| `charansoma3001/posctrl-full2-demos_20260622_181834` | Main experiment. 100 free-space placement demos (50 at 3 cm, 50 at 8 cm). |
| `charansoma3001/posctrl-full2-depth-maps` | Depth Anything V2 depth targets (64x64) for the main-experiment demos, used to train C2. |
| `charansoma3001/depth-smolvla-blue-demos_20260603_154319` | Preliminary pick-and-place task. 100 teleoperated demos of the blue object. |
| `charansoma3001/smolvla-depth-maps` | Depth targets for the preliminary-task demos. |

## HuggingFace models

| Repo ID | Role |
|---------|------|
| `charansoma3001/smolvla_posctrl_c1_full2` | Main experiment C1 Baseline (also reused for H2). |
| `charansoma3001/smolvla_posctrl_c2_full2` | Main experiment C2 Implicit-depth (also reused for H2). |
| `charansoma3001/smolvla_c1_blue_baseline` | Preliminary task C1 baseline. |
| `charansoma3001/smolvla_c2_blue_lambda05` | Preliminary task C2 (lambda = 0.05). |

## Evaluation rollouts

Each rollout dataset holds 10 rollouts with the joint trajectory logged for forward
kinematics (the release-height metric).

Main experiment (positive control), on HuggingFace, one per policy per height:

- `charansoma3001/rollout_posctrl-full-c1-4cm_20260625_112340`
- `charansoma3001/rollout_posctrl-full-c1-5cm_20260625_115349`
- `charansoma3001/rollout_posctrl-full-c1-6cm_20260625_121400`
- `charansoma3001/rollout_posctrl-full-c1-7cm_20260625_134813`
- `charansoma3001/rollout_posctrl-full-c2-4cm_20260625_113337`
- `charansoma3001/rollout_posctrl-full-c2-5cm_20260625_120331`
- `charansoma3001/rollout_posctrl-full-c2-6cm_20260625_122831`
- `charansoma3001/rollout_posctrl-full-c2-7cm_20260625_133315`

H2 visual shift (6 cm), on HuggingFace, one per policy per condition:

- `charansoma3001/rollout_h2-eval-c1-6cm-A0_20260721_154458`
- `charansoma3001/rollout_h2-eval-c1-6cm-V2_20260721_171340`
- `charansoma3001/rollout_h2-eval-c1-6cm-V1b_20260721_182013`
- `charansoma3001/rollout_h2-eval-c1-6cm-V1_20260721_162347`
- `charansoma3001/rollout_h2-eval-c1-6cm-SHADOW_20260721_184627`
- `charansoma3001/rollout_h2-eval-c2-6cm-A0_20260721_160456`
- `charansoma3001/rollout_h2-eval-c2-6cm-V2_20260721_170219`
- `charansoma3001/rollout_h2-eval-c2-6cm-V1b_20260721_180234`
- `charansoma3001/rollout_h2-eval-c2-6cm-V1_20260721_161818`
- `charansoma3001/rollout_h2-eval-c2-6cm-SHADOW_20260721_185348`

The condition codes in the names are historical and do not read in light order:
V2 = 550 lux, V1b = 475 lux, V1 = 315 lux (A0 = clean 775 lux reference; SHADOW =
directional shadow at about A0 brightness).

## Commands

### 1. Record demonstrations (Mac, teleoperation)

```bash
uv run lerobot-record \
  --robot.type=so101_follower --robot.port=/dev/tty.usbmodem5B421390331 --robot.id=follower \
  --teleop.type=so101_leader --teleop.port=/dev/tty.usbmodem5B421392361 --teleop.id=leader \
  --robot.cameras="{overhead: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30}}" \
  --dataset.repo_id=charansoma3001/posctrl-full2-demos_20260622_181834 \
  --dataset.single_task="Pick up the blue object and place it on the stack" \
  --dataset.num_episodes=100 --dataset.episode_time_s=60 --dataset.reset_time_s=20 \
  --display_data=true
```

### 2. Generate depth targets (server, GPU)

```bash
uv run python scripts/preprocess_depth.py \
  --dataset_repo_id=charansoma3001/posctrl-full2-demos_20260622_181834 \
  --output_repo_id=charansoma3001/posctrl-full2-depth-maps \
  --depth_size=64 --batch_size=32 --device=cuda --push_to_hub
```

### 3. Train C1, RGB-only baseline (server, 3 GPUs)

```bash
torchrun --nproc_per_node=3 -m lerobot.scripts.lerobot_train \
  --dataset.repo_id=charansoma3001/posctrl-full2-demos_20260622_181834 \
  --policy.type=smolvla --policy.device=cuda --policy.freeze_vision_encoder=false \
  --batch_size=22 --steps=20000 --policy.scheduler_decay_steps=20000 --save_freq=2000 \
  --output_dir=outputs/train/smolvla_posctrl_c1_full2 \
  --job_name=smolvla_posctrl_c1_full2 \
  --policy.repo_id=charansoma3001/smolvla_posctrl_c1_full2 \
  --wandb.enable=true --wandb.project=depth-smolvla
```

### 4. Train C2, auxiliary depth (server, 3 GPUs)

`train_c2.py` wraps the standard training entry point and adds the depth target to each
batch (`observation.depth`), which drives the combined loss `L = L_flow + lambda * L_depth`.

```bash
torchrun --nproc_per_node=3 scripts/train_c2.py \
  --depth_repo_id=charansoma3001/posctrl-full2-depth-maps \
  --dataset.repo_id=charansoma3001/posctrl-full2-demos_20260622_181834 \
  --policy.type=smolvla --policy.device=cuda --policy.freeze_vision_encoder=false \
  --policy.depth_loss_weight=0.05 \
  --batch_size=22 --steps=20000 --policy.scheduler_decay_steps=20000 --save_freq=2000 \
  --output_dir=outputs/train/smolvla_posctrl_c2_full2 \
  --job_name=smolvla_posctrl_c2_full2 \
  --policy.repo_id=charansoma3001/smolvla_posctrl_c2_full2 \
  --wandb.enable=true --wandb.project=depth-smolvla
```

### 5. Evaluation rollout (Mac, sync inference)

`--strategy.type=episodic` is required so the joint trajectory is saved for forward
kinematics; `--strategy.type=base` records nothing. Run once per height (4/5/6/7 cm),
swapping `--policy.path` between the C1 and C2 checkpoints.

```bash
uv run lerobot-rollout \
  --strategy.type=episodic \
  --policy.path=charansoma3001/smolvla_posctrl_c1_full2 \
  --robot.type=so101_follower --robot.port=/dev/tty.usbmodem5B421390331 --robot.id=follower \
  --robot.cameras="{overhead: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30}}" \
  --task="Pick up the blue object and place it on the stack" \
  --dataset.repo_id=charansoma3001/rollout_posctrl-full-c1-6cm \
  --dataset.num_episodes=10 --dataset.episode_time_s=60 \
  --dataset.streaming_encoding=true \
  --display_data=true
```

For the H2 test the same command was run at 6 cm only, changing the visual condition
between rollouts and naming the dataset `...h2-eval-c1-6cm-<A0|V1|V2|SHADOW>`.

### 6. Analysis

Release height per episode by forward kinematics, then the significance tests:

```bash
python scripts/compute_ee_z.py --dataset <local rollout dir> --episode N --true-height-cm 6.0
python scripts/posctrl_stats.py
python scripts/h2_stats.py
```
