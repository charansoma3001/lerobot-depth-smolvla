# Evidence

Supporting code and data for the dissertation:

***Enabling 3D Spatial Reasoning in Vision Language Action Models*** <br>
Charan Sai Soma (H00520704) <br>
Supervisor: Dr. Ioannis Konstas <br>
MSc Artificial Intelligence, Heriot-Watt University (F21MP)

These files are provided as evidence of the work described in the report. The two statistics scripts read their data from the CSV files next to them, so the reported numbers can be recomputed.

## What each file is

| File | What it does |
|------|--------------|
| `scripts/preprocess_depth.py` | Runs Depth Anything V2 over every demonstration frame and saves a 64x64 depth map per frame, used as the depth target during training. |
| `scripts/train_c2.py` | Trains the SmolVLA policy with the auxiliary depth-reconstruction loss `L = L_flow + lambda * L_depth`. |
| `scripts/compute_ee_z.py` | Computes the release height of a placement by forward kinematics on the recorded SO-101 joint angles. This is the evaluation metric. |
| `scripts/time_inference.py` | Measures time for one forward pass for the Baseline and Implicit-depth policies, to show same inference cost |
| `scripts/posctrl_stats.py` | Fisher exact, stratified permutation, and Pearson tests for the positive control. Reads `posctrl_full2_episodes.csv`. |
| `scripts/h2_stats.py` | Per-condition Fisher and permutation tests, plus the H2 difference-in-differences test. Reads `h2_visual_shift_episodes.csv`. |
| `assets/SO101/so101_new_calib.urdf` | The SO-101 arm description used by the forward-kinematics metric. |
| `framework_changes/c2_depth_head.diff` | The change made to the LeRobot framework to add the C2 auxiliary depth head, as a diff. See `framework_changes/README.md`. |
| `datasets_models_and_commands.md` | The HuggingFace datasets and model checkpoints for the project, and the commands used to record data, train, run rollouts, and analyse them. |
| `video_evidence/` | Side-view recordings of the release height for the Baseline and Implicit-depth policies at 4/5/6/7 cm, plus a `preliminary_task_failure/` folder with the two 9 cm clips from the preliminary task. |
| `h2_shadow_images/` | Photographs of the directional-shadow (SHADOW) condition used in the H2 visual-shift test, at rest and mid-motion. |

## Data files

- `scripts/posctrl_full2_episodes.csv` - one row per positive-control trial: `variant, height, episode, signed_err, success`. A blank `signed_err` is a grasp failure with no release; it counts as a failed trial but is left out of the error tests.
- `scripts/h2_visual_shift_episodes.csv` - one row per visual-shift trial: `variant, condition, episode, signed_err, success`. Conditions are A0 (clean reference), L550 (lower ambient light), and SHADOW (directional off-axis shadow).

## Reproducing the statistics

Both statistics scripts are self-contained with their CSV:

```
uv run --with scipy --with numpy python scripts/posctrl_stats.py
uv run --with scipy --with numpy python scripts/h2_stats.py
```

Expected positive-control output:

```
Fisher exact (two-sided): baseline 30/40 vs implicit 40/40, p = 0.0010
Release error, observed mean|err| diff = +0.241 cm
permutation: p = 0.0540 (one-sided), 0.0921 (two-sided)
Pearson r vs height: baseline -0.159, implicit +0.308
```

The permutation p-values use 10,000 random shuffles with a fixed seed, so they can vary in the last digit on other machines. The conclusions do not.

## Notes

- `compute_ee_z.py` reads a recorded episode from a local dataset cache and needs the arm's mesh files to load the kinematics backend. Neither is bundled here. The arm description (URDF) is kept because it is the geometry the metric relies on.
- The URDF was exported from the SO-101 arm's CAD model. `_new_calib` is the calibrated version.
