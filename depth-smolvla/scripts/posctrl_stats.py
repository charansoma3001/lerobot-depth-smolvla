"""
The positive control checks that the depth-trained policy (implicit) is genuinely better than the plain one (baseline) at placing the object at a target height. Three tests, on
the per-trial data in posctrl_full2_episodes.csv (one row per trial, across four target heights 3.8, 4.9, 6.0, 6.8 cm):

  1. Fisher exact test on task success (did the object successfully place).
  2. A permutation test on the release-height error of the successful placements.
  3. Pearson correlation of that error against the target height.

A blank signed_err in the CSV is a grasp failure (nothing was released), counted as a failed trial but left out of the error tests.
"""

import csv
import os
import numpy as np
from scipy.stats import fisher_exact, pearsonr

CSV = os.path.join(os.path.dirname(__file__), "posctrl_full2_episodes.csv")
rng = np.random.default_rng(0)
N_PERM = 10000
# Read the CSV. err[variant][height] holds the signed errors of released trials only, success and n are counted every trial so we can build the success table.
err = {"baseline": {}, "implicit": {}}
success = {"baseline": 0, "implicit": 0}
n = {"baseline": 0, "implicit": 0}
for row in csv.DictReader(open(CSV)):
    v, h = row["variant"], float(row["height"])
    success[v] += int(row["success"])
    n[v] += 1
    err[v].setdefault(h, [])
    if row["signed_err"] != "":
        err[v][h].append(float(row["signed_err"]))

heights = sorted(err["baseline"])

# 1. Fisher exact test: the standard test for a 2x2 table of successes vs failures.
table = [[success["baseline"], n["baseline"] - success["baseline"]],
         [success["implicit"], n["implicit"] - success["implicit"]]]
p_fisher = fisher_exact(table)[1]


# 2. Permutation test on the placement error (mean of |error|, baseline minus implicit).
# The two variants were run back-to-back at each height, so we shuffle the baseline/implicit labels within each height rather than across everything.
b_all = np.array([abs(e) for h in heights for e in err["baseline"][h]])
i_all = np.array([abs(e) for h in heights for e in err["implicit"][h]])
observed = b_all.mean() - i_all.mean()

combined = {h: np.abs(err["baseline"][h] + err["implicit"][h]) for h in heights}
n_base = {h: len(err["baseline"][h]) for h in heights}

hits_two = hits_one = 0
for _ in range(N_PERM):
    b_sum = i_sum = b_ct = i_ct = 0
    for h in heights:
        shuffled = combined[h].copy()
        rng.shuffle(shuffled)
        k = n_base[h]
        b_sum += shuffled[:k].sum(); b_ct += k
        i_sum += shuffled[k:].sum(); i_ct += len(shuffled) - k
    stat = b_sum / b_ct - i_sum / i_ct
    hits_two += abs(stat) >= abs(observed) - 1e-12
    hits_one += stat >= observed - 1e-12

# add-one so a p-value is never reported as exactly zero
p_two = (hits_two + 1) / (N_PERM + 1)
p_one = (hits_one + 1) / (N_PERM + 1)

# 3. Pearson correlation of signed error against target height, per variant. A positive
# correlation means the policy overshoots more as the target gets taller.
def corr_vs_height(variant):
    xs = [h for h in heights for _ in err[variant][h]]
    ys = [e for h in heights for e in err[variant][h]]
    return pearsonr(xs, ys)[0]


rb = corr_vs_height("baseline")
ri = corr_vs_height("implicit")


print(f"Fisher exact (two-sided): baseline {success['baseline']}/{n['baseline']} vs implicit {success['implicit']}/{n['implicit']}, p = {p_fisher:.4f}")
print(f"Release error, observed mean|err| diff = {observed:+.3f} cm")
print(f"permutation: p = {p_one:.4f} (one-sided), {p_two:.4f} (two-sided)")
print(f"Pearson r vs height: baseline {rb:+.3f}, implicit {ri:+.3f}")
