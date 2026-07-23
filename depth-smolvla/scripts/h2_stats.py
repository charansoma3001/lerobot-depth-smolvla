"""
H2 visual-shift statistics. Two things are measured per condition:
  - TSR, the task success rate (how many of the 10 trials placed the object), and
  - the release-height error on the trials that did succeed.
The data is in h2_visual_shift_episodes.csv, one row per trial. A0 is the clean reference; L550 (dimmer light) and SHADOW (a directional shadow) are the two shifts.
"""

import csv
import os
import numpy as np
from scipy.stats import fisher_exact

CSV = os.path.join(os.path.dirname(__file__), "h2_visual_shift_episodes.csv")
rng = np.random.default_rng(0)
N_PERM = 10000

# Read the CSV into data[condition][variant]: succ = one 1/0 success flag per trial, err = the errors on the successful trials only.
data = {}
for row in csv.DictReader(open(CSV)):
    cond, var = row["condition"], row["variant"]
    data.setdefault(cond, {}).setdefault(var, {"succ": [], "err": []})
    data[cond][var]["succ"].append(int(row["success"]))
    if row["signed_err"] != "":
        data[cond][var]["err"].append(float(row["signed_err"]))


def perm_test(a, b):
    """
    Permutation test: We take the real difference in means, then repeatedly shuffle all the values between the two groups at random and re-measure. 
    The p-value is the fraction of shuffles whose gap is at least as big as the real one; a small value means the real gap is unlikely to be chance.
    """
    a, b = np.abs(a), np.abs(b)
    observed = a.mean() - b.mean()
    pool = np.concatenate([a, b])
    hits = 0
    for _ in range(N_PERM):
        rng.shuffle(pool)
        if abs(pool[:len(a)].mean() - pool[len(a):].mean()) >= abs(observed) - 1e-12:
            hits += 1
    return observed, hits / N_PERM


# 1. Per-condition: check if Baseline different from Implicit within each lighting condition
for cond in ["A0", "L550", "SHADOW"]:
    b, i = data[cond]["baseline"], data[cond]["implicit"]
    # Fisher exact test: the standard test for a 2x2 table of successes vs failures.
    table = [[sum(b["succ"]), len(b["succ"]) - sum(b["succ"])],
             [sum(i["succ"]), len(i["succ"]) - sum(i["succ"])]]
    p_tsr = fisher_exact(table)[1]
    diff, p_err = perm_test(np.array(b["err"]), np.array(i["err"]))
    print(f"\n{cond}: TSR baseline {sum(b['succ'])}/10, implicit {sum(i['succ'])}/10  (Fisher p={p_tsr:.4f})")
    print(f"mean |error| baseline {np.abs(b['err']).mean():.2f}, implicit {np.abs(i['err']).mean():.2f}  (permutation p={p_err:.4f})")


# 2. The H2 test: check if Implicit degrade LESS than Baseline from A0 to each shift
# We compare the two policies' degradations with a difference-in-differences. To get a p-value we shuffle the policy labels within each condition and see how often the
# shuffled gap beats the real one.
for shift in ["L550", "SHADOW"]:
    # TSR degradation = drop in successes from the clean condition to the shifted one.
    b_drop = sum(data["A0"]["baseline"]["succ"]) - sum(data[shift]["baseline"]["succ"])
    i_drop = sum(data["A0"]["implicit"]["succ"]) - sum(data[shift]["implicit"]["succ"])
    observed = i_drop - b_drop

    # Shuffle: pool the two policies' trials in each condition, and compute again.
    a0 = data["A0"]["baseline"]["succ"] + data["A0"]["implicit"]["succ"]
    sh = data[shift]["baseline"]["succ"] + data[shift]["implicit"]["succ"]
    n = len(data["A0"]["baseline"]["succ"])
    hits = 0
    for _ in range(N_PERM):
        rng.shuffle(a0)
        rng.shuffle(sh)
        pb = sum(a0[:n]) - sum(sh[:n])
        pi = sum(a0[n:]) - sum(sh[n:])
        if abs(pi - pb) >= abs(observed) - 1e-12:
            hits += 1

    print(f"\nA0 to {shift}: baseline lost {b_drop}/10, implicit lost {i_drop}/10 (permutation p={hits / N_PERM:.4f})")
