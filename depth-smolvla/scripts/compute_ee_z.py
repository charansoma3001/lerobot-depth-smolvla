"""
Release height of a placement, via forward kinematics on SO-101 joint angles
"""

import glob
import json
import os
import numpy as np
import pyarrow.parquet as pq
from lerobot.model.kinematics import RobotKinematics

# pick a recorded episode and the SO-101 arm model
DATASET = os.path.expanduser("~/.cache/huggingface/lerobot/charansoma3001/posctrl-acceptance-probe_20260616_141127")
# A URDF file is the robot's physical description: the length of each rigid link and how the joints connect them. Forward kinematics reads it to know the arm's geometry.
URDF = os.path.join(os.path.dirname(os.path.dirname(__file__)), "assets", "SO101", "so101_new_calib.urdf")
TRUE_HEIGHT_CM = 8.0  # the stack height this episode placed onto

# 1. load the joint trajectory
# observation.state is what the robot logged each frame: the six SO-101 joint angles in degrees. The trajectory is stored as Parquet
info = json.load(open(os.path.join(DATASET, "meta", "info.json")))
names = info["features"]["observation.state"]["names"]  # [..., 'gripper.pos']
table = pq.read_table(glob.glob(os.path.join(DATASET, "data", "**", "*.parquet"), recursive=True)[0])
state = np.array(table["observation.state"].to_pylist())  # shape [T, 6]
gripper = state[:, names.index("gripper.pos")]

# 2. forward kinematics: joint angles to get gripper height (z) for every frame
joints = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
arm = RobotKinematics(urdf_path=URDF, target_frame_name="gripper_frame_link", joint_names=joints)
# forward_kinematics returns a 4x4 transform (a way to bundle a rotation and a translation into one matrix). Its last column holds the gripper's (x, y, z) in metres, 
# so [2, 3] is the height z. We scale metres to centimetres.
z_cm = np.array([arm.forward_kinematics(q)[2, 3] for q in state]) * 100.0   # metres to cm

# 3. find the pickup (table grab) and the deposit (lowest point while placing)
# gripper.pos is high when open, low when closed. The first closing = grab the object off the table.
mid = (gripper.min() + gripper.max()) / 2.0
closes = np.where((gripper[:-1] >= mid) & (gripper[1:] < mid))[0] + 1  # closing events
pick = closes[0] # first close = pickup

# The object is now held. It is "released" when the gripper rises above its held level, counting even partial opens.
hold = np.median(gripper[pick:pick + 15])
release = next(f for f in range(pick + 10, len(gripper)) if gripper[f] > hold + max(6, 0.15 * (gripper.max() - gripper.min())))

# The deposit is the lowest the gripper gets while placing.
apex = pick + np.argmax(z_cm[pick:release])  # top of the carry
deposit = apex + np.argmin(z_cm[apex:release])  # lowest point of the placing descent

# 4. release height above the table = deposit z minus the pickup z (cancels grip offset)
# Subtracting the pickup height (gripper at the table) removes that constant offset, leaving how far above the table the object was actually set down.
release_height = z_cm[deposit] - z_cm[pick]
error = release_height - TRUE_HEIGHT_CM

print(f"pick frame {pick}: z = {z_cm[pick]:.2f} cm")
print(f"deposit frame {deposit}: z = {z_cm[deposit]:.2f} cm")
print(f"release height above table = {release_height:.2f} cm")
print(f"release-height error vs true {TRUE_HEIGHT_CM:.1f} cm = {error:+.2f} cm")
