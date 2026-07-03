import numpy as np
from sapien.core import Pose
from transforms3d.euler import euler2quat

CAM2ROBOT = Pose.from_transformation_matrix(np.array(
    [[0.60346958, 0.36270068, -0.7101216, 0.962396],
     [0.7960018, -0.22156729, 0.56328419, -0.35524235],
     [0.04696384, -0.90518294, -0.42241951, 0.31896536],
     [0., 0., 0., 1.]]
))

DESK2ROBOT_Z_AXIS = 0.00

# Relocate
RELOCATE_BOUND = [0.0, 1.125, -0.5, 0.5, DESK2ROBOT_Z_AXIS + 0.005, 1.0]

# TODO:
ROBOT2BASE = Pose(p=np.array([0.765, -0.09, -DESK2ROBOT_Z_AXIS]))
CAM_ROT = [-2.270, 0.049, -3.050]
CAM_Q = euler2quat(CAM_ROT[0], CAM_ROT[1], CAM_ROT[2])

# Table size
TABLE_XY_SIZE = np.array([1.125, 1.08])
TABLE_ORIGIN = np.array([0.5625, 0])

# original table setting
ORI_DESK2ROBOT_Z_AXIS = -0.05

ORI_RELOCATE_BOUND = [0.0, 1.2, -0.6, 0.6, ORI_DESK2ROBOT_Z_AXIS + 0.005, 0.6]

ORI_ROBOT2BASE = Pose(p=np.array([-0.55, 0., -ORI_DESK2ROBOT_Z_AXIS]))

ORI_TABLE_XY_SIZE = np.array([0.6, 1.2])
ORI_TABLE_ORIGIN = np.array([0, -0.15])

