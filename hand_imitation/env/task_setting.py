import sapien.core as sapien
import transforms3d
import numpy as np
from hand_imitation.real_world import lab

ROBOT_TABLE_MARGIN_X = 0.06
ROBOT_TABLE_MARGIN_Y = 0.04


ROBUSTNESS_INIT_CAMERA_CONFIG = {
    'laptop': {'r': 1, 'phi': np.pi / 2, 'theta': np.pi / 2,
               'center': np.array([0, 0, 0.5])},
}

TRAIN_CONFIG = {
    "faucet": {
        'seen': [148, 693, 822, 857, 991, 1011, 1053, 1288, 1343, 1370, 1466],
        'unseen': [1556, 1633, 1646, 1667, 1741, 1832, 1925]
    },
    "faucet_half": {
        'seen': [148, 693, 822, 857, 991],
        'unseen': [1556, 1633, 1646, 1667, 1741, 1832, 1925]
    },
    "bucket": {
        'seen': [100431, 100435, 100438, 100439, 100441, 100444, 100446, 100448, 100454, 100461, 100462],
        'unseen': [100468, 100470, 100473, 100482, 100484, 100486, 102352, 102358]
    },
    "bucket_half": {
        'seen': [100431, 100435, 100438, 100439, 100441],
        'unseen': [100468, 100470, 100473, 100482, 100484, 100486, 102352, 102358]
    },
    "laptop": {
        'seen': [11395, 11405, 11406, 11477, 11581, 11586, 9996, 10090, 10098, 10101, 10125],
        'unseen': [9748, 9912, 9918, 9960, 9968, 9992],
    },
    "laptop_half": {
        'seen': [11395, 11405, 11406, 11477, 11581],
        'unseen': [9748, 9912, 9918, 9960, 9968, 9992],
    },
    "toilet": {
        'seen': [102677, 102687, 102689, 102692, 102697, 102699, 102701, 102703, 102707, 102708, 103234, 102663, 102666,
                 102667, 102669, 102670, 102675],
        'unseen': [101320, 102621, 102622, 102630, 102634, 102645, 102648, 102651, 102652, 102654, 102658],
    },
    "toilet_half": {
        'seen': [102677, 102687, 102689, 102692, 102697, 102699, 102701, 102703],
        'unseen': [101320, 102621, 102622, 102630, 102634, 102645, 102648, 102651, 102652, 102654, 102658],
    },
}

TASK_CONFIG = {
    "faucet": [148, 693, 822, 857, 991, 1011, 1053, 1288, 1343, 1370, 1466, 1556, 1633, 1646, 1667, 1741, 1832, 1925,
               ],
    "bucket": [100431, 100435, 100438, 100439, 100441, 100444, 100446, 100448, 100454, 100461,
               100462, 100468, 100470, 100473, 100482, 100484, 100486, 102352, 102358,
               ],
    "laptop": [9748, 9912, 9918, 9960, 9968, 9992, 11395, 11405, 11406, 11477, 11581, 11586, 9996, 10090, 10098, 10101, 10125],
    "toilet": [101320, 102621, 102622, 102630, 102634, 102645, 102648, 102651, 102652, 102654,
                102658, 102677, 102687, 102689, 102692, 102697, 102699, 102701, 102703, 102707,
               102708, 103234, 102663, 102666, 102667, 102669, 102670, 102675],
}

# Camera config
CAMERA_CONFIG = {
    "relocate": {
        "instance_1": dict(
            pose=sapien.Pose(p=np.array([0.309377, 0.988342, 0.766590]), q=lab.CAM_Q),
            fov=np.deg2rad(42.5), resolution=(640, 360)
            ),
    },
    "viz_only": {  # only for visualization (human), not for visual observation
        "relocate_viz": dict(
            pose=sapien.Pose(p=np.array([0.5, 1.2, 0.5]), q=transforms3d.euler.euler2quat(np.pi / 2, np.pi, 0)),
            fov=np.deg2rad(69.4), resolution=(640, 480)),
    },
}

EVAL_CAM_NAMES_CONFIG = {
    "faucet": ["faucet_viz"],
    "bucket": ['bucket_viz'],
    "laptop": ['laptop_viz'],
    "toilet": ['toilet_viz'],
}

# Observation config type
OBS_CONFIG = {
    "instance_rgb": {
        "instance_1": {"rgb": True},
    },
    "instance": {
        "instance_1": {"point_cloud": {"num_points": 2048, "use_seg": False}},
    },
    "instance_real": {
        "instance_1": {"point_cloud": {"num_points": 512}},
    },
    "instance_noise": {
        "instance_1": {
            "point_cloud": {"num_points": 512, "pose_perturb_level": 0.5,
                            "process_fn_kwargs": {"noise_level": 0.5}},
        },
    },
    "instance_pc_seg": {
        "instance_1": {
            "point_cloud": {"use_seg": True, "use_2frame": True, "num_points": 512, "pose_perturb_level": 0.5,
                            "process_fn_kwargs": {"noise_level": 0.5}},
        },
    },
}

IMG_CONFIG = {
    "robot": {
        "robot": {
            "right_base_link_inertia": 8, "right_shoulder_link": 8, "right_upper_arm_link": 8, "right_forearm_link": 8, "right_wrist_1_link": 8, "right_wrist_2_link": 8, "right_wrist_3_link": 8, 
            "right_gripper_link_15_tip": 8, "right_gripper_link_03_tip": 8, "right_gripper_link_07_tip": 8, "right_gripper_link_11_tip": 8, "right_gripper_link_15": 8, "right_gripper_link_03": 8, "right_gripper_link_07": 8, "right_gripper_link_11": 8, "right_gripper_link_14": 8, "right_gripper_link_02": 8, "right_gripper_link_06": 8, "right_gripper_link_10": 8},
    },
}

RANDOM_CONFIG = {"bucket": {"rand_pos": 0.05, "rand_degree": 0}, "laptop": {"rand_pos": 0.1, "rand_degree": 60},
                 "faucet": {"rand_pos": 0.1, "rand_degree": 90}, 'toilet': {"rand_pos": 0.2, "rand_degree": 45},}
