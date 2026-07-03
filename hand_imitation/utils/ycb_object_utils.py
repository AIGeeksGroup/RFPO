from pathlib import Path
import os

import numpy as np
import sapien.core as sapien


def get_ycb_root_dir():
    current_dir = Path(__file__).parent
    ycb_dir = current_dir.parent.parent / "assets" / "ycb"
    return ycb_dir.resolve()


YCB_CLASSES = {
    1: '002_master_chef_can',
    2: '003_cracker_box',
    3: '004_sugar_box',
    4: '005_tomato_soup_can',
    5: '006_mustard_bottle',
    6: '007_tuna_fish_can',
    7: '008_pudding_box',
    8: '009_gelatin_box',
    9: '010_potted_meat_can',
    10: '011_banana',
    11: '019_pitcher_base',
    12: '021_bleach_cleanser',
    13: '024_bowl',
    14: '025_mug',
    15: '035_power_drill',
    16: '036_wood_block',
    17: '037_scissors',
    18: '040_large_marker',
    19: '051_large_clamp',
    20: '052_extra_large_clamp',
    21: '061_foam_brick',
}

YCB_SIZE = {
    "master_chef_can": (0.1025, 0.1023, 0.1401),
    "cracker_box": (0.2134, 0.1640, 0.0717),
    "tuna_fish_can": (0.2134, 0.1640, 0.0717),
    "pudding_box": (0.2134, 0.1640, 0.0717),
    "gelatin_box": (0.2134, 0.1640, 0.0717),
    "sugar_box": (0.0495, 0.0940, 0.1760),
    "tomato_soup_can": (0.0677, 0.0679, 0.1018),
    "mustard_bottle": (0.0576, 0.0959, 0.1913),
    "potted_meat_can": (0.0576, 0.1015, 0.0835),
    "banana": (0.1088, 0.1784, 0.0366),
    "pitcher_base": (0.1024, 0.0677, 0.2506),
    "bleach_cleanser": (0.1024, 0.0677, 0.2506),
    "bowl": (0.1614, 0.1611, 0.0550),
    "mug": (0.1169, 0.0930, 0.0813),
    "wood_block": (0.1169, 0.0930, 0.0813),
    "large_clamp": (0.1659, 0.1216, 0.0364),
    "extra_large_clamp": (0.1659, 0.1216, 0.0364),
    "foam_brick": (0.1659, 0.1216, 0.0364),
}

YCB_ORIENTATION = {
    "master_chef_can": (1, 0, 0, 0),
    "cracker_box": (1, 0, 0, 0),
    "tuna_fish_can": (1, 0, 0, 0),
    "pudding_box": (1, 0, 0, 0),
    "gelatin_box": (1, 0, 0, 0),
    "sugar_box": (1, 0, 0, 0),
    "tomato_soup_can": (1, 0, 0, 0),
    "mustard_bottle": (0.9659, 0, 0, 0.2588),
    "potted_meat_can": (1, 0, 0, 0),
    "banana": (1, 0, 0, 0),
    "pitcher_base": (1, 0, 0, 0),
    "bleach_cleanser": (1, 0, 0, 0),
    "bowl": (1, 0, 0, 0),
    "mug": (1, 0, 0, 0),
    "wood_block": (1, 0, 0, 0),
    "large_clamp": (0, 0, 0, 1),
    "extra_large_clamp": (0, 0, 0, 1),
    "foam_brick": (1, 0, 0, 0),
}


YCB_ORIENTATION_UNSEEN = {
    "master_chef_can": [(1, 0, 0, 0), (0.12027103, 0.45452249, -0.5335682, -0.70302856), (0.19333922, 0.52328889, 0.47366695, 0.68148977)],
    "tuna_fish_can": [(0.3494909, 0.1472301, 0.68593786, 0.62102227), (0.17424509, 0.60444172, -0.29906572, 0.71752948), (0.20272036, 0.53268835, 0.42021453, -0.706093)],
    "pudding_box": [(0.09068191, -0.25823498, 0.6855793, 0.67459062), (0.12695425, -0.56293949, -0.42054814, -0.70008643), (0.70269325, 0.38668464, -0.59422202, 0.0599781)],
    "gelatin_box": [(0.68327578, -0.28880612, 0.66258229, 0.10348886), (0.47786957, 0.57276809, -0.2893157, 0.59989483), (0.64143463, -0.02496997, -0.01550254, 0.7666145)],
    "potted_meat_can": [(1, 0, 0, 0), (0.23321052, -0.64450454, -0.28572726, -0.66976614), (0.6487909, 0.28179533, 0.63069786, -0.31918956)],
    "banana": [(0.01708955, -0.99206736, 0.12339044, -0.01688462), (0.01631956, 0.65687861, 0.75352605, -0.02103946), (0.83246721, -0.00176304, -0.01220784, -0.553937)],
    "pitcher_base": [(1, 0, 0, 0), (0.03543607, 0.6109757, -0.33756347, 0.715195), (0.66770816, -0.01827821, 0.03174889, 0.74352117)],
    "bleach_cleanser": [(1, 0, 0, 0), (0.69672928, 0.23661359, 0.64575008, -0.20393417), (0.45180177, 0.63545715, 0.33115619, 0.53141787)],
    "wood_block": [(1, 0, 0, 0), (0.65389746, 0.66045802, 0.26496458, 0.25691845), (0.45180177, 0.63545715, 0.33115619, 0.53141787)],
    "foam_brick": [(1, 0, 0, 0), (0.12629753, 0.14283687, 0.69497909, 0.69328971), (0.03543607, 0.6109757, -0.33756347, 0.715195)],
}

YCB_HEIGHT_UNSEEN = {
    "master_chef_can": [0.07547712326049805, 0.05100474879145622, 0.05100474879145622],
    "tuna_fish_can": [0.042931679636240005, 0.042931679636240005, 0.042931679636240005],
    "pudding_box": [0.04558473825454712, 0.05777805298566818, 0.05777805298566818],
    "gelatin_box": [0.03749748691916466, 0.0370822474360466, 0.015252859331667423],
    "potted_meat_can": [0.05057743191719055, 0.052958279848098755, 0.048243723809719086],
    "banana": [0.01868840865790844, 0.01868840865790844, 0.01868840865790844],
    "pitcher_base": [0.13843435049057007, 0.05867094546556473, 0.13843435049057007],
    "bleach_cleanser": [0.10761045664548874, 0.04818165674805641, 0.03261013329029083],
    "wood_block": [0.12235338985919952, 0.04958049952983856, 0.04958049952983856],
    "foam_brick": [0.017615724354982376, 0.03946557641029358, 0.025685379281640053],
}

INVERSE_YCB_CLASSES = {"_".join(value.split("_")[1:]): key for key, value in YCB_CLASSES.items()}
YCB_OBJECT_NAMES = list(INVERSE_YCB_CLASSES.keys())
YCB_ROOT = get_ycb_root_dir()


def load_ycb_object(scene: sapien.Scene, object_name, scale=1, visual_only=False, material=None, static=False):
    ycb_id = INVERSE_YCB_CLASSES[object_name]
    ycb_name = YCB_CLASSES[ycb_id]
    visual_file = YCB_ROOT / "visual" / ycb_name / "textured_simple.obj"
    collision_file = YCB_ROOT / "collision" / ycb_name / "collision.obj"
    builder = scene.create_actor_builder()
    scales = np.array([scale] * 3)
    density = float(os.environ.get("VIVIDEX_YCB_DENSITY", 1000))
    if material is None:
        static_friction = float(os.environ.get("VIVIDEX_YCB_STATIC_FRICTION", 1.5))
        dynamic_friction = float(os.environ.get("VIVIDEX_YCB_DYNAMIC_FRICTION", 1.0))
        restitution = float(os.environ.get("VIVIDEX_YCB_RESTITUTION", 0.1))
        material = scene.engine.create_physical_material(static_friction, dynamic_friction, restitution)
    if not visual_only:
        builder.add_multiple_collisions_from_file(str(collision_file), scale=scales, density=density, material=material)
    if visual_only:
        visual_file = YCB_ROOT / "visual" / ycb_name / "textured_simple.stl"

    builder.add_visual_from_file(str(visual_file), scale=scales)
    if not visual_only and not static:
        actor = builder.build(name=YCB_CLASSES[ycb_id])
    else:
        actor = builder.build_static(name=f"{YCB_CLASSES[ycb_id]}_visual")
    return actor
