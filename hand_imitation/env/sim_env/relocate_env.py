import os
import numpy as np
import sapien.core as sapien
from hand_imitation.env.sim_env.base import BaseSimulationEnv
from hand_imitation.real_world import lab
from hand_imitation.utils.render_scene_utils import set_entity_color
from hand_imitation.utils.ycb_object_utils import load_ycb_object, YCB_SIZE, YCB_ORIENTATION


def configured_ycb_object_scale(task_name, task_kwargs=None):
    object_scale = 0.9 if task_name in ("pour", "place") else 1.0
    override = None
    if task_kwargs is not None:
        override = task_kwargs.get("object_scale", task_kwargs.get("ycb_object_scale", None))
    if override is None:
        override = os.environ.get("VIVIDEX_YCB_OBJECT_SCALE", None)
    if override not in (None, ""):
        object_scale *= float(override)
    return object_scale


class LabRelocateEnv(BaseSimulationEnv):
    def __init__(self, use_gui=False, frame_skip=10, robot_name="allegro_hand_ur5", object_category="YCB", object_name="tomato_soup_can", **renderer_kwargs):
        super().__init__(use_gui=use_gui, frame_skip=frame_skip, **renderer_kwargs)
        # Object info
        self.robot_name = robot_name
        self.object_category = object_category
        self.object_name = object_name
        self.object_scale = configured_ycb_object_scale(
            getattr(self, "task_name", "relocate"),
            getattr(self, "task_kwargs", None),
        )
        self.target_pose = sapien.Pose()

        # Dynamics info
        self.randomness_scale = 1
        self.friction = 1

        # Construct scene
        self.scene = self.engine.create_scene()
        self.scene.set_timestep(0.005)

        # Dummy camera creation to initial geometry object
        if self.renderer and not self.no_rgb:
            cam = self.scene.add_camera("init_not_used", width=10, height=10, fovy=1, near=0.1, far=1)
            self.scene.remove_camera(cam)

        # Load table
        table_height = 0.79
        self.tables = self.create_lab_tables(table_height=table_height)

        # Load object
        self.manipulated_object, self.target_object, self.object_height = self.load_object(object_name)

    def load_object(self, object_name):
        object_scale = configured_ycb_object_scale(
            getattr(self, "task_name", "relocate"),
            getattr(self, "task_kwargs", None),
        )
        self.object_scale = object_scale

        manipulated_object = load_ycb_object(self.scene, object_name, scale=object_scale)
        if self.is_vision:
            target_object = None
        else:
            target_object = load_ycb_object(self.scene, object_name, scale=object_scale, visual_only=True)
            target_object.set_name("target_object")
            if self.renderer and not self.no_rgb:
                set_entity_color([target_object], [0, 1, 0, 0.6])
        object_height = YCB_SIZE[self.object_name][2] * object_scale / 2

        return manipulated_object, target_object, object_height

    def generate_random_object_pose(self, randomness_scale):
        pos = np.array([self.np_random.uniform(low=0.2, high=0.4), 0.18]) * randomness_scale
        orientation = YCB_ORIENTATION[self.object_name]
        position = np.array([pos[0], pos[1], self.object_height])
        pose = sapien.Pose(position, orientation)
        return pose

    def generate_random_target_pose(self, randomness_scale):
        pos = np.array([self.np_random.uniform(low=0.2, high=0.4), 0.18]) * randomness_scale
        height = 0.25
        position = np.array([pos[0], pos[1], height])
        # No randomness for the orientation. Keep the canonical orientation.
        orientation = YCB_ORIENTATION[self.object_name]
        pose = sapien.Pose(position, orientation)
        return pose

    def reset_env(self):
        if "any" in self.object_name:
            self.scene.remove_actor(self.manipulated_object)
            self.scene.remove_actor(self.target_object)
            self.manipulated_object, self.target_object, self.object_height = self.load_object(self.object_name)

        pose = self.generate_random_object_pose(self.randomness_scale)
        self.manipulated_object.set_pose(pose)

        # Target pose
        if not self.is_vision:
            pose = self.generate_random_target_pose(self.randomness_scale)
            self.target_object.set_pose(pose)
            self.target_pose = pose

    def create_lab_tables(self, table_height):
        # Build object table first
        builder = self.scene.create_actor_builder()
        table_thickness = 0.78

        # Top
        top_pose = sapien.Pose(np.array([lab.TABLE_ORIGIN[0], lab.TABLE_ORIGIN[1], -table_thickness / 2]))
        top_material = self.scene.create_physical_material(1, 0.5, 0.01)
        table_half_size = np.concatenate([lab.TABLE_XY_SIZE / 2, [table_thickness / 2]])
        builder.add_box_collision(pose=top_pose, half_size=table_half_size, material=top_material)
        # Leg
        if self.renderer is not None:
            table_visual_material = self.renderer.create_material()
            table_visual_material.set_metallic(0.0)
            table_visual_material.set_specular(0.3)
            table_visual_material.set_base_color(np.array([0.9, 0.9, 0.9, 1]))
            table_visual_material.set_roughness(0.3)
            builder.add_box_visual(pose=top_pose, half_size=table_half_size, material=table_visual_material)
        object_table = builder.build_static("object_table")

        return object_table
