import os
import numpy as np
from hand_imitation.env import task_setting
from hand_imitation.env.motion_paths import resolve_existing_motion_path
from hand_imitation.env.rl_env.relocate_env import AllegroRelocateRLEnv
from hand_imitation.env.sim_env.constructor import add_default_scene_light


def create_env(name, task_kwargs=None, use_gui=False, is_eval=False, is_vision=False, is_demo_rollout=False, is_real_robot=False, pc_noise=False, point_cs="world", norm_traj=False, robot_name="allegro_hand_ur5"):
    if robot_name == "rm75_inspire_right":
        from hand_imitation.env.create_rm75_env import create_rm75_env

        return create_rm75_env(
            name=name,
            task_kwargs=task_kwargs,
            use_gui=use_gui,
            is_eval=is_eval,
            is_vision=is_vision,
            is_demo_rollout=is_demo_rollout,
            is_real_robot=is_real_robot,
            pc_noise=pc_noise,
            point_cs=point_cs,
            norm_traj=norm_traj,
            robot_name=robot_name,
        )

    cur_dir = os.path.dirname(__file__)
    repo_root = os.path.abspath(os.path.join(cur_dir, "../../"))
    allow_legacy_fallback = os.environ.get("VIVIDEX_ALLOW_LEGACY_TRAJ_FALLBACK", "0") == "1"
    motion_path = resolve_existing_motion_path(
        repo_root,
        name,
        norm_traj,
        robot_name,
        allow_legacy_fallback=allow_legacy_fallback,
    )

    try:
        motion_file = np.load(motion_path)
        motion_file = {k:v for k, v in motion_file.items()}
        if name == "pour":
            motion_file['task_name'] = "pour"
        elif name == "place":
            motion_file['task_name'] = "place"
        else:
            motion_file['task_name'] = "relocate"
    except:
        motion_file = name

    headless_no_render = os.environ.get("VIVIDEX_HEADLESS_NO_RENDER", "0") == "1"
    env_params = dict(motion_file=motion_file, use_gui=use_gui, task_kwargs=task_kwargs, is_eval=is_eval, is_vision=is_vision, is_demo_rollout=is_demo_rollout, is_real_robot=is_real_robot, pc_noise=pc_noise, point_cs=point_cs, norm_traj=norm_traj, robot_name=robot_name)

    if is_eval or is_vision or is_demo_rollout:
        env_params["no_rgb"] = False
        env_params["need_offscreen_render"] = True

    if headless_no_render and not use_gui and not is_vision:
        env_params["no_rgb"] = True
        env_params["need_offscreen_render"] = False

    # Specify rendering device if the computing device is given
    if "CUDA_VISIBLE_DEVICES" in os.environ and not headless_no_render:
        env_params["device"] = "cuda"

    env = AllegroRelocateRLEnv(**env_params)

    if is_eval and not (headless_no_render and not is_vision):
        if is_vision:
            if is_real_robot:
                env.setup_camera_from_config(task_setting.CAMERA_CONFIG["relocate"])
                add_default_scene_light(env.scene, env.renderer)
                env.setup_visual_obs_config(task_setting.OBS_CONFIG["instance_real"])
            else:
                env.setup_camera_from_config(task_setting.CAMERA_CONFIG["relocate"])
                add_default_scene_light(env.scene, env.renderer)
                env.setup_visual_obs_config(task_setting.OBS_CONFIG["instance"])
        else:
            env.setup_camera_from_config(task_setting.CAMERA_CONFIG["viz_only"])
            add_default_scene_light(env.scene, env.renderer)

    # flush cache
    env.action_space
    env.observation_space

    return env
