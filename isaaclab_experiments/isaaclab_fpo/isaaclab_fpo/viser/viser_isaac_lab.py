#!/usr/bin/env python3
"""
ViserIsaacLab: Real-time visualization of Isaac Lab environments using Viser.

This class provides a bridge between Isaac Lab simulations and Viser web-based
3D visualization. It loads pre-extracted assets and updates their transforms
in real-time based on the simulation state.

Example usage:
    viser_viz = ViserIsaacLab(
        asset_dir=Path("output/isaac_velocity_flat_unitree_go2_v0"),
        port=8080
    )
    viser_viz.load_from_env(env)

    # In simulation loop:
    viser_viz.update_from_env(env)
"""

from __future__ import annotations

import colorsys
import hashlib
import numpy as np
import time
import torch
import trimesh
import yaml
from collections import deque
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import viser
import viser.uplot


class ViserIsaacLab:
    """Bridge between Isaac Lab and Viser for real-time visualization."""

    def __init__(
        self,
        asset_dir: Path,
        port: int = 8080,
        host: str = "0.0.0.0",
        num_envs: int = 1,
        update_freq: int = 1,
        show_axes: bool = True,
        axes_size: float = 0.5,
        env_spacing: float = 0.0,
        fps: int = 30,
        random_offsets: Optional[np.ndarray] = None,
    ):
        """Initialize ViserIsaacLab visualization.

        Args:
            asset_dir: Directory containing extracted assets from isaac_asset_extractor.py
            port: Port for Viser web server
            host: Host address for Viser server
            num_envs: Number of environments to visualize
            update_freq: Update every N simulation steps
            show_axes: Whether to show coordinate axes
            axes_size: Size of coordinate axes
            env_spacing: Spacing between environments when visualizing multiple
            fps: Target frame rate for visualization updates (default: 30)
            random_offsets: Optional array of shape (num_envs, 3) with random XY offsets for each environment
        """
        self.asset_dir = Path(asset_dir)
        self.port = port
        self.host = host
        self.num_envs = num_envs
        self.update_freq = update_freq
        self.step_count = 0
        self.env_spacing = env_spacing
        self.random_offsets = random_offsets

        # Frame rate limiting for smooth visualization
        self.fps = fps
        self.last_update_time = 0.0
        self.min_update_interval = 1.0 / fps if fps > 0 else 0.0

        # Playback control
        self.is_playing = True

        # Environment reference for reset functionality
        self.env = None
        self.reset_requested = False
        
        # Reward tracking
        self.reward_history_length = 100  # Keep last 100 timesteps
        self.reward_history = deque(maxlen=self.reward_history_length)
        self.time_steps = deque(maxlen=self.reward_history_length)
        self.current_timestep = 0
        self.reward_plot_handle = None
        
        # Action tracking
        self.action_history = deque(maxlen=self.reward_history_length)
        self.action_plot_handles = []  # List of plot handles, one per action dimension
        self.action_dim = None  # Will be set when we first receive actions
        
        # Selected environment for plotting
        self.selected_env_idx = 0
        self.gui_env_selector = None
        self.plots_folder = None  # Store the plots folder handle

        # Validate asset directory
        if not self.asset_dir.exists():
            raise ValueError(f"Asset directory does not exist: {self.asset_dir}")

        # Store number of environments
        self.num_envs = num_envs
        if num_envs > 1:
            print(f"🔢 Visualizing {num_envs} environments using batched rendering")

        # Create Viser server
        self.server = viser.ViserServer(host=host, port=port)
        print(f"🌐 Viser server started at http://{host}:{port}")

        # Load metadata
        self._load_metadata()

        # Track loaded meshes and handles
        self.loaded_meshes: Dict[str, trimesh.Trimesh] = {}
        self.mesh_handles: Dict[str, viser.SceneNodeHandle] = {}  # For single env
        self.batched_mesh_handles: Dict[
            str, viser.SceneNodeHandle
        ] = {}  # For multi env
        self.body_to_mesh: Dict[str, str] = {}
        self.body_to_handle: Dict[str, viser.SceneNodeHandle] = {}  # For single env
        self.body_idx_to_batched_handle: Dict[
            int, viser.SceneNodeHandle
        ] = {}  # For multi env
        # Separate tracking for visual and collision geometry
        self.body_idx_to_visual_handle: Dict[int, viser.SceneNodeHandle] = {}
        self.body_idx_to_collision_handle: Dict[int, viser.SceneNodeHandle] = {}

        # Velocity visualization
        self.velocity_lines_handle = None
        self.show_velocity = True
        self.velocity_scale = 1.0  # Scale factor for velocity arrows
        self.envs_per_row = int(np.ceil(np.sqrt(max(1, self.num_envs))))

        # Scene setup
        self.server.scene.add_grid("/ground", plane="xy", width=500.0, height=500.0)
        if show_axes:
            self.axes_handle = self.server.scene.add_frame(
                "/world_axes", axes_length=axes_size, axes_radius=0.01
            )

        # Load scene assets
        self._load_scene()

        # Add GUI controls
        self._setup_gui()

        print(f"✅ ViserIsaacLab initialized with {len(self.mesh_handles)} meshes")

    def _load_metadata(self):
        """Load metadata files from extraction output."""
        hierarchy_file = self.asset_dir / "scene_hierarchy.yaml"
        mapping_file = self.asset_dir / "prim_to_mesh.yaml"
        info_file = self.asset_dir / "extraction_info.yaml"

        # Check files exist
        for f in [hierarchy_file, mapping_file, info_file]:
            if not f.exists():
                raise FileNotFoundError(f"Missing metadata file: {f}")

        # Load YAML files
        with open(hierarchy_file, "r") as f:
            hierarchy_data = yaml.safe_load(f)
            self.hierarchy = hierarchy_data["hierarchy"]

        with open(mapping_file, "r") as f:
            mapping_data = yaml.safe_load(f)
            self.prim_to_mesh = mapping_data["mappings"]

        with open(info_file, "r") as f:
            self.info = yaml.safe_load(f)

        print(
            f"📊 Loaded metadata: {len(self.hierarchy)} nodes, {len(self.prim_to_mesh)} mesh mappings"
        )

    def _load_scene(self):
        """Load all GLB meshes and create scene structure."""
        # First, load all unique meshes
        unique_meshes = set(self.prim_to_mesh.values())
        print(f"📦 Loading {len(unique_meshes)} unique meshes...")

        for mesh_file in unique_meshes:
            mesh_path = self.asset_dir / mesh_file
            if mesh_path.exists():
                try:
                    # Load mesh with trimesh
                    mesh = trimesh.load(str(mesh_path))
                    self.loaded_meshes[mesh_file] = mesh
                except Exception as e:
                    print(f"  ❌ Failed to load {mesh_file}: {e}")

        # Create body name to mesh mapping
        for prim_path, mesh_file in self.prim_to_mesh.items():
            # Extract body name from path
            parts = prim_path.split("/")
            if parts:
                # Look for robot body parts
                for i, part in enumerate(parts):
                    if "robot" in part.lower() and i + 1 < len(parts):
                        body_name = parts[i + 1]
                        # Skip visual/collision nodes
                        if not any(
                            s in body_name.lower() for s in ["visual", "collision"]
                        ):
                            self.body_to_mesh[body_name] = mesh_file
                            break

        # Add meshes to Viser scene atomically
        with self.server.atomic():
            for prim_path, mesh_file in self.prim_to_mesh.items():
                if mesh_file in self.loaded_meshes:
                    # Skip non-robot meshes for now
                    if "robot" not in prim_path.lower():
                        continue

                    # Create Viser path
                    viser_path = self._create_viser_path(prim_path)

                    try:
                        mesh = self.loaded_meshes[mesh_file]

                        # Handle both single mesh and scene cases
                        if isinstance(mesh, trimesh.Scene):
                            mesh = mesh.to_geometry()

                        # Generate color based on mesh file name
                        color = self._name_to_color(mesh_file)

                        if self.num_envs == 1:
                            # Single environment - use regular mesh
                            # Set initial visibility based on whether it's visual or collision
                            is_collision = "collision" in prim_path.lower()
                            handle = self.server.scene.add_mesh_simple(
                                name=viser_path,
                                vertices=mesh.vertices,
                                faces=mesh.faces,
                                color=color,
                                flat_shading=True,
                                visible=not is_collision,  # Visual geometry visible by default
                            )
                            self.mesh_handles[prim_path] = handle

                            # Map body names to handles
                            body_name = self._extract_body_name(prim_path)
                            if body_name and body_name not in self.body_to_handle:
                                self.body_to_handle[body_name] = handle
                        else:
                            # Multiple environments - create batched mesh later
                            # For now, just store the mesh info
                            # We'll map body names in _create_batched_meshes
                            pass

                    except Exception as e:
                        print(f"  ❌ Failed to add {viser_path}: {e}")

    def load_from_env(self, env):
        """Initialize mapping from Isaac Lab environment.

        Args:
            env: Isaac Lab environment (unwrapped)
        """
        # Store environment reference for reset functionality
        self.env = env

        # Debug: Show what's in the scene
        print(f"\n[DEBUG] Scene type: {type(env.scene)}")
        if hasattr(env.scene, "keys"):
            try:
                scene_keys = list(env.scene.keys())
                print(f"[DEBUG] Scene keys: {scene_keys}")
            except Exception as e:
                print(f"[DEBUG] Error getting scene keys: {e}")

        # Find the robot articulation
        self.robot = None
        self.robot_name = None

        # Try common names
        for name in ["robot", "anymal", "unitree", "go2", "cartpole"]:
            try:
                self.robot = env.scene[name]
                self.robot_name = name
                break
            except (KeyError, TypeError) as e:
                continue

        if self.robot is None:
            # Try articulations dict
            if hasattr(env.scene, "articulations"):
                try:
                    if env.scene.articulations:
                        self.robot_name = list(env.scene.articulations.keys())[0]
                        self.robot = env.scene.articulations[self.robot_name]
                except Exception as e:
                    print(f"[DEBUG] Error accessing articulations: {e}")

        if self.robot is None:
            print("[ERROR] Could not find robot articulation in environment")
            print(
                f"[ERROR] Available scene attributes: {[attr for attr in dir(env.scene) if not attr.startswith('_')][:10]}"
            )
            raise RuntimeError("Could not find robot articulation in environment")

        # Update envs_per_row based on actual number of environments
        self.envs_per_row = int(np.ceil(np.sqrt(max(1, self.num_envs))))

        print(f"🤖 Found robot '{self.robot_name}' with {self.robot.num_bodies} bodies")
        print(f"   Body names: {self.robot.body_names}")

        # Debug: Show available mesh handles
        print(f"\n[DEBUG] Available mesh handles:")
        for i, (prim_path, handle) in enumerate(list(self.mesh_handles.items())[:5]):
            print(f"   {i}: {prim_path}")
        print(f"   ... and {len(self.mesh_handles) - 5} more")

        # Create mapping from Isaac body indices to Viser handles
        self.body_idx_to_handle: Dict[int, viser.SceneNodeHandle] = {}

        if self.num_envs == 1:
            # Single environment mode - map to individual mesh handles
            for body_idx, body_name in enumerate(self.robot.body_names):
                # Try different strategies to find the mesh
                handle = None

                # Strategy 1: Direct body name lookup
                if body_name in self.body_to_handle:
                    handle = self.body_to_handle[body_name]

                # Strategy 2: Look for prim path containing body name
                if handle is None:
                    # Clean up body name (remove namespace prefixes)
                    clean_body_name = (
                        body_name.split("/")[-1] if "/" in body_name else body_name
                    )

                    for prim_path, mesh_handle in self.mesh_handles.items():
                        # Check if the clean body name appears in the prim path
                        if (
                            clean_body_name in prim_path
                            and "visual" in prim_path.lower()
                        ):
                            handle = mesh_handle
                            break
                        # Also try without the visual requirement
                        elif clean_body_name in prim_path and handle is None:
                            handle = mesh_handle

                if handle:
                    self.body_idx_to_handle[body_idx] = handle
                    print(f"   ✅ Mapped body {body_idx} '{body_name}' to Viser mesh")
                else:
                    print(f"   ❌ No mesh found for body {body_idx} '{body_name}'")

        print(
            f"✅ Mapped {len(self.body_idx_to_handle)} / {self.robot.num_bodies} bodies"
        )

        # If multiple environments, create batched meshes
        if self.num_envs > 1:
            self._create_batched_meshes()

        # Create velocity visualization for any number of environments
        if self.num_envs > 0:
            self._create_velocity_visualization()

    def _create_batched_meshes(self):
        """Create batched meshes for multiple environments."""
        print(f"\n🔢 Creating batched meshes for {self.num_envs} environments...")

        # Separate visual and collision meshes
        body_visual_meshes = {}
        body_collision_meshes = {}

        for body_idx, body_name in enumerate(self.robot.body_names):
            # Clean up body name (remove namespace prefixes)
            clean_body_name = (
                body_name.split("/")[-1] if "/" in body_name else body_name
            )

            # Find visual mesh for this body
            for prim_path, mf in self.prim_to_mesh.items():
                if clean_body_name in prim_path and "robot" in prim_path.lower():
                    if "visual" in prim_path.lower() and mf in self.loaded_meshes:
                        mesh = self.loaded_meshes[mf]
                        if isinstance(mesh, trimesh.Scene):
                            mesh = mesh.to_geometry()
                        body_visual_meshes[body_idx] = mesh
                        print(f"   Found visual mesh for body {body_idx} '{body_name}'")
                    elif "collision" in prim_path.lower() and mf in self.loaded_meshes:
                        mesh = self.loaded_meshes[mf]
                        if isinstance(mesh, trimesh.Scene):
                            mesh = mesh.to_geometry()
                        body_collision_meshes[body_idx] = mesh
                        print(
                            f"   Found collision mesh for body {body_idx} '{body_name}'"
                        )

        # Create batched meshes atomically
        with self.server.atomic():
            # Create visual meshes
            for body_idx, mesh in body_visual_meshes.items():
                body_name = self.robot.body_names[body_idx]
                clean_name = body_name.split("/")[-1] if "/" in body_name else body_name

                # Generate batched colors for visual mesh
                batched_colors = self._generate_batched_colors(body_idx, body_name)

                # Compute LOD ratio based on mesh complexity (following mjlab pattern)
                lod_ratio = 1000.0 / mesh.vertices.shape[0]
                lod_param = ((2.0, lod_ratio),) if lod_ratio < 0.5 else "off"

                # Create batched visual mesh
                handle = self.server.scene.add_batched_meshes_simple(
                    name=f"/batched_visual_{clean_name}",
                    vertices=mesh.vertices,
                    faces=mesh.faces,
                    batched_wxyzs=np.tile([1.0, 0.0, 0.0, 0.0], (self.num_envs, 1)),
                    batched_positions=np.zeros((self.num_envs, 3)),
                    batched_colors=batched_colors,
                    flat_shading=True,
                    visible=True,  # Visual meshes visible by default
                    lod=lod_param,
                )

                self.body_idx_to_visual_handle[body_idx] = handle
                self.body_idx_to_batched_handle[body_idx] = (
                    handle  # Default to visual for compatibility
                )
                lod_status = "enabled" if lod_ratio < 0.5 else "disabled"
                print(f"   ✅ Created visual mesh for body {body_idx} '{clean_name}' ({mesh.vertices.shape[0]} verts, LOD {lod_status})")

            # Create collision meshes
            for body_idx, mesh in body_collision_meshes.items():
                body_name = self.robot.body_names[body_idx]
                clean_name = body_name.split("/")[-1] if "/" in body_name else body_name

                # Use a different color for collision meshes (e.g., semi-transparent red)
                collision_colors = np.tile([1.0, 0.3, 0.3], (self.num_envs, 1))

                # Compute LOD ratio based on mesh complexity (following mjlab pattern)
                lod_ratio = 1000.0 / mesh.vertices.shape[0]
                lod_param = ((2.0, lod_ratio),) if lod_ratio < 0.5 else "off"

                # Create batched collision mesh
                handle = self.server.scene.add_batched_meshes_simple(
                    name=f"/batched_collision_{clean_name}",
                    vertices=mesh.vertices,
                    faces=mesh.faces,
                    batched_wxyzs=np.tile([1.0, 0.0, 0.0, 0.0], (self.num_envs, 1)),
                    batched_positions=np.zeros((self.num_envs, 3)),
                    batched_colors=collision_colors,
                    flat_shading=True,
                    opacity=0.5,
                    visible=False,  # Collision meshes hidden by default
                    lod=lod_param,
                )

                self.body_idx_to_collision_handle[body_idx] = handle
                lod_status = "enabled" if lod_ratio < 0.5 else "disabled"
                print(
                    f"   ✅ Created collision mesh for body {body_idx} '{clean_name}' ({mesh.vertices.shape[0]} verts, LOD {lod_status})"
                )

        # Store for body mapping reference
        self.body_idx_to_handle = self.body_idx_to_batched_handle

        # Precompute grid offsets for efficient updates
        if self.random_offsets is not None:
            # Use provided random offsets
            self.grid_offsets = self.random_offsets.copy()
            print(f"   📐 Using random offsets for {self.num_envs} environments")
        else:
            # Use regular grid layout (zero-centered)
            grid_size = int(np.ceil(np.sqrt(self.num_envs)))
            self.grid_offsets = np.zeros((self.num_envs, 3))

            # Calculate center offset to make grid zero-centered
            max_row = (self.num_envs - 1) // grid_size
            max_col = min(grid_size - 1, self.num_envs - 1)
            center_x = max_col * self.env_spacing / 2
            center_y = max_row * self.env_spacing / 2

            for i in range(self.num_envs):
                row = i // grid_size
                col = i % grid_size
                self.grid_offsets[i, 0] = col * self.env_spacing - center_x
                self.grid_offsets[i, 1] = row * self.env_spacing - center_y
            print(
                f"   📐 Precomputed zero-centered {grid_size}x{grid_size} grid layout with spacing {self.env_spacing}"
            )

        print(f"✅ Created {len(self.body_idx_to_batched_handle)} batched meshes")

    def update_from_env(self, env, velocity_commands: bool = True, force: bool = False, rewards: Optional[torch.Tensor] = None, actions: Optional[torch.Tensor] = None):
        """Update Viser visualization from Isaac Lab environment state.

        Args:
            env: Isaac Lab environment (unwrapped)
            velocity_commands: Whether to update velocity command visualization. Default is True.
                Set to False for non-locomotion tasks (e.g., motion tracking) that don't have velocity commands.
            force: Force update regardless of update frequency
            rewards: Optional tensor of rewards from env.step()
            actions: Optional tensor of actions sent to env.step()
        """
        self.step_count += 1

        # Check if paused
        if not self.is_playing and not force:
            return

        # Check update frequency (step-based throttling)
        if not force and self.step_count % self.update_freq != 0:
            return

        # Check frame rate limit (time-based throttling for 30 FPS)
        current_time = time.time()
        if (
            not force
            and (current_time - self.last_update_time) < self.min_update_interval
        ):
            return
        self.last_update_time = current_time

        if self.robot is None:
            return

        if self.num_envs == 1:
            # Single environment mode
            env_idx = 0

            # Get environment origin to subtract from world positions
            if hasattr(env.scene, "env_origins"):
                env_origin = env.scene.env_origins[env_idx].cpu().numpy()
            else:
                env_origin = np.zeros(3)

            # Debug output every 100 steps
            if self.step_count % 100 == 0:
                print(f"[DEBUG] ViserIsaacLab update step {self.step_count}")
                if len(self.body_idx_to_handle) > 0:
                    first_body_idx = list(self.body_idx_to_handle.keys())[0]
                    pos_world = (
                        self.robot.data.body_link_pos_w[env_idx, first_body_idx]
                        .cpu()
                        .numpy()
                    )
                    pos_local = pos_world - env_origin
                    print(f"   First body position (world): {pos_world}")
                    print(f"   First body position (local): {pos_local}")
                    print(f"   Environment origin: {env_origin}")

            # Update all bodies atomically for smoother visualization
            with self.server.atomic():
                # Update each mapped body
                for body_idx, handle in self.body_idx_to_handle.items():
                    # Get transform from Isaac Lab (in world coordinates)
                    pos_world = (
                        self.robot.data.body_link_pos_w[env_idx, body_idx].cpu().numpy()
                    )
                    quat = (
                        self.robot.data.body_link_quat_w[env_idx, body_idx]
                        .cpu()
                        .numpy()
                    )

                    # Convert to local coordinates relative to environment origin
                    pos_local = pos_world - env_origin

                    # Apply to Viser
                    handle.position = pos_local
                    handle.wxyz = quat
        else:
            # Multiple environments mode - use batched updates
            # Get all environment origins
            if hasattr(env.scene, "env_origins"):
                env_origins = env.scene.env_origins[: self.num_envs].cpu().numpy()
            else:
                env_origins = np.zeros((self.num_envs, 3))

            # Debug output every 100 steps
            if self.step_count % 100 == 0:
                print(f"[DEBUG] ViserIsaacLab batched update step {self.step_count}")
                print(f"   Updating {self.num_envs} environments")

            # Update all bodies atomically for smoother visualization
            with self.server.atomic():
                # Get all transforms at once for efficiency
                for body_idx in range(self.robot.num_bodies):
                    # Get transforms for all environments
                    positions_world = (
                        self.robot.data.body_link_pos_w[: self.num_envs, body_idx]
                        .cpu()
                        .numpy()
                    )
                    quaternions = (
                        self.robot.data.body_link_quat_w[: self.num_envs, body_idx]
                        .cpu()
                        .numpy()
                    )

                    # Convert to local coordinates and add precomputed grid offsets
                    positions_local = positions_world - env_origins + self.grid_offsets

                    # Update visual mesh if it exists
                    if body_idx in self.body_idx_to_visual_handle:
                        self.body_idx_to_visual_handle[
                            body_idx
                        ].batched_positions = positions_local
                        self.body_idx_to_visual_handle[
                            body_idx
                        ].batched_wxyzs = quaternions

                    # Update collision mesh if it exists
                    if body_idx in self.body_idx_to_collision_handle:
                        self.body_idx_to_collision_handle[
                            body_idx
                        ].batched_positions = positions_local
                        self.body_idx_to_collision_handle[
                            body_idx
                        ].batched_wxyzs = quaternions

        # Update velocity visualization
        if velocity_commands:
            self.update_velocity_visualization(env)
        
        # Update reward tracking and plot
        if rewards is not None:
            self._update_reward_tracking(rewards)
            
        # Update action tracking for first environment
        if actions is not None:
            self._update_action_tracking(actions)

    def _create_viser_path(self, prim_path: str) -> str:
        """Convert USD prim path to Viser-compatible path."""
        # Remove leading slash
        if prim_path.startswith("/"):
            prim_path = prim_path[1:]

        # Simplify path
        path = prim_path.replace("World/", "")
        path = path.replace("envs/env_0/", "")

        return f"/{path}"

    def _extract_body_name(self, prim_path: str) -> Optional[str]:
        """Extract body name from prim path."""
        parts = prim_path.split("/")

        # Look for body name (skip visual/collision suffixes)
        for i in range(len(parts) - 1, -1, -1):
            part = parts[i]
            if not any(s in part.lower() for s in ["visual", "collision", "mesh"]):
                return part

        return None

    def _name_to_color(self, name: str) -> Tuple[float, float, float]:
        """Generate a consistent color based on name hash."""
        # Create a hash of the name
        hash_obj = hashlib.md5(name.encode())
        hash_hex = hash_obj.hexdigest()

        # Use first 6 characters for RGB
        r = int(hash_hex[0:2], 16) / 255.0
        g = int(hash_hex[2:4], 16) / 255.0
        b = int(hash_hex[4:6], 16) / 255.0

        # Ensure minimum brightness
        min_brightness = 0.3
        r = min_brightness + (1 - min_brightness) * r
        g = min_brightness + (1 - min_brightness) * g
        b = min_brightness + (1 - min_brightness) * b

        return (r, g, b)

    def _generate_batched_colors(self, body_idx: int, body_name: str) -> np.ndarray:
        """Generate HSV-based colors for batched meshes.

        Each environment gets a base hue from the color wheel, and each body part
        gets a slight variation in saturation and value.

        Args:
            body_idx: Index of the body part
            body_name: Name of the body part

        Returns:
            Array of shape (num_envs, 3) with RGB colors for each environment
        """
        colors = np.zeros((self.num_envs, 3))

        # Generate a hash-based offset for this specific body part
        # This ensures consistent coloring across runs
        body_hash = hashlib.md5(body_name.encode()).hexdigest()
        body_offset = int(body_hash[:4], 16) / 0xFFFF  # Normalize to [0, 1]

        # Create variation parameters for this body
        saturation_variation = 0.6 + 0.3 * body_offset  # Range: [0.7, 1.0]
        value_variation = 0.8 + 0.2 * body_offset  # Range: [0.8, 1.0]

        # Add a small hue shift based on body index
        hue_shift = (body_idx * 0.005) % 1.0  # Small shift per body part

        for env_idx in range(self.num_envs):
            # Base hue for this environment (evenly distributed around color wheel)
            base_hue = (env_idx / max(1, self.num_envs - 1)) % 1.0

            # Apply body-specific hue shift
            hue = (base_hue + hue_shift) % 1.0

            # Convert HSV to RGB
            r, g, b = colorsys.hsv_to_rgb(hue, saturation_variation, value_variation)
            colors[env_idx] = [r, g, b]

        return colors

    def _create_velocity_visualization(self):
        """Create batched line segments for velocity visualization."""
        if self.num_envs == 0:
            return

        # Create initial points array - shape (num_envs, 2, 3)
        # Each line segment has a start and end point
        points = np.zeros((self.num_envs, 2, 3))

        # Initialize with small vertical lines at origin
        for i in range(self.num_envs):
            points[i, 0] = [0, 0, 0.1]  # Start slightly above ground
            points[i, 1] = [0, 0, 0.2]  # End point

        # Create colors for each line - bright colors from HSV wheel
        colors = np.zeros((self.num_envs, 2, 3))
        for i in range(self.num_envs):
            hue = i / max(1, self.num_envs)
            color = colorsys.hsv_to_rgb(hue, 1.0, 1.0)
            colors[i, 0] = color  # Start point color
            colors[i, 1] = color  # End point color (same)

        # Create the line segments
        self.velocity_lines_handle = self.server.scene.add_line_segments(
            name="/velocity_arrows",
            points=points,
            colors=colors,
            line_width=1.5,
            visible=self.show_velocity,
        )

        print(f"✅ Created velocity visualization for {self.num_envs} environments")

    def update_velocity_visualization(self, env):
        """Update velocity arrows based on current velocity commands."""
        if not self.show_velocity or self.velocity_lines_handle is None:
            return

        # Try to get velocity commands from the environment
        velocity_commands = None

        # Check common locations for velocity commands
        if hasattr(env, "command_manager") and hasattr(
            env.command_manager, "get_command"
        ):
            # IsaacLab standard command manager
            velocity_commands = env.command_manager.get_command("base_velocity")
        elif hasattr(env, "commands"):
            # Direct commands attribute
            velocity_commands = env.commands
        elif hasattr(env.unwrapped, "commands"):
            # Try unwrapped environment
            velocity_commands = env.unwrapped.commands

        if velocity_commands is None:
            return

        # Ensure we have the right shape
        if isinstance(velocity_commands, torch.Tensor):
            velocity_commands = velocity_commands.cpu().numpy()

        # Create points array for line segments
        points = np.zeros((self.num_envs, 2, 3))

        # Get environment origins if available
        if hasattr(env.scene, "env_origins"):
            env_origins = env.scene.env_origins.cpu().numpy()
        else:
            env_origins = np.zeros((self.num_envs, 3))

        for env_idx in range(self.num_envs):
            # Get robot base position and orientation in world coordinates
            if self.robot is not None:
                base_pos_world = self.robot.data.root_pos_w[env_idx].cpu().numpy()
                base_quat_w = (
                    self.robot.data.root_quat_w[env_idx].cpu().numpy()
                )  # [w, x, y, z]

                # Convert to local coordinates by subtracting environment origin
                base_pos_local = base_pos_world - env_origins[env_idx]
                # Add grid offset to match batched mesh positions
                base_pos = base_pos_local + self.grid_offsets[env_idx]
            else:
                base_pos = self.grid_offsets[env_idx]
                base_quat_w = np.array([1.0, 0.0, 0.0, 0.0])  # Identity quaternion

            # Extract velocity command (typically [vx, vy, omega])
            if velocity_commands.shape[0] > env_idx and velocity_commands.shape[1] >= 2:
                # Velocity in robot's local frame
                vx_local = velocity_commands[env_idx, 0] * self.velocity_scale
                vy_local = velocity_commands[env_idx, 1] * self.velocity_scale

                # Convert quaternion to rotation matrix to transform velocity to world frame
                # Quaternion format: [w, x, y, z]
                w, x, y, z = base_quat_w

                # Rotation matrix from quaternion (only need the 2D rotation part)
                # R = [[1-2(y²+z²), 2(xy-wz), 2(xz+wy)],
                #      [2(xy+wz), 1-2(x²+z²), 2(yz-wx)],
                #      [2(xz-wy), 2(yz+wx), 1-2(x²+y²)]]
                R00 = 1 - 2 * (y * y + z * z)
                R01 = 2 * (x * y - w * z)
                R10 = 2 * (x * y + w * z)
                R11 = 1 - 2 * (x * x + z * z)

                # Transform velocity from local to world frame
                vx_world = R00 * vx_local + R01 * vy_local
                vy_world = R10 * vx_local + R11 * vy_local

                # Create arrow from base position in direction of velocity
                # Start point (slightly above ground)
                points[env_idx, 0] = base_pos + np.array([0, 0, 0.1])

                # End point - in direction of velocity (now in world frame)
                points[env_idx, 1] = points[env_idx, 0] + np.array(
                    [vx_world, vy_world, 0]
                )
            else:
                # No velocity data - just show a small vertical line
                points[env_idx, 0] = base_pos + np.array([0, 0, 0.1])
                points[env_idx, 1] = base_pos + np.array([0, 0, 0.15])

        # Update the line segments
        self.velocity_lines_handle.points = points

    def _setup_gui(self):
        """Setup GUI controls in Viser."""
        with self.server.gui.add_folder("Controls"):
            # Play/Pause buttons (two separate buttons with visibility toggle)
            self.gui_pause_button = self.server.gui.add_button(
                "Pause",
                icon=viser.Icon.PLAYER_PAUSE,
                visible=True,
            )

            self.gui_play_button = self.server.gui.add_button(
                "Play",
                icon=viser.Icon.PLAYER_PLAY,
                visible=False,
            )

            # Reset all environments button
            self.gui_reset_button = self.server.gui.add_button(
                "Reset All Environments",
                icon=viser.Icon.REFRESH,
                visible=True,
            )

            # Update frequency slider
            self.gui_update_freq = self.server.gui.add_slider(
                "Update Frequency",
                min=1,
                max=10,
                step=1,
                initial_value=self.update_freq,
            )

            # Geometry visibility checkboxes
            self.gui_show_visual = self.server.gui.add_checkbox(
                "Show Visual Geometry",
                initial_value=True,
            )

            self.gui_show_collision = self.server.gui.add_checkbox(
                "Show Collision Geometry",
                initial_value=False,
            )

            # Velocity visualization controls
            self.gui_show_velocity = self.server.gui.add_checkbox(
                "Show Velocity Commands",
                initial_value=self.show_velocity,
            )

            self.gui_velocity_scale = self.server.gui.add_slider(
                "Velocity Arrow Scale",
                min=0.1,
                max=5.0,
                step=0.1,
                initial_value=self.velocity_scale,
            )
            
        # All plots in one folder
        self.plots_folder = self.server.gui.add_folder("Plots")
        with self.plots_folder:
            # Environment selector
            self.gui_env_selector = self.server.gui.add_number(
                "Environment Index",
                initial_value=0,
                min=0,
                max=self.num_envs - 1,
                step=1,
            )
            
            # Reward plot section
            self.server.gui.add_html("<h4 style='text-align: center; margin: 5px 0;'>Rewards</h4>")
            
            # Create initial empty data
            initial_time = np.array([0.0], dtype=np.float32)
            initial_value = np.array([0.0], dtype=np.float32)
            
            # Create series for single environment
            series = (
                viser.uplot.Series(label="time"),
                viser.uplot.Series(
                    label="Reward",
                    stroke="rgb(0, 150, 255)",
                    width=3,
                )
            )
            
            self.reward_plot_handle = self.server.gui.add_uplot(
                data=(initial_time, initial_value),
                series=series,
                scales={
                    "x": viser.uplot.Scale(time=False, auto=True),
                    "y": viser.uplot.Scale(auto=True),
                },
                legend=viser.uplot.Legend(show=False),
                aspect=1.0,
            )
            
            # Action plots will be created dynamically when we know the action dimension
            
        # Show statistics using HTML for better formatting
        with self.server.gui.add_folder("Info"):
            stats_html = f"""
            <div style="padding: 10px; background-color: #f0f0f0; border-radius: 5px; margin-top: 10px;">
                <table style="width: 100%;">
                    <tr><td><b>Environments:</b></td><td>{self.num_envs}</td></tr>
                    <tr><td><b>Target FPS:</b></td><td>{self.fps}</td></tr>
                    <tr><td><b>Update Freq:</b></td><td>Every {self.update_freq} steps</td></tr>
                </table>
            </div>
            """
            self.gui_stats = self.server.gui.add_html(stats_html)

        # Connect callbacks
        @self.gui_pause_button.on_click
        def _(event):
            self.is_playing = False
            self.gui_pause_button.visible = False
            self.gui_play_button.visible = True

        @self.gui_play_button.on_click
        def _(event):
            self.is_playing = True
            self.gui_play_button.visible = False
            self.gui_pause_button.visible = True

        @self.gui_update_freq.on_update
        def _(event):
            self.update_freq = event.target.value
            # Update statistics HTML
            self._update_stats_html()

        @self.gui_show_visual.on_update
        def _(event):
            self._update_geometry_visibility()

        @self.gui_show_collision.on_update
        def _(event):
            self._update_geometry_visibility()

        @self.gui_reset_button.on_click
        def _(event):
            if self.env is not None:
                # Set flag to request reset in main thread
                self.reset_requested = True
                # Clear reward and action history
                self.reward_history.clear()
                self.action_history.clear()
                self.time_steps.clear()
                self.current_timestep = 0
                print(f"🔄 Reset requested for all {self.num_envs} environments")

    def _update_stats_html(self):
        """Update the statistics HTML display."""
        stats_html = f"""
        <div style="padding: 10px; background-color: #f0f0f0; border-radius: 5px; margin-top: 10px;">
            <table style="width: 100%;">
                <tr><td><b>Environments:</b></td><td>{self.num_envs}</td></tr>
                <tr><td><b>Target FPS:</b></td><td>{self.fps}</td></tr>
                <tr><td><b>Update Freq:</b></td><td>Every {self.update_freq} steps</td></tr>
            </table>
        </div>
        """
        self.gui_stats.content = stats_html

    def _update_geometry_visibility(self):
        """Update visibility of visual and collision geometry."""
        show_visual = self.gui_show_visual.value
        show_collision = self.gui_show_collision.value

        # Update visibility for all mesh handles
        if self.num_envs == 1:
            # Single environment mode
            for prim_path, handle in self.mesh_handles.items():
                if "visual" in prim_path.lower():
                    handle.visible = show_visual
                elif "collision" in prim_path.lower():
                    handle.visible = show_collision
        else:
            # Multi-environment mode with batched meshes
            # Update visual meshes
            for handle in self.body_idx_to_visual_handle.values():
                handle.visible = show_visual

            # Update collision meshes
            for handle in self.body_idx_to_collision_handle.values():
                handle.visible = show_collision

        # Velocity visualization callbacks
        @self.gui_show_velocity.on_update
        def _(event):
            self.show_velocity = event.target.value
            if self.velocity_lines_handle is not None:
                self.velocity_lines_handle.visible = self.show_velocity

        @self.gui_velocity_scale.on_update
        def _(event):
            self.velocity_scale = event.target.value
            
        # Environment selector callback
        @self.gui_env_selector.on_update
        def _(event):
            self.selected_env_idx = int(event.target.value)
            # Force update of plots with new selection
            if len(self.reward_history) > 0:
                self._update_reward_tracking(None)  # Update with existing data
            if len(self.action_history) > 0:
                self._update_action_tracking(None)  # Update with existing data

    def _update_reward_tracking(self, rewards: Optional[torch.Tensor]):
        """Update reward history and plot.
        
        Args:
            rewards: Tensor of rewards from env.step() or None to just update plot
        """
        # Only add new data if rewards provided
        if rewards is not None:
            # Convert rewards to numpy
            if isinstance(rewards, torch.Tensor):
                rewards_np = rewards.cpu().numpy()
            else:
                rewards_np = np.array(rewards)
            
            # Update timestep
            self.current_timestep += 1
            
            # Add to history
            self.time_steps.append(float(self.current_timestep))
            self.reward_history.append(rewards_np.copy())
        
        # Update plot if we have enough data
        if len(self.time_steps) > 1 and self.reward_plot_handle is not None:
            # Convert to arrays
            time_array = np.array(self.time_steps, dtype=np.float32)
            rewards_array = np.array(self.reward_history)  # Shape: (timesteps, num_envs)
            
            # Get selected environment index
            selected_idx = int(self.selected_env_idx)
            if selected_idx < rewards_array.shape[1]:
                reward_data = rewards_array[:, selected_idx].astype(np.float32)
            else:
                reward_data = np.zeros_like(time_array)
            
            # Update plot
            self.reward_plot_handle.data = (time_array, reward_data)
    
    def _update_action_tracking(self, actions: Optional[torch.Tensor]):
        """Update action history and plot for selected environment.
        
        Args:
            actions: Tensor of actions sent to env.step() or None to just update plot
        """
        # Only add new data if actions provided
        if actions is not None:
            # Convert actions to numpy
            if isinstance(actions, torch.Tensor):
                actions_np = actions.cpu().numpy()
            else:
                actions_np = np.array(actions)
            
            # Initialize action plot on first call
            if self.action_dim is None and actions_np.shape[0] > 0:
                self.action_dim = actions_np.shape[1]
                self._create_action_plot()
            
            # Add all environments' actions to history
            if actions_np.shape[0] > 0:
                self.action_history.append(actions_np.copy())
        
        # Update plots if we have enough data
        if len(self.action_history) > 1 and len(self.action_plot_handles) > 0:
            # Convert to arrays
            time_array = np.array(self.time_steps, dtype=np.float32)
            actions_array = np.array(self.action_history)  # Shape: (timesteps, num_envs, action_dim)
            
            # Get selected environment index
            selected_idx = int(self.selected_env_idx)
            
            # Update each individual action plot
            for i in range(self.action_dim):
                if selected_idx < actions_array.shape[1]:
                    action_data = actions_array[:, selected_idx, i].astype(np.float32)
                else:
                    action_data = np.zeros_like(time_array)
                self.action_plot_handles[i].data = (time_array, action_data)
    
    def _create_action_plot(self):
        """Create individual plots for each action dimension."""
        if len(self.action_plot_handles) > 0:
            return
            
        # Create initial empty data
        initial_time = np.array([0.0], dtype=np.float32)
        initial_value = np.array([0.0], dtype=np.float32)
        
        # Add action plots to the existing Plots folder
        with self.plots_folder:
            # Create a separate plot for each action dimension
            for i in range(self.action_dim):
                # Generate a unique color for this action
                hue = i / max(1, self.action_dim - 1)
                r, g, b = colorsys.hsv_to_rgb(hue, 0.8, 0.9)
                color = f"rgb({int(r*255)}, {int(g*255)}, {int(b*255)})"
                
                # Create series for this action
                series = (
                    viser.uplot.Series(label="time"),
                    viser.uplot.Series(
                        label=f"Action {i}",
                        stroke=color,
                        width=2,
                    )
                )
                
                # Add HTML title above the plot (centered)
                self.server.gui.add_html(
                    f"<h4 style='margin: 5px 0; color: {color}; text-align: center;'>Action {i}</h4>"
                )
                
                # Create the plot
                plot_handle = self.server.gui.add_uplot(
                    data=(initial_time, initial_value),
                    series=series,
                    scales={
                        "x": viser.uplot.Scale(time=False, auto=True),
                        "y": viser.uplot.Scale(auto=True),
                    },
                    legend=viser.uplot.Legend(show=False),
                    aspect=1.0,  # Square aspect ratio
                )
                
                self.action_plot_handles.append(plot_handle)
    
    def check_reset_request(self):
        """Check if reset was requested and clear the flag.
        
        Returns:
            bool: True if reset was requested, False otherwise.
        """
        if self.reset_requested:
            self.reset_requested = False
            return True
        return False
    
    def close(self):
        """Close the Viser server."""
        print("👋 Shutting down Viser server...")
        # Server cleanup is handled automatically


# Example usage function
def example_usage():
    """Example of how to use ViserIsaacLab with a play.py script."""

    print("""
    Example integration with play.py:

    # After creating environment
    env = gym.make(args.task, cfg=env_cfg)

    # Initialize Viser visualization
    if args.viser:
        from viser_isaac_lab import ViserIsaacLab

        viser_viz = ViserIsaacLab(
            asset_dir=Path(args.asset_dir),
            port=args.viser_port,
            update_freq=args.viser_update_freq
        )
        viser_viz.load_from_env(env.unwrapped)

    # In main loop
    while simulation_app.is_running():
        actions = policy(obs)
        obs, _, _, _ = env.step(actions)

        # Update Viser
        if args.viser:
            viser_viz.update_from_env(env.unwrapped)
    """)


if __name__ == "__main__":
    example_usage()
