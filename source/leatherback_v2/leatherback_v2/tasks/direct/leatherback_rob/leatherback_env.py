from __future__ import annotations

import torch
import csv
import os
import datetime
import math
from collections.abc import Sequence
import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils import configclass
from .waypoint import WAYPOINT_CFG
from .leatherback import LEATHERBACK_CFG
from isaaclab.markers import VisualizationMarkers

@configclass
class LeatherbackEnvCfg(DirectRLEnvCfg):
    decimation = 4
    episode_length_s = 20.0
    action_space = 2  # [forward_speed, steering_angle] for Ackermann
    observation_space = 9  # Increased by 1 for current_speed observation
    state_space = 0
    sim: SimulationCfg = SimulationCfg(dt=1 / 60, render_interval=decimation)
    robot_cfg: ArticulationCfg = LEATHERBACK_CFG.replace(prim_path="/World/envs/env_.*/Robot")
    waypoint_cfg = WAYPOINT_CFG

    # Ackermann vehicle parameters
    wheelbase: float = 2.5  # Distance between front and rear axles (meters)
    track_width: float = 1.8  # Distance between left and right wheels (meters)
    wheel_radius: float = 0.3  # Wheel radius (meters)
    
    # Action scaling
    max_speed: float = 5.0  # Maximum forward speed (m/s)
    max_steering_angle: float = 0.5  # Maximum steering angle (radians, ~30 degrees)

    # Joint names for Ackermann control
    wheel_joint_names = [
        "Wheel__Knuckle__Front_Left",
        "Wheel__Knuckle__Front_Right", 
        "Wheel__Upright__Rear_Left",
        "Wheel__Upright__Rear_Right"
    ]
    steering_joint_names = [
        "Knuckle__Upright__Front_Left",
        "Knuckle__Upright__Front_Right",
    ]

    # Legacy support - keeping for backward compatibility but not used in Ackermann
    throttle_dof_name = wheel_joint_names
    steering_dof_name = steering_joint_names

    env_spacing = 32.0
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=4096, env_spacing=env_spacing, replicate_physics=True)

class LeatherbackEnv(DirectRLEnv):
    cfg: LeatherbackEnvCfg

    def __init__(self, cfg: LeatherbackEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        
        # Get joint indices for Ackermann control
        self._wheel_dof_idx, _ = self.leatherback.find_joints(self.cfg.wheel_joint_names)
        self._steering_dof_idx, _ = self.leatherback.find_joints(self.cfg.steering_joint_names)
        
        # Legacy support
        self._throttle_dof_idx = self._wheel_dof_idx
        
        # Ackermann control state variables
        self._wheel_velocities = torch.zeros((self.num_envs, 4), device=self.device, dtype=torch.float32)
        self._steering_angles = torch.zeros((self.num_envs, 2), device=self.device, dtype=torch.float32)
        self._current_speed = torch.zeros((self.num_envs, 1), device=self.device, dtype=torch.float32)
        
        # Legacy state variables (for compatibility)
        self._throttle_state = torch.zeros((self.num_envs,4), device=self.device, dtype=torch.float32)
        self._steering_state = torch.zeros((self.num_envs,2), device=self.device, dtype=torch.float32)
        
        self._last_actions = torch.zeros((self.num_envs, 2), device=self.device, dtype=torch.float32)
        self._smoothed_actions = torch.zeros((self.num_envs, 2), device=self.device, dtype=torch.float32)
        self._action_history = torch.zeros((self.num_envs, 5, 2), device=self.device, dtype=torch.float32)  # 5-step history
        self._goal_reached = torch.zeros((self.num_envs), device=self.device, dtype=torch.int32)
        self.task_completed = torch.zeros((self.num_envs), device=self.device, dtype=torch.bool)
        self._num_goals = 10
        self._target_positions = torch.zeros((self.num_envs, self._num_goals, 2), device=self.device, dtype=torch.float32)
        self._markers_pos = torch.zeros((self.num_envs, self._num_goals, 3), device=self.device, dtype=torch.float32)
        self.env_spacing = self.cfg.env_spacing
        self.course_length_coefficient = 2.5
        self.course_width_coefficient = 2.0
        self.position_tolerance = 0.15
        self.goal_reached_bonus = 10.0
        self.position_progress_weight = 1.0
        self.heading_coefficient = 0.25
        self.heading_progress_weight = 0.05
        self._target_index = torch.zeros((self.num_envs), device=self.device, dtype=torch.int32)
        
        # Action smoothing parameters
        self.action_smoothing_factor = 0.3  # How much to smooth (0=no smoothing, 1=maximum smoothing)
        self.max_steering_rate = 0.5  # Maximum steering change per timestep
        self.max_speed_rate = 1.0     # Maximum speed change per timestep
        
        # Setup CSV logging for instability detection
        self._setup_instability_logging()

    def _setup_scene(self):
        # Create a large ground plane without grid
        spawn_ground_plane(
            prim_path="/World/ground",
            cfg=GroundPlaneCfg(
                size=(500.0, 500.0),  # Much larger ground plane (500m x 500m)
                color=(0.2, 0.2, 0.2),  # Dark gray color
                physics_material=sim_utils.RigidBodyMaterialCfg(
                    friction_combine_mode="multiply",
                    restitution_combine_mode="multiply",
                    static_friction=1.0,
                    dynamic_friction=1.0,
                    restitution=0.0,
                ),
            ),
        )

        # Setup rest of the scene
        self.leatherback = Articulation(self.cfg.robot_cfg)
        self.waypoints = VisualizationMarkers(self.cfg.waypoint_cfg)
        self.object_state = []
        
        self.scene.clone_environments(copy_from_source=False)
        self.scene.filter_collisions(global_prim_paths=[])
        self.scene.articulations["leatherback"] = self.leatherback

        # Add lighting
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        # Store raw actions for debugging
        self._last_actions = actions.clone()
        
        # Check for NaN or infinite actions
        if torch.any(torch.isnan(actions)) or torch.any(torch.isinf(actions)):
            print(f"[WARNING] Invalid actions detected at step {getattr(self, 'common_step_counter', 0)}")
            # Replace invalid actions with zeros
            actions = torch.where(torch.isnan(actions) | torch.isinf(actions), 
                                torch.zeros_like(actions), actions)
        
        # Clamp actions to safe range before smoothing
        actions = torch.clamp(actions, -1.0, 1.0)
        
        # Apply action smoothing and rate limiting for more stable control
        actions = self._apply_action_smoothing(actions)
        
        # Extract Ackermann actions: [forward_speed, steering_angle]
        forward_speed = actions[:, 0] * self.cfg.max_speed  # Scale to max speed
        steering_angle = actions[:, 1] * self.cfg.max_steering_angle  # Scale to max steering angle
        
        # Additional safety clamps after scaling
        forward_speed = torch.clamp(forward_speed, -self.cfg.max_speed, self.cfg.max_speed)
        steering_angle = torch.clamp(steering_angle, -self.cfg.max_steering_angle, self.cfg.max_steering_angle)
        
        # Compute Ackermann steering angles for left and right wheels
        left_steering, right_steering = self._compute_ackermann_angles(steering_angle)
        
        # Compute individual wheel velocities based on Ackermann geometry
        wheel_velocities = self._compute_wheel_velocities(forward_speed, steering_angle)
        
        # Safety checks on computed values
        if torch.any(torch.isnan(wheel_velocities)) or torch.any(torch.isinf(wheel_velocities)):
            print(f"[WARNING] Invalid wheel velocities at step {getattr(self, 'common_step_counter', 0)}")
            wheel_velocities = torch.zeros_like(wheel_velocities)
            
        if torch.any(torch.isnan(left_steering)) or torch.any(torch.isnan(right_steering)):
            print(f"[WARNING] Invalid steering angles at step {getattr(self, 'common_step_counter', 0)}")
            left_steering = torch.zeros_like(left_steering)
            right_steering = torch.zeros_like(right_steering)
        
        # Store computed values
        self._wheel_velocities = wheel_velocities
        self._steering_angles = torch.stack([left_steering, right_steering], dim=1)
        self._current_speed = torch.norm(self.leatherback.data.root_lin_vel_b[:, :2], dim=1, keepdim=True)
        
        # Update legacy state variables for compatibility
        self._throttle_state = wheel_velocities
        self._steering_state = self._steering_angles

    def _compute_ackermann_angles(self, steering_angle: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute left and right steering angles using Ackermann geometry"""
        # Clamp steering angle to prevent division by zero and numerical issues
        steering_angle = torch.clamp(steering_angle, -self.cfg.max_steering_angle * 0.95, 
                                   self.cfg.max_steering_angle * 0.95)
        
        # Handle near-zero steering angles to avoid numerical issues
        small_angle_mask = torch.abs(steering_angle) < 1e-4
        safe_steering_angle = torch.where(small_angle_mask, 
                                        torch.sign(steering_angle) * 1e-4, 
                                        steering_angle)
        
        # Compute turning radius with safety checks
        tan_angle = torch.tan(torch.abs(safe_steering_angle))
        tan_angle = torch.clamp(tan_angle, min=1e-6)  # Prevent division by zero
        turning_radius = self.cfg.wheelbase / tan_angle
        
        # Clamp turning radius to reasonable bounds
        turning_radius = torch.clamp(turning_radius, min=2.0, max=1000.0)
        
        # Ackermann steering equations with safety
        left_radius = turning_radius + torch.where(safe_steering_angle > 0, 
                                                 -self.cfg.track_width / 2,
                                                 self.cfg.track_width / 2)
        right_radius = turning_radius + torch.where(safe_steering_angle > 0,
                                                  self.cfg.track_width / 2,
                                                  -self.cfg.track_width / 2)
        
        # Ensure radii are not too small
        left_radius = torch.clamp(torch.abs(left_radius), min=1.0) * torch.sign(left_radius)
        right_radius = torch.clamp(torch.abs(right_radius), min=1.0) * torch.sign(right_radius)
        
        left_angle = torch.atan(self.cfg.wheelbase / left_radius)
        right_angle = torch.atan(self.cfg.wheelbase / right_radius)
        
        # Apply steering direction and clamp final angles
        left_angle = torch.where(safe_steering_angle > 0, left_angle, -left_angle)
        right_angle = torch.where(safe_steering_angle > 0, right_angle, -right_angle)
        
        # Final safety clamp
        left_angle = torch.clamp(left_angle, -self.cfg.max_steering_angle, self.cfg.max_steering_angle)
        right_angle = torch.clamp(right_angle, -self.cfg.max_steering_angle, self.cfg.max_steering_angle)
        
        return left_angle, right_angle
    
    def _compute_wheel_velocities(self, forward_speed: torch.Tensor, steering_angle: torch.Tensor) -> torch.Tensor:
        """Compute individual wheel velocities for Ackermann steering"""
        # Clamp input values for safety
        forward_speed = torch.clamp(forward_speed, -self.cfg.max_speed, self.cfg.max_speed)
        steering_angle = torch.clamp(steering_angle, -self.cfg.max_steering_angle * 0.95, 
                                   self.cfg.max_steering_angle * 0.95)
        
        # Angular velocity of vehicle about its center
        # Use safe division to prevent numerical issues
        safe_steering = torch.where(torch.abs(steering_angle) < 1e-4,
                                  torch.sign(steering_angle) * 1e-4,
                                  steering_angle)
        
        angular_vel = forward_speed * torch.tan(safe_steering) / self.cfg.wheelbase
        
        # Clamp angular velocity to reasonable bounds
        max_angular_vel = self.cfg.max_speed / (self.cfg.track_width / 2)  # Based on max speed and track width
        angular_vel = torch.clamp(angular_vel, -max_angular_vel, max_angular_vel)
        
        # Linear velocities at each wheel position
        track_half = self.cfg.track_width / 2
        
        # Front wheels
        v_fl = forward_speed + angular_vel * track_half  # Front left
        v_fr = forward_speed - angular_vel * track_half  # Front right
        
        # Rear wheels (no steering, same as vehicle center for rear-wheel reference)
        v_rl = forward_speed + angular_vel * track_half  # Rear left
        v_rr = forward_speed - angular_vel * track_half  # Rear right
        
        # Convert linear velocity to angular velocity (rad/s) for wheel joints
        wheel_linear_vels = torch.stack([v_fl, v_fr, v_rl, v_rr], dim=1)
        
        # Safety check: ensure wheel radius is not zero
        safe_wheel_radius = max(self.cfg.wheel_radius, 0.01)
        wheel_angular_vels = wheel_linear_vels / safe_wheel_radius
        
        # Clamp wheel velocities to reasonable bounds (prevent excessive speeds)
        max_wheel_vel = 100.0  # rad/s, about 1800 RPM for 0.3m wheel
        wheel_angular_vels = torch.clamp(wheel_angular_vels, -max_wheel_vel, max_wheel_vel)
        
        return wheel_angular_vels

    def _apply_action(self) -> None:
        # Apply wheel velocities
        self.leatherback.set_joint_velocity_target(self._wheel_velocities, joint_ids=self._wheel_dof_idx)
        
        # Apply steering angles
        self.leatherback.set_joint_position_target(self._steering_angles, joint_ids=self._steering_dof_idx)

    def _get_observations(self) -> dict:
        current_target_positions = self._target_positions[self.leatherback._ALL_INDICES, self._target_index]
        self._position_error_vector = current_target_positions - self.leatherback.data.root_pos_w[:, :2]
        self._previous_position_error = getattr(self, '_position_error', torch.zeros_like(self._position_error_vector[:, 0]))
        self._position_error = torch.norm(self._position_error_vector, dim=-1)

        heading = self.leatherback.data.heading_w
        target_heading_w = torch.atan2(
            self._target_positions[self.leatherback._ALL_INDICES, self._target_index, 1] - self.leatherback.data.root_link_pos_w[:, 1],
            self._target_positions[self.leatherback._ALL_INDICES, self._target_index, 0] - self.leatherback.data.root_link_pos_w[:, 0],
        )
        self.target_heading_error = torch.atan2(torch.sin(target_heading_w - heading), torch.cos(target_heading_w - heading))

        # Update current speed with safety check
        lin_vel = self.leatherback.data.root_lin_vel_b[:, :2]
        if torch.any(torch.isnan(lin_vel)) or torch.any(torch.isinf(lin_vel)):
            lin_vel = torch.zeros_like(lin_vel)
        self._current_speed = torch.norm(lin_vel, dim=1, keepdim=True)

        # Safety checks on all observation components
        def safe_tensor(tensor, default_val=0.0):
            if torch.any(torch.isnan(tensor)) or torch.any(torch.isinf(tensor)):
                return torch.full_like(tensor, default_val)
            return tensor

        obs_components = [
            safe_tensor(self._position_error.unsqueeze(dim=1)),
            safe_tensor(torch.cos(self.target_heading_error).unsqueeze(dim=1)),
            safe_tensor(torch.sin(self.target_heading_error).unsqueeze(dim=1)),
            safe_tensor(self.leatherback.data.root_lin_vel_b[:, 0].unsqueeze(dim=1)),  # Forward velocity
            safe_tensor(self.leatherback.data.root_lin_vel_b[:, 1].unsqueeze(dim=1)),  # Lateral velocity
            safe_tensor(self.leatherback.data.root_ang_vel_w[:, 2].unsqueeze(dim=1)),  # Yaw rate
            safe_tensor(self._current_speed),  # Current speed magnitude
            safe_tensor(self._last_actions[:, 0].unsqueeze(dim=1)),  # Last forward speed action
            safe_tensor(self._last_actions[:, 1].unsqueeze(dim=1)),  # Last steering action
        ]

        obs = torch.cat(obs_components, dim=-1)
        
        if torch.any(obs.isnan()):
            # Log detailed instability data to CSV
            self._log_instability_to_csv(obs)
            
            # Brief console notification
            nan_envs = torch.any(obs.isnan(), dim=1)
            affected_envs = torch.sum(nan_envs).item()
            print(f"[INSTABILITY] Step {self.common_step_counter}: {affected_envs}/{self.num_envs} envs affected. Data logged to CSV.")
            
            raise ValueError("Observations cannot be NAN")

        return {"policy": obs}
    
    def _get_rewards(self) -> torch.Tensor:
        position_progress_rew = self._previous_position_error - self._position_error
        target_heading_rew = torch.exp(-torch.abs(self.target_heading_error) / self.heading_coefficient)
        goal_reached = self._position_error < self.position_tolerance
        self._target_index = self._target_index + goal_reached
        self.task_completed = self._target_index > (self._num_goals -1)
        self._target_index = self._target_index % self._num_goals

        composite_reward = (
            position_progress_rew * self.position_progress_weight +
            target_heading_rew * self.heading_progress_weight +
            goal_reached * self.goal_reached_bonus
        )

        one_hot_encoded = torch.nn.functional.one_hot(self._target_index.long(), num_classes=self._num_goals)
        marker_indices = one_hot_encoded.view(-1).tolist()
        self.waypoints.visualize(marker_indices=marker_indices)

        if torch.any(composite_reward.isnan()):
            raise ValueError("Rewards cannot be NAN")

        return composite_reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        task_failed = self.episode_length_buf > self.max_episode_length
        return task_failed, self.task_completed

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = self.leatherback._ALL_INDICES
        super()._reset_idx(env_ids)

        num_reset = len(env_ids)
        default_state = self.leatherback.data.default_root_state[env_ids]
        leatherback_pose = default_state[:, :7]
        leatherback_velocities = default_state[:, 7:]
        joint_positions = self.leatherback.data.default_joint_pos[env_ids]
        joint_velocities = self.leatherback.data.default_joint_vel[env_ids]

        leatherback_pose[:, :3] += self.scene.env_origins[env_ids]
        leatherback_pose[:, 0] -= self.env_spacing / 2
        leatherback_pose[:, 1] += 2.0 * torch.rand((num_reset), dtype=torch.float32, device=self.device) * self.course_width_coefficient

        angles = torch.pi / 6.0 * torch.rand((num_reset), dtype=torch.float32, device=self.device)
        leatherback_pose[:, 3] = torch.cos(angles * 0.5)
        leatherback_pose[:, 6] = torch.sin(angles * 0.5)

        self.leatherback.write_root_pose_to_sim(leatherback_pose, env_ids)
        self.leatherback.write_root_velocity_to_sim(leatherback_velocities, env_ids)
        self.leatherback.write_joint_state_to_sim(joint_positions, joint_velocities, None, env_ids)

        self._target_positions[env_ids, :, :] = 0.0
        self._markers_pos[env_ids, :, :] = 0.0

        spacing = 2 / self._num_goals
        target_positions = torch.arange(-0.8, 1.1, spacing, device=self.device) * self.env_spacing / self.course_length_coefficient
        self._target_positions[env_ids, :len(target_positions), 0] = target_positions
        self._target_positions[env_ids, :, 1] = torch.rand((num_reset, self._num_goals), dtype=torch.float32, device=self.device) + self.course_length_coefficient
        self._target_positions[env_ids, :] += self.scene.env_origins[env_ids, :2].unsqueeze(1)

        self._target_index[env_ids] = 0
        self._markers_pos[env_ids, :, :2] = self._target_positions[env_ids]
        visualize_pos = self._markers_pos.view(-1, 3)
        self.waypoints.visualize(translations=visualize_pos)

        current_target_positions = self._target_positions[self.leatherback._ALL_INDICES, self._target_index]
        self._position_error_vector = current_target_positions[:, :2] - self.leatherback.data.root_pos_w[:, :2]
        self._position_error = torch.norm(self._position_error_vector, dim=-1)
        self._previous_position_error = self._position_error.clone()

        heading = self.leatherback.data.heading_w[:]
        target_heading_w = torch.atan2( 
            self._target_positions[:, 0, 1] - self.leatherback.data.root_pos_w[:, 1],
            self._target_positions[:, 0, 0] - self.leatherback.data.root_pos_w[:, 0],
        )
        self._heading_error = torch.atan2(torch.sin(target_heading_w - heading), torch.cos(target_heading_w - heading))
        self._previous_heading_error = self._heading_error.clone()
        
        # Initialize action smoothing variables for reset environments
        if hasattr(self, '_smoothed_actions'):
            self._smoothed_actions[env_ids] = 0.0
        if hasattr(self, '_action_history'):
            self._action_history[env_ids] = 0.0

    def _setup_instability_logging(self):
        """Setup CSV logging for instability detection and analysis."""
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        log_dir = "logs/instability_analysis"
        os.makedirs(log_dir, exist_ok=True)
        
        csv_filename = f"instability_analysis_{timestamp}.csv"
        self._instability_csv_path = os.path.join(log_dir, csv_filename)
        self._instability_csv_file = None
        self._instability_csv_writer = None
        self._csv_initialized = False
        print(f"[INFO] Instability logging setup: {self._instability_csv_path}")
    
    def _log_instability_to_csv(self, obs):
        """Log detailed instability data to CSV file."""
        if self._instability_csv_file is None:
            self._instability_csv_file = open(self._instability_csv_path, 'w', newline='')
        
        if not self._csv_initialized:
            # Initialize CSV headers
            headers = [
                "step", "timestamp",
                # Action statistics
                "raw_action_min", "raw_action_max", "raw_action_mean", "raw_action_std",
                "wheel_vel_min", "wheel_vel_max", "wheel_vel_mean",
                "steering_angle_min", "steering_angle_max", "steering_angle_mean",
                # Physics state
                "root_pos_min", "root_pos_max", "root_pos_mean",
                "root_vel_min", "root_vel_max", "root_vel_mean",
                "root_ang_vel_min", "root_ang_vel_max", "root_ang_vel_mean",
                # Observation components
                "position_error_min", "position_error_max", "position_error_nan_count",
                "heading_error_min", "heading_error_max", "heading_error_nan_count", 
                "lin_vel_nan_count", "ang_vel_nan_count",
                # Environment info
                "affected_envs", "total_envs", "first_nan_env_id"
            ]
            
            self._instability_csv_writer = csv.writer(self._instability_csv_file)
            self._instability_csv_writer.writerow(headers)
            self._csv_initialized = True
        
        # Collect data
        nan_envs = torch.any(obs.isnan(), dim=1)
        affected_envs = torch.sum(nan_envs).item()
        first_nan_env = torch.where(nan_envs)[0][0].item() if affected_envs > 0 else -1
        
        # Safe min/max calculations that handle NaN values
        def safe_min_max_mean(tensor):
            valid_tensor = tensor[~tensor.isnan()]
            if len(valid_tensor) > 0:
                return valid_tensor.min().item(), valid_tensor.max().item(), valid_tensor.mean().item()
            return float('nan'), float('nan'), float('nan')
        
        raw_actions_min, raw_actions_max, raw_actions_mean = safe_min_max_mean(self._last_actions)
        wheel_vel_min, wheel_vel_max, wheel_vel_mean = safe_min_max_mean(self._wheel_velocities)
        steering_min, steering_max, steering_mean = safe_min_max_mean(self._steering_angles)
        
        pos_min, pos_max, pos_mean = safe_min_max_mean(self.leatherback.data.root_pos_w)
        vel_min, vel_max, vel_mean = safe_min_max_mean(self.leatherback.data.root_lin_vel_b)
        ang_vel_min, ang_vel_max, ang_vel_mean = safe_min_max_mean(self.leatherback.data.root_ang_vel_w)
        
        pos_err_min, pos_err_max, _ = safe_min_max_mean(self._position_error)
        head_err_min, head_err_max, _ = safe_min_max_mean(self.target_heading_error)
        
        # Write data row
        row_data = [
            self.common_step_counter, datetime.datetime.now().isoformat(),
            raw_actions_min, raw_actions_max, raw_actions_mean, self._last_actions.std().item(),
            wheel_vel_min, wheel_vel_max, wheel_vel_mean,
            steering_min, steering_max, steering_mean,
            pos_min, pos_max, pos_mean,
            vel_min, vel_max, vel_mean,
            ang_vel_min, ang_vel_max, ang_vel_mean,
            pos_err_min, pos_err_max, torch.sum(self._position_error.isnan()).item(),
            head_err_min, head_err_max, torch.sum(self.target_heading_error.isnan()).item(),
            torch.sum(self.leatherback.data.root_lin_vel_b.isnan()).item(),
            torch.sum(self.leatherback.data.root_ang_vel_w.isnan()).item(),
            affected_envs, self.num_envs, first_nan_env
        ]
        
        self._instability_csv_writer.writerow(row_data)
        self._instability_csv_file.flush()  # Ensure data is written immediately

    def _apply_action_smoothing(self, actions: torch.Tensor) -> torch.Tensor:
        """
        Apply action smoothing and rate limiting to reduce oscillations.
        
        Args:
            actions: Raw actions from the policy [num_envs, 2]
            
        Returns:
            Smoothed actions [num_envs, 2]
        """
        # Initialize smoothed actions on first call
        if not hasattr(self, '_smoothed_actions_initialized'):
            self._smoothed_actions[:] = actions
            self._smoothed_actions_initialized = True
            return actions.clone()
        
        # Exponential moving average smoothing
        # smoothed = alpha * current + (1 - alpha) * previous
        alpha = 1.0 - self.action_smoothing_factor
        self._smoothed_actions = alpha * actions + self.action_smoothing_factor * self._smoothed_actions
        
        # Rate limiting: restrict maximum change per timestep
        action_delta = actions - self._smoothed_actions
        
        # Limit speed changes
        speed_delta = torch.clamp(action_delta[:, 0], -self.max_speed_rate, self.max_speed_rate)
        
        # Limit steering changes
        steering_delta = torch.clamp(action_delta[:, 1], -self.max_steering_rate, self.max_steering_rate)
        
        # Apply rate-limited changes
        rate_limited_actions = self._smoothed_actions + torch.stack([speed_delta, steering_delta], dim=1)
        
        # Update smoothed actions and ensure they stay within bounds
        self._smoothed_actions = torch.clamp(rate_limited_actions, -1.0, 1.0)
        
        return self._smoothed_actions.clone()

    def close(self):
        """Clean up resources including CSV logging."""
        if hasattr(self, '_instability_csv_file') and self._instability_csv_file is not None:
            self._instability_csv_file.close()
            print(f"[INFO] Instability CSV logging closed: {self._instability_csv_path}")
        super().close()