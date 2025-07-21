# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import csv
from datetime import datetime 
import os

import math
import torch
from collections.abc import Sequence

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils.math import sample_uniform

from .leatherback_env_cfg import LeatherbackEnvCfg

from isaaclab.markers import VisualizationMarkers
from isaaclab.assets import RigidObjectCollection
from .cone import CONES_CFG

class LeatherbackEnv(DirectRLEnv):
    cfg: LeatherbackEnvCfg

    def __init__(self, cfg: LeatherbackEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        self._num_goals = self.cfg.num_goals
        self.env_spacing = self.cfg.env_spacing
        self.course_length_coefficient = self.cfg.course_length_coefficient
        self.course_width_coefficient = self.cfg.course_width_coefficient

        self._throttle_dof_idx, _ = self.leatherback.find_joints(self.cfg.throttle_dof_name)
        self._steering_dof_idx, _ = self.leatherback.find_joints(self.cfg.steering_dof_name)

        self._throttle_state =  torch.zeros((self.num_envs,4), device=self.device, dtype=torch.float32)
        self._steering_state =  torch.zeros((self.num_envs,2), device=self.device, dtype=torch.float32)

        self._goal_reached =  torch.zeros((self.num_envs), device=self.device, dtype=torch.int32)
        self.task_completed =  torch.zeros((self.num_envs), device=self.device, dtype=torch.bool)

        self._target_positions =  torch.zeros((self.num_envs, self._num_goals, 2), device=self.device, dtype=torch.float32)
        self._target_index = torch.zeros((self.num_envs), device=self.device, dtype=torch.int32)
        self._markers_pos =  torch.zeros((self.num_envs, self._num_goals, 3), device=self.device, dtype=torch.float32)

        # reward parameters
        """Tolerance for the position of the robot. Defaults to 1cm."""
        self.position_tolerance: float = 0.15
        self.goal_reached_bonus: float = 10.0
        self.position_progress_weight: float = 1.0
        self.heading_coefficient: float = 0.25
        self.heading_progress_weight: float = 0.05

        # region Logger
        # Init for Logger
        # this dont work really well
        self.enable_csv_logging = False  # Set to False to disable logging
        # done using the YAML for the env config
        # self.enable_csv_logging = getattr(self.cfg, "enable_csv_logging", True)
        # Possibly add a CLI args ?
        # Directory for logs
        self.csv_log_dir = os.path.join(os.getcwd(), "leatherback_logs")
        os.makedirs(self.csv_log_dir, exist_ok=True)

        # File names for each environment
        self.csv_filenames = {i: os.path.join(self.csv_log_dir, f"env_{i}_obs.csv") for i in range(self.num_envs)}
        self.csv_step_counter = 0

        # Write CSV headers
        if self.enable_csv_logging:
            for filename in self.csv_filenames.values():
                if not os.path.exists(filename):
                    with open(filename, 'w', newline='') as f:
                        writer = csv.writer(f)
                        writer.writerow([
                            "timestamp",
                            "position_error",
                            "target_heading_cos",
                            "target_heading_sin",
                            "root_lin_vel_x",
                            "root_lin_vel_y",
                            "root_ang_vel_z",
                            "throttle_state",
                            "steering_state",
                            "action_throttle",
                            "action_steering",
                        ])
        # end of region Logger

    # region Setup Scene
    def _setup_scene(self):
        self.leatherback = Articulation(self.cfg.robot_cfg)
        self.Waypoints = VisualizationMarkers(self.cfg.waypoint_cfg)
        self.cones = RigidObjectCollection(self.cfg.cone_collection_cfg) # AttributeError: 'RigidObjectCollection' object has no attribute '_data'. Did you mean: 'data'?
        self.object_state = []
        # Playing with the Ground planes
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

        # clone, filter and replicate
        self.scene.clone_environments(copy_from_source=False) # Clones child environments from parent environment
        self.scene.filter_collisions(global_prim_paths=[])    # Prevents environments from colliding with each other

        # add articulation to scene
        self.scene.articulations["Leatherback"] = self.leatherback
        # Add as a collection
        self.scene.rigid_object_collections["cones"] = self.cones # A dictionary of rigid object collections in the scene.

        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    # region _pre_physics_step
    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        """
        Multiplier for the throttle velocity.
        Multiplier for the steering position. 
        The actions should be in the range [-1, 1] and the radius of the wheel is 0.06m"""
        throttle_scale = 5 # when set to 2 it trains but the cars are flying, 3 you get NaNs
        throttle_max = 50.0 # throttle_max = 60.0
        steering_scale = 0.01 #0.01 # steering_scale = math.pi / 4.0
        steering_max = 0.75
        # region Logging
        if self.enable_csv_logging:
            # Compute observation here temporarily
            obs = self._get_observations()["policy"]
            self._log_observations_to_csv(obs, actions)
        # print(actions)

        self._throttle_action = actions[:, 0].repeat_interleave(4).reshape((-1, 4)) * throttle_scale
        # self._throttle_action += self._throttle_state 
        self.throttle_action = torch.clamp(self._throttle_action, -throttle_max, throttle_max * 1) # negative goes forward and positive goes backward
        self._throttle_state = self.throttle_action
        # print(self.throttle_action)

        self._steering_action = actions[:, 1].repeat_interleave(2).reshape((-1, 2)) * steering_scale # Must add a smooth curve ??
        # self._steering_action += self._steering_state
        self.steering_action = torch.clamp(self._steering_action, -steering_max, steering_max)
        self._steering_state = self.steering_action
        # print(self._steering_action)
    #     # Log data
    #     self.scalar_logger.log("robot_state", "AVG/throttle_action", self._throttle_action[:, 0])
    #     self.scalar_logger.log("robot_state", "AVG/steering_action", self._steering_action[:, 0])
    # end region _pre_physics_step

    def _apply_action(self) -> None:
        self.leatherback.set_joint_velocity_target(self.throttle_action, joint_ids=self._throttle_dof_idx)
        self.leatherback.set_joint_position_target(self._steering_state, joint_ids=self._steering_dof_idx)
    
    # region _get_observations
    def _get_observations(self) -> dict:

        # position error
        current_target_positions = self._target_positions[self.leatherback._ALL_INDICES, self._target_index]
        self._position_error_vector = current_target_positions - self.leatherback.data.root_pos_w[:, :2]
        self._previous_position_error = self._position_error.clone()
        self._position_error = torch.norm(self._position_error_vector, dim=-1) # had placed dim=1

        # heading error
        heading = self.leatherback.data.heading_w
        target_heading_w = torch.atan2(
            self._target_positions[self.leatherback._ALL_INDICES, self._target_index, 1] - self.leatherback.data.root_link_pos_w[:, 1],
            self._target_positions[self.leatherback._ALL_INDICES, self._target_index, 0] - self.leatherback.data.root_link_pos_w[:, 0],
        )
        # cleaning the code
        # replace below by the following: self.target_heading_error = target_heading_w - heading
        # self.target_heading_error = torch.atan2(torch.sin(target_heading_w - heading), torch.cos(target_heading_w - heading))
        self.target_heading_error = target_heading_w - heading

        # Eric Added the Cones to Observations
        obs = torch.cat(
            (
                self._position_error.unsqueeze(dim=1),
                torch.cos(self.target_heading_error).unsqueeze(dim=1), # torch.cos(self.target_heading_error).unsqueeze(dim=1),
                torch.sin(self.target_heading_error).unsqueeze(dim=1), # torch.sin(self.target_heading_error).unsqueeze(dim=1),
                self.leatherback.data.root_lin_vel_b[:, 0].unsqueeze(dim=1),
                self.leatherback.data.root_lin_vel_b[:, 1].unsqueeze(dim=1),
                self.leatherback.data.root_ang_vel_w[:, 2].unsqueeze(dim=1),
                self._throttle_state[:, 0].unsqueeze(dim=1),
                self._steering_state[:, 0].unsqueeze(dim=1),
            ),
            dim=-1,
        )
        # print(obs)
        # TODO add flag to turn logging on/off.
        # Log observations to CSV --- Hacky solution degrades performance
        # if self.enable_csv_logging:
        #     self._log_observations_to_csv(obs)
        
        if torch.any(obs.isnan()):
            raise ValueError("Observations cannot be NAN")

        observations = {"policy": obs}
        return observations
    # end of region _get_observations
    
    # region logging
    # TODO add functionality to log the Observations
    def _log_observations_to_csv(self, obs: torch.Tensor, actions: torch.Tensor):
        """
        Log observations every 5 steps
        """
        if self.csv_step_counter % 5 == 0:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:$S.%f")[:-3]

            # convert observations to numpy for easier handling
            obs_np = obs.cpu().numpy()

            # Write to each environment CSV file
            for env_id in self.csv_filenames.keys():
                row = [
                    timestamp,
                    float(obs_np[env_id, 0]), # position_error
                    float(obs_np[env_id, 1]), # target_heading_cos
                    float(obs_np[env_id, 2]), # target_heading_sin
                    float(obs_np[env_id, 3]), # root_lin_vel_x
                    float(obs_np[env_id, 4]), # root_lin_vel_y
                    float(obs_np[env_id, 5]), # root_ang_vel_z
                    float(obs_np[env_id, 6]), # throttle_state
                    float(obs_np[env_id, 7]), # steering_state
                    float(actions[env_id, 0]), # raw throttle action
                    float(actions[env_id, 1]), # raw steering action
                ]

                # Write to the specific environment's CSV file
                with open(self.csv_filenames[env_id], 'a', newline='') as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerow(row)
        self.csv_step_counter += 1
    
    # region _get_rewards
    def _get_rewards(self) -> torch.Tensor:
        # Trying to Implement TorchScript
        composite_reward, self.task_completed, self._target_index = compute_rewards(
            self.heading_coefficient,
            self.position_tolerance,
            self._num_goals,
            self.position_progress_weight,
            self.heading_progress_weight,
            self.goal_reached_bonus,
            self._previous_position_error,
            self._position_error,
            self.target_heading_error,
            self.task_completed,
            self._target_index,
        )

        # region debugging
        # Update Waypoints so the goal reached waypoint turns blue - marker0 is RED - marker1 is BLUE
        one_hot_encoded = torch.nn.functional.one_hot(self._target_index.long(), num_classes=self._num_goals) # one_hot - all zeros except the target_index
        marker_indices = one_hot_encoded.view(-1).tolist()
        self.Waypoints.visualize(marker_indices=marker_indices)

        return composite_reward
    # end of region _get_rewards
    # region _get_dones
    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        
        task_failed = self.episode_length_buf > self.max_episode_length

        # task completed is calculated in get_rewards before target_index is wrapped around
        return task_failed, self.task_completed
    # end of region _get_dones
    # region _reset_idx
    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = self.leatherback._ALL_INDICES
        super()._reset_idx(env_ids)

        num_reset = len(env_ids)
        # region reset robot
        # reset from config
        default_state = self.leatherback.data.default_root_state[env_ids]   # first there are pos, next 4 quats, next 3 vel,next 3 ang vel, 
        leatherback_pose = default_state[:, :7]                             # proper way of getting default pose from config file
        leatherback_velocities = default_state[:, 7:]                       # proper way of getting default velocities from config file
        joint_positions = self.leatherback.data.default_joint_pos[env_ids]  # proper way to get joint positions from config file
        joint_velocities = self.leatherback.data.default_joint_vel[env_ids] # proper way to get joint velocities from config file

        leatherback_pose[:, :3] += self.scene.env_origins[env_ids] # Adds center of each env position in leatherback position

        # Randomize Steering position at start of track
        leatherback_pose[:, 0] -= self.env_spacing / 2
        leatherback_pose[:, 1] += 2.0 * torch.rand((num_reset), dtype=torch.float32, device=self.device) * self.course_width_coefficient
        # pose at zero on Y-axis
        # leatherback_pose[:, 1] += 0

        # Randomize Starting Heading
        angles = torch.pi / 6.0 * torch.rand((num_reset), dtype=torch.float32, device=self.device)
        # set angle to zero which should be the X-axis
        # angles = torch.zeros((num_reset), dtype=torch.float32, device=self.device)

        # Isaac Sim Quaternions are w first (w, x, y, z) To rotate about the Z axis, we will modify the W and Z values
        leatherback_pose[:, 3] = torch.cos(angles * 0.5)
        leatherback_pose[:, 6] = torch.sin(angles * 0.5)

        self.leatherback.write_root_pose_to_sim(leatherback_pose, env_ids)
        self.leatherback.write_root_velocity_to_sim(leatherback_velocities, env_ids)
        self.leatherback.write_joint_state_to_sim(joint_positions, joint_velocities, None, env_ids)
        # end region reset robot

        # region reset goals
        self._target_positions[env_ids, :, :] = 0.0
        self._markers_pos[env_ids, :, :] = 0.0

        spacing = 2 / self._num_goals
        target_positions = torch.arange(-0.8, 1.1, spacing, device=self.device) * self.env_spacing / self.course_length_coefficient
        self._target_positions[env_ids, :len(target_positions), 0] = target_positions
        # varies the target position in th Y-axis
        self._target_positions[env_ids, :, 1] = torch.rand((num_reset, self._num_goals), dtype=torch.float32, device=self.device) + self.course_length_coefficient
        # Generate same track straight line
        # self._target_positions[env_ids, :, 1] = 0
        self._target_positions[env_ids, :] += self.scene.env_origins[env_ids, :2].unsqueeze(1)

        self._target_index[env_ids] = 0

        # Update the visual markers
        self._markers_pos[env_ids, :, :2] = self._target_positions[env_ids]
        visualize_pos = self._markers_pos.view(-1, 3)
        self.Waypoints.visualize(translations=visualize_pos)
        # end Region Reset Goals

        # region Cones
        num_objects = len(CONES_CFG.rigid_objects)
        self.object_state = self.cones.data.default_object_state.clone() # 	Default object state [pos, quat, lin_vel, ang_vel] in local environment frame.
        object_ids = torch.arange(num_objects, device=self.device)  # Each object has a unique index
        
        # Reset Cones
        offset = 0.75
        self.cone_positions1 = self._target_positions[env_ids]
        self.cone_positions2 = self._target_positions[env_ids]
        offset = torch.full((num_reset, self._num_goals), device=self.device, fill_value=offset, dtype=torch.float32)
        sign_pattern = torch.tensor([1 if j% 2 == 0 else -1 for j in range(self._num_goals)], device=self.device)
        offset[:, :] *= sign_pattern
        self.cone_positions1[:, :, 1] += offset # Tensor size self._num_goals
        self.cone_positions2[:, :, 1] -= offset # Tensor size self._num_goals
        self.cone_positions = torch.cat([self.cone_positions1 ,self.cone_positions2 ], dim=1)

        # The idea is to add the self.cone_positions to the self.object_state with the correct Cone position for each Cone in the ENV
        # Pad the tensor to have 3 dimensions by adding a column of zeros for the third dimension (z)
        # padded_cone_positions = torch.cat((self.cone_positions, 
        #     torch.zeros(
        #         self.cone_positions.shape[0], 
        #         self.cone_positions.shape[1], 
        #         1, 
        #         device=self.cone_positions.device
        #         )
        #     ), dim=2)
        
        # Instead of zeroes controlling the Z value it resets the cones
        z_value = 0.0
        batch_size, num_cones, _ = self.cone_positions.shape

        # Create a tensor filled with z_value
        z_vals = torch.full(
            (batch_size, num_cones, 1), 
            fill_value=z_value, 
            device=self.cone_positions.device
        )
        padded_cone_positions = torch.cat((self.cone_positions, z_vals), dim=2)
        self.object_state[env_ids, :, :3] = padded_cone_positions
        # print(self.object_state)
        # print(f"env_ids before function call: {env_ids}, type: {type(env_ids)}, device: {env_ids.device if isinstance(env_ids, torch.Tensor) else 'CPU'}")
        self.cones.write_object_link_pose_to_sim(self.object_state[env_ids, :, :7], env_ids, object_ids) # Set the object pose over selected environment and object indices into the simulation.

        # reset positions error
        current_target_positions = self._target_positions[self.leatherback._ALL_INDICES, self._target_index]
        self._position_error_vector = current_target_positions[:, :2] - self.leatherback.data.root_pos_w[:, :2]
        self._position_error = torch.norm(self._position_error_vector, dim=-1)
        self._previous_position_error = self._position_error.clone()

        # reset heading error
        heading = self.leatherback.data.heading_w[:]
        target_heading_w = torch.atan2( 
            self._target_positions[:, 0, 1] - self.leatherback.data.root_pos_w[:, 1],
            self._target_positions[:, 0, 0] - self.leatherback.data.root_pos_w[:, 0],
        )
        self._heading_error = torch.atan2(torch.sin(target_heading_w - heading), torch.cos(target_heading_w - heading))
        self._previous_heading_error = self._heading_error.clone()
        # end region

@torch.jit.script
def compute_rewards(
    heading_coefficient: float,
    position_tolerance: float,
    _num_goals: int,
    position_progress_weight: float,
    heading_progress_weight: float,
    goal_reached_bonus: float,
    _previous_position_error: torch.Tensor,
    _position_error: torch.Tensor,
    target_heading_error: torch.Tensor,
    task_completed: torch.Tensor,
    _target_index: torch.Tensor,
):
    # position progress
    position_progress_rew = _previous_position_error - _position_error
    # Heading Distance - changing the numerator to positive make it drive backwards
    target_heading_rew = torch.exp(-torch.abs(target_heading_error) / heading_coefficient)
    # Checks if the goal is reached
    goal_reached = _position_error < position_tolerance
    # if the goal is reached, the target index is updated
    _target_index = _target_index + goal_reached
    task_completed = _target_index > (_num_goals -1)
    _target_index = _target_index % _num_goals

    composite_reward = (
        position_progress_rew*position_progress_weight +
        target_heading_rew*heading_progress_weight +
        goal_reached*goal_reached_bonus
    )

    if torch.any(composite_reward.isnan()):
        raise ValueError("Rewards cannot be NAN")

    return composite_reward, task_completed, _target_index