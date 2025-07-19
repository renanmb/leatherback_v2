# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for the leatherback robot."""

import os
import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg

# Get absolute path to workspace root
WORKSPACE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__)))

# USD path with proper resolution for cross-platform compatibility
USD_PATH = os.path.join(WORKSPACE_ROOT, "custom_assets", "leatherback_simple_better.usd")

LEATHERBACK_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=USD_PATH,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            rigid_body_enabled=True,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=100.0,
            enable_gyroscopic_forces=True,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=0,
            sleep_threshold=0.005,
            stabilization_threshold=0.001,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.05),
        joint_pos={
            "Wheel__Knuckle__Front_Left": 0.0,
            "Wheel__Knuckle__Front_Right": 0.0,
            "Wheel__Upright__Rear_Right": 0.0,
            "Wheel__Upright__Rear_Left": 0.0,
            "Knuckle__Upright__Front_Right": 0.0,
            "Knuckle__Upright__Front_Left": 0.0,
        },
    ),
    actuators={
        "wheel_drive": ImplicitActuatorCfg(
            joint_names_expr=["Wheel.*"],
            effort_limit=400.0,  # Reduced for more realistic control
            velocity_limit=50.0,  # Reasonable wheel speed limit
            stiffness=0.0,  # For velocity control
            damping=50.0,  # Moderate damping for stability
        ),
        "steering": ImplicitActuatorCfg(
            joint_names_expr=["Knuckle__Upright__Front.*"],
            effort_limit=200.0,  # Sufficient for steering
            velocity_limit=10.0,  # Reasonable steering speed
            stiffness=2000.0,  # Higher stiffness for precise position control
            damping=100.0,  # Good damping for smooth steering without oscillation
        ),
    },
)
