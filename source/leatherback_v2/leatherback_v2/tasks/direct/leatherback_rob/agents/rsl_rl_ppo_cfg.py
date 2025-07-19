# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg


@configclass
class LeatherbackPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """Configuration for training Leatherback with RSL-RL PPO."""
    
    # Training parameters (matching SKRL config exactly)
    num_steps_per_env = 32  # rollouts in SKRL
    max_iterations = 500    # Increased for better convergence
    save_interval = 50
    experiment_name = "leatherback_rob_direct"
    empirical_normalization = False
    
    # Note: Action clipping will be handled in play.py for inference
    # clip_actions = True  # Disabled due to type incompatibility with RSL-RL wrapper
    
    policy = RslRlPpoActorCriticCfg(
        # Network architecture - Reduced noise for smoother actions
        init_noise_std=0.3,  # Reduced from 1.0 to prevent erratic exploration
        actor_hidden_dims=[64, 64],  # Increased from [32, 32] for better policy representation
        critic_hidden_dims=[64, 64],  # Increased for better value estimation
        activation="elu",
    )
    
    algorithm = RslRlPpoAlgorithmCfg(
        # Tuned for smoother, more stable policy learning
        value_loss_coef=2.0,     # value_loss_scale in SKRL
        use_clipped_value_loss=True,  # clip_predicted_values in SKRL
        clip_param=0.1,          # Reduced from 0.2 for more conservative updates
        entropy_coef=0.05,       # Reduced from 0.2 to discourage random actions
        num_learning_epochs=10,  # Increased from 8 for better convergence
        num_mini_batches=8,      # mini_batches in SKRL
        learning_rate=3e-4,      # Reduced from 5e-4 for more stable learning
        schedule="adaptive",     # learning_rate_scheduler: KLAdaptiveLR
        #schedule="linear",     # learning_rate_scheduler: KLAdaptiveLR
        gamma=0.99,              # discount_factor in SKRL
        lam=0.95,                # lambda in SKRL
        desired_kl=0.008,        # kl_threshold in SKRL
        max_grad_norm=0.5,       # Reduced from 1.0 for more stable gradients
    )
