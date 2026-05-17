# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for RSL-RL agents for cylinder grasping."""

from rsl_rl.algorithms import PPO
from rsl_rl.modules import ActorCritic, ActorCriticRecurrent

from isaaclab.utils import configclass


@configclass
class CylinderGraspingPPORunnerCfg:
    """Configuration for PPO runner for cylinder grasping."""

    # class_name = "rsl_rl.runners.OnPolicyRunner"
    class_name = "rsl_rl.runners.OnPolicyRunner"

    algorithm_class_name = "PPO"
    num_learning_epochs = 5
    num_mini_batches = 4
    learning_rate = 1.0e-3
    schedule = "adaptive"
    gamma = 0.99
    lam = 0.95
    desired_kl = 0.01
    max_grad_norm = 1.0

    # Policy configuration
    policy_init_noise_std = 1.0
    policy_actor_hidden_dims = [512, 256, 128]
    policy_critic_hidden_dims = [512, 256, 128]
    policy_activation = "elu"

    # Algorithm configuration
    algorithm_value_loss_coef = 1.0
    algorithm_use_clipped_value_loss = True
    algorithm_clip_param = 0.2
    algorithm_entropy_coef = 0.0
    algorithm_num_learning_epochs = 5
    algorithm_num_mini_batches = 4
    algorithm_learning_rate = 1.0e-3
    algorithm_schedule = "adaptive"
    algorithm_gamma = 0.99
    algorithm_lam = 0.95
    algorithm_desired_kl = 0.01
    algorithm_max_grad_norm = 1.0

    # Runner configuration
    runner_policy_class_name = "ActorCritic"
    runner_algorithm_class_name = "PPO"
    runner_num_steps_per_env = 24
    runner_max_iterations = 1500
    runner_save_interval = 50
    runner_experiment_name = "cylinder_grasping"
    runner_run_name = ""
    runner_logger = "tensorboard"
    runner_neptune_project = ""
    runner_neptune_run_name = ""
    runner_wandb_project = "cylinder_grasping"
    runner_wandb_entity = ""
    runner_wandb_name = ""
    runner_wandb_group = ""
    runner_resume = False
    runner_load_run = -1
    runner_checkpoint = -1
    runner_num_policies = 1
    runner_policy_save_interval = 50
    runner_empirical_normalization = False
    runner_asym_policy = False
    runner_num_agents = 0
