# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg, RslRlSymmetryCfg

from .. import mdp


@configclass
class PPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 16
    max_iterations = 3000
    save_interval = 50
    experiment_name = "go2_velocity_flat_height_baseline"
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_obs_normalization=True,
        critic_obs_normalization=True,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.005,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )


@configclass
class RoughPPORunnerCfg(PPORunnerCfg):
    """PPO configuration with a separate log directory for rough terrain."""

    experiment_name = "go2_velocity_rough_height"

    def __post_init__(self):
        super().__post_init__()
        # Fine-tuning should move behavior into the deterministic policy mean
        # used by play, rather than relying on persistent action noise.
        self.algorithm.entropy_coef = 0.001


@configclass
class LegacyPPORunnerCfg(PPORunnerCfg):
    """Network layout used by the original 48-D flat checkpoint."""

    experiment_name = "go2_velocity"

    def __post_init__(self):
        super().__post_init__()
        self.policy.actor_hidden_dims = [128, 128, 64]
        self.policy.critic_hidden_dims = [128, 128, 64]


@configclass
class HeightTransferPPORunnerCfg(LegacyPPORunnerCfg):
    """Fine-tuning layout for a zero-initialized 187-D height-scan expansion."""

    experiment_name = "go2_velocity_flat_height_transfer"

    def __post_init__(self):
        super().__post_init__()
        self.algorithm.learning_rate = 3.0e-4
        self.algorithm.entropy_coef = 0.001


@configclass
class RoughHeightTransferPPORunnerCfg(HeightTransferPPORunnerCfg):
    """Compatible runner for staged rough-terrain height transfer."""

    experiment_name = "go2_velocity_rough_height_transfer"


@configclass
class StairsHeightTransferPPORunnerCfg(HeightTransferPPORunnerCfg):
    """Runner for stair-only specialization."""

    experiment_name = "go2_velocity_stairs_height_transfer"

    def __post_init__(self):
        super().__post_init__()
        self.algorithm.symmetry_cfg = RslRlSymmetryCfg(
            use_data_augmentation=True,
            use_mirror_loss=True,
            data_augmentation_func=mdp.compute_symmetric_states,
            mirror_loss_coeff=0.1,
        )


@configclass
class HighStepPPORunnerCfg(StairsHeightTransferPPORunnerCfg):
    """Transfer from model_12493, without changing the checkpoint network."""

    experiment_name = "go2_high_step_28cm"
    num_steps_per_env = 32
    max_iterations = 5000
    save_interval = 100

    def __post_init__(self):
        super().__post_init__()
        self.algorithm.learning_rate = 1.0e-4
        # Adaptive PPO can raise 1e-4 to 1e-2; keep transfer updates small while
        # the critic adapts to a new completion reward and shorter episodes.
        self.algorithm.schedule = "fixed"
        self.algorithm.entropy_coef = 0.003
        self.algorithm.symmetry_cfg.mirror_loss_coeff = 0.05


@configclass
class HighStepStandPPORunnerCfg(HighStepPPORunnerCfg):
    experiment_name = "go2_high_step_28cm_stand"
    max_iterations = 2000

    def __post_init__(self):
        super().__post_init__()
        self.algorithm.learning_rate = 5.0e-5
        self.algorithm.entropy_coef = 0.0003
