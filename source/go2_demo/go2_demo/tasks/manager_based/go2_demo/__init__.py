# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym

from . import agents

for _demo_name, _demo_cfg in (
    ("Flat", "GO2FlatVideoEnvCfg"),
    ("Rubble", "GO2RubbleVideoEnvCfg"),
    ("Uphill", "GO2UphillVideoEnvCfg"),
    ("Downhill", "GO2DownhillVideoEnvCfg"),
    ("Stairs-Up8cm", "GO2UpstairsVideoEnvCfg"),
    ("Stairs-Down8cm", "GO2DownstairsVideoEnvCfg"),
    ("Step-Down28cm", "GO2StepDown28cmVideoEnvCfg"),
):
    gym.register(
        id=f"Go2-Demo-{_demo_name}-Play-v0",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{__name__}.go2_demo_videos:{_demo_cfg}",
            "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:StairsHeightTransferPPORunnerCfg",
        },
    )

##
# Register Gym environments.
##


gym.register(
    id="Template-Go2-Demo-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.go2_demo_env_cfg:Go2DemoEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:PPORunnerCfg",
    },
)
gym.register(
    id="Go2-Velocity-Flat-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.go2_demo_velicoty:GO2RobotDemoEnv"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:PPORunnerCfg"
        ),
    },
)

gym.register(
    id="Go2-Velocity-Flat-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.go2_demo_velicoty:GO2RobotDemoEnv_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:PPORunnerCfg",
    },
)

gym.register(
    id="Go2-Velocity-Flat-Legacy-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.go2_demo_velicoty:GO2RobotDemoLegacyPlayEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:LegacyPPORunnerCfg",
    },
)

gym.register(
    id="Go2-Velocity-Flat-Height-Transfer-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.go2_demo_velicoty:GO2RobotDemoEnv",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:HeightTransferPPORunnerCfg",
    },
)

gym.register(
    id="Go2-Velocity-Flat-Height-Transfer-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.go2_demo_velicoty:GO2RobotDemoEnv_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:HeightTransferPPORunnerCfg",
    },
)

gym.register(
    id="Go2-Velocity-Rough-Height-Transfer-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.go2_demo_rough:GO2RobotRoughHeightTransferEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:RoughHeightTransferPPORunnerCfg",
    },
)

gym.register(
    id="Go2-Velocity-Rough-Height-Transfer-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.go2_demo_rough:GO2RobotRoughHeightTransferEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:RoughHeightTransferPPORunnerCfg",
    },
)

gym.register(
    id="Go2-Velocity-Stairs-Height-Transfer-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.go2_demo_rough:GO2RobotStairsHeightTransferEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:StairsHeightTransferPPORunnerCfg",
    },
)

gym.register(
    id="Go2-Velocity-Stairs-Height-Transfer-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.go2_demo_rough:GO2RobotStairsHeightTransferEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:StairsHeightTransferPPORunnerCfg",
    },
)

gym.register(
    id="Go2-Velocity-Stairs-Height-Transfer-DR-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.go2_demo_rough:GO2RobotStairsHeightTransferDREnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:StairsHeightTransferPPORunnerCfg",
    },
)

gym.register(
    id="Go2-High-Step-28cm-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.go2_demo_high_step:GO2RobotHighStepEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:HighStepPPORunnerCfg",
    },
)

gym.register(
    id="Go2-High-Step-28cm-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.go2_demo_high_step:GO2RobotHighStepEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:HighStepPPORunnerCfg",
    },
)

gym.register(
    id="Go2-High-Step-28cm-Stand-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.go2_demo_high_step:GO2RobotHighStepStandEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:HighStepStandPPORunnerCfg",
    },
)

gym.register(
    id="Go2-High-Step-28cm-Stand-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.go2_demo_high_step:GO2RobotHighStepStandEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:HighStepStandPPORunnerCfg",
    },
)

gym.register(
    id="Go2-High-Step-28cm-Color-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.go2_demo_two_step:GO2RobotColoredSingleStepEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:HighStepStandPPORunnerCfg",
    },
)

gym.register(
    id="Go2-High-Step-Two-Level-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.go2_demo_two_step:GO2RobotTwoStepEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:HighStepStandPPORunnerCfg",
    },
)

gym.register(
    id="Go2-Velocity-Rough-v0",
    entry_point=f"{__name__}.aer_env:AERManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.go2_demo_rough:GO2RobotRoughEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:RoughPPORunnerCfg",
    },
)

gym.register(
    id="Go2-Velocity-Rough-Play-v0",
    entry_point=f"{__name__}.aer_env:AERManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.go2_demo_rough:GO2RobotRoughEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:RoughPPORunnerCfg",
    },
)
