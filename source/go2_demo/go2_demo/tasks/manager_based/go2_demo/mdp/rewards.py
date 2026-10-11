# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import wrap_to_pi

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def joint_pos_target_l2(env: ManagerBasedRLEnv, target: float, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize joint position deviation from a target value."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    # wrap the joint positions to (-pi, pi)
    joint_pos = wrap_to_pi(asset.data.joint_pos[:, asset_cfg.joint_ids])
    # compute the reward
    return torch.sum(torch.square(joint_pos - target), dim=1)


def energy_new_actual(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    sigma_lin: float = 1000.0,
    sigma_ang: float = 500.0,
    clip_lin: float = 0.2,
    clip_ang: float = 0.2,
) -> torch.Tensor:
    r"""Reward energy-efficient locomotion.

    The numerator is the total absolute mechanical joint power. The denominator
    normalizes it by the robot's forward and yaw speeds::

        exp(-sum_i(|tau_i| |qdot_i|) /
            (sigma_lin |v_x| + sigma_ang |omega_z|))

    Minimum speed values keep the denominator well-conditioned while standing.
    Time-step scaling is intentionally omitted here because the reward manager
    applies it exactly once when combining reward terms.
    """
    asset: Articulation = env.scene[asset_cfg.name]

    joint_power = torch.sum(torch.abs(asset.data.applied_torque * asset.data.joint_vel), dim=1)
    forward_speed = torch.clamp(torch.abs(asset.data.root_lin_vel_b[:, 0]), min=clip_lin)
    yaw_speed = torch.clamp(torch.abs(asset.data.root_ang_vel_b[:, 2]), min=clip_ang)
    speed_scale = sigma_lin * forward_speed + sigma_ang * yaw_speed

    return torch.exp(-joint_power / speed_scale)
