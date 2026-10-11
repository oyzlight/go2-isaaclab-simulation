"""Left-right symmetry augmentation for Unitree Go2 velocity policies."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from tensordict import TensorDict

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def _mirrored_joint_name(name: str) -> str:
    """Return the matching joint name on the opposite side of the robot."""
    if name.startswith("FL_"):
        return "FR_" + name[3:]
    if name.startswith("FR_"):
        return "FL_" + name[3:]
    if name.startswith("RL_"):
        return "RR_" + name[3:]
    if name.startswith("RR_"):
        return "RL_" + name[3:]
    raise ValueError(f"Cannot mirror unknown Go2 joint name: {name}")


def _mirror_joint_data(env: ManagerBasedRLEnv, data: torch.Tensor) -> torch.Tensor:
    """Swap left/right legs and negate abduction/adduction joints."""
    joint_names = env.scene["robot"].joint_names
    index_by_name = {name: index for index, name in enumerate(joint_names)}
    permutation = [index_by_name[_mirrored_joint_name(name)] for name in joint_names]

    mirrored = data[..., permutation].clone()
    hip_indices = [index for index, name in enumerate(joint_names) if "_hip_joint" in name]
    mirrored[..., hip_indices] *= -1.0
    return mirrored


def _mirror_policy_observation(env: ManagerBasedRLEnv, obs: torch.Tensor) -> torch.Tensor:
    """Reflect the concatenated velocity-policy observation across the sagittal plane."""
    mirrored = obs.clone()
    device = obs.device

    mirrored[:, 0:3] *= torch.tensor([1.0, -1.0, 1.0], device=device)
    mirrored[:, 3:6] *= torch.tensor([-1.0, 1.0, -1.0], device=device)
    mirrored[:, 6:9] *= torch.tensor([1.0, -1.0, 1.0], device=device)
    mirrored[:, 9:12] *= torch.tensor([1.0, -1.0, -1.0], device=device)
    mirrored[:, 12:24] = _mirror_joint_data(env, obs[:, 12:24])
    mirrored[:, 24:36] = _mirror_joint_data(env, obs[:, 24:36])
    mirrored[:, 36:48] = _mirror_joint_data(env, obs[:, 36:48])

    if obs.shape[1] == 235:
        mirrored[:, 48:235] = obs[:, 48:235].view(-1, 11, 17).flip(dims=[1]).reshape(-1, 187)

    return mirrored


@torch.no_grad()
def compute_symmetric_states(
    env: ManagerBasedRLEnv,
    obs: TensorDict | None = None,
    actions: torch.Tensor | None = None,
):
    """Append one left-right mirrored copy of every observation and action."""
    if obs is not None:
        batch_size = obs.batch_size[0]
        obs_aug = obs.repeat(2)
        obs_aug["policy"][:batch_size] = obs["policy"]
        obs_aug["policy"][batch_size:] = _mirror_policy_observation(env.unwrapped, obs["policy"])
    else:
        obs_aug = None

    if actions is not None:
        actions_aug = torch.cat((actions, _mirror_joint_data(env.unwrapped, actions)), dim=0)
    else:
        actions_aug = None

    return obs_aug, actions_aug
