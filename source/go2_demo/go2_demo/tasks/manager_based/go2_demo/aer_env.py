from __future__ import annotations

import torch

from isaaclab.envs import ManagerBasedEnv, ManagerBasedRLEnv
from isaaclab.managers import CommandManager, CurriculumManager, RewardManager, TerminationManager


class AERRewardManager(RewardManager):
    """Combine task rewards and auxiliary penalties using exponential shaping."""

    # feet_air_time is also positive in the inherited Go2 configuration. It must
    # not be accumulated in the auxiliary-penalty exponent.
    positive_terms = {
        "track_lin_vel_xy_exp",
        "track_ang_vel_z_exp",
        "energy_new_actual",
        "feet_air_time",
    }
    sigma_aux = 0.02

    def compute(self, dt: float) -> torch.Tensor:
        """Compute AER reward, applying ``dt`` once to every reward term."""
        self._reward_buf[:] = 0.0
        positive_reward = torch.zeros_like(self._reward_buf)
        negative_reward = torch.zeros_like(self._reward_buf)

        for term_idx, (name, term_cfg) in enumerate(zip(self._term_names, self._term_cfgs)):
            if term_cfg.weight == 0.0:
                self._step_reward[:, term_idx] = 0.0
                continue

            value = term_cfg.func(self._env, **term_cfg.params) * term_cfg.weight * dt
            self._episode_sums[name] += value
            self._step_reward[:, term_idx] = value / dt

            if name in self.positive_terms:
                positive_reward += value
            else:
                negative_reward += value

        self._reward_buf[:] = positive_reward * torch.exp(negative_reward / self.sigma_aux)
        return self._reward_buf


class AERManagerBasedRLEnv(ManagerBasedRLEnv):
    """Manager-based RL environment using :class:`AERRewardManager`."""

    def load_managers(self):
        # Keep the same manager initialization order as ManagerBasedRLEnv.
        self.command_manager = CommandManager(self.cfg.commands, self)
        print("[INFO] Command Manager: ", self.command_manager)

        # Loads the action, observation, event and recorder managers.
        ManagerBasedEnv.load_managers(self)

        self.termination_manager = TerminationManager(self.cfg.terminations, self)
        print("[INFO] Termination Manager: ", self.termination_manager)

        self.reward_manager = AERRewardManager(self.cfg.rewards, self)
        print("[INFO] AER Reward Manager: ", self.reward_manager)

        self.curriculum_manager = CurriculumManager(self.cfg.curriculum, self)
        print("[INFO] Curriculum Manager: ", self.curriculum_manager)

        self._configure_gym_env_spaces()

        if "startup" in self.event_manager.available_modes:
            self.event_manager.apply(mode="startup")
