"""Goal-directed high-step rewards and an outcome-based curriculum.

Only the command generator uses the goal. The policy still receives exactly the
original 48 proprioceptive + 187 height-scan observations. Geometry/contacts used
by rewards and evaluation are not appended to the policy input.
"""

import torch

from isaaclab.envs.mdp.commands import UniformVelocityCommand
from isaaclab.utils.math import wrap_to_pi


def _target_geometry(cfg):
    """Final ledge and total platform height; unchanged for existing single steps."""
    count = getattr(cfg, "step_count", 1)
    edge = cfg.ledge_distance + (count - 1) * getattr(cfg, "step_depth", 0.0)
    return edge, tuple(count * height for height in cfg.height_levels)


class HighStepVelocityCommand(UniformVelocityCommand):
    """Approach in world +X, stay in the lane, and slow down on the platform."""

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.approach_speed = torch.zeros(self.num_envs, device=self.device)

    def _resample_command(self, env_ids):
        super()._resample_command(env_ids)
        self.approach_speed[env_ids] = self.vel_command_b[env_ids, 0]
        # CommandManager.reset resamples but does not call _update_command.
        self._update_command()

    def _update_command(self):
        cfg = self._env.cfg.scene.terrain.terrain_generator.sub_terrains["single_step"]
        position = self.robot.data.root_pos_w - self._env.scene.env_origins
        yaw = self.robot.data.heading_w
        final_edge, _ = _target_geometry(cfg)
        distance = final_edge + 0.9 - position[:, 0]
        vx_world = self.approach_speed * (distance / 0.4).clamp(0.0, 1.0)
        vy_world = (-0.5 * position[:, 1]).clamp(-0.15, 0.15)
        # Yaw-frame rotation, consistent with body-frame velocity commands.
        self.vel_command_b[:, 0] = torch.cos(yaw) * vx_world + torch.sin(yaw) * vy_world
        self.vel_command_b[:, 1] = -torch.sin(yaw) * vx_world + torch.cos(yaw) * vy_world
        self.vel_command_b[:, 2] = (-1.5 * wrap_to_pi(yaw)).clamp(-0.4, 0.4)


def _state(env):
    if not hasattr(env, "_high_step_state"):
        terrain_cfg = env.cfg.scene.terrain.terrain_generator.sub_terrains["single_step"]
        final_edge, target_heights = _target_geometry(terrain_cfg)
        robot = env.scene["robot"]
        feet, foot_names = robot.find_bodies(".*_foot")
        contact_feet, _ = env.scene["contact_forces"].find_bodies(foot_names, preserve_order=True)
        if len(feet) != 4 or len(contact_feet) != 4:
            raise ValueError("High-step success requires exactly four foot bodies/contact sensors")
        zeros = lambda: torch.zeros(env.num_envs, device=env.device)
        env._high_step_state = {
            "heights": torch.tensor(target_heights, device=env.device),
            "feet": feet,
            "contact_feet": contact_feet,
            "edge": final_edge,
            "age": zeros(),
            "hold": zeros(),
            "success": zeros().bool(),
            "last_success": zeros().bool(),
            "previous_x": robot.data.root_pos_w[:, 0].clone(),
            "progress": zeros(),
            "landings": zeros(),
            "ever_landed": torch.zeros(env.num_envs, 4, device=env.device, dtype=torch.bool),
            "win_streak": zeros().long(),
            "loss_streak": zeros().long(),
            "attempts": torch.zeros(len(terrain_cfg.height_levels), device=env.device),
            "wins": torch.zeros(len(terrain_cfg.height_levels), device=env.device),
            "last_step": -1,
        }
    return env._high_step_state


def reset_high_step_state(env, env_ids):
    """Run after the physical reset; keep completed outcomes for the evaluator."""
    state = _state(env)
    for name in ("age", "hold", "success", "progress", "landings", "ever_landed"):
        state[name][env_ids] = 0
    state["previous_x"][env_ids] = env.scene["robot"].data.root_pos_w[env_ids, 0]


def high_step_success(env, hold_time=0.6):
    """Four feet beyond the edge, supported, upright, and stable for hold_time."""
    state = _state(env)
    if state["last_step"] == env.common_step_counter:
        return state["success"]
    state["last_step"] = env.common_step_counter
    robot = env.scene["robot"]
    origins = env.scene.env_origins
    position = robot.data.root_pos_w - origins
    feet = robot.data.body_pos_w[:, state["feet"], :] - origins[:, None, :]
    height = state["heights"][env.scene.terrain.terrain_levels]
    forces = env.scene["contact_forces"].data.net_forces_w[:, state["contact_feet"], :]
    contact = torch.linalg.vector_norm(forces, dim=-1) > 2.0
    on_top = (feet[:, :, 0] > state["edge"] + 0.10) & (
        feet[:, :, 2] > height[:, None] - 0.025
    )
    landed = on_top & contact
    state["landings"] = (landed & ~state["ever_landed"]).sum(dim=1).float()
    state["ever_landed"] |= landed
    x = torch.minimum(robot.data.root_pos_w[:, 0], origins[:, 0] + state["edge"] + 0.9)
    # Ignore reset jumps. Reverse travel has a matching negative reward.
    state["progress"] = torch.where(
        state["age"] > 0, (x - state["previous_x"]).clamp(-0.04, 0.04), 0.0
    )
    state["previous_x"] = x.clone()
    state["age"] += 1
    stable = (
        on_top.all(dim=1)
        & (contact.sum(dim=1) >= 2)
        & (position[:, 0] > state["edge"] + 0.65)
        & (position[:, 1].abs() < 0.65)
        & (position[:, 2] > height + 0.19)
        & (robot.data.projected_gravity_b[:, 2] < -0.75)
        & (torch.linalg.vector_norm(robot.data.root_lin_vel_w[:, :2], dim=-1) < 0.35)
        & (torch.linalg.vector_norm(robot.data.root_ang_vel_b, dim=-1) < 1.0)
    )
    state["hold"] = torch.where(stable, state["hold"] + env.step_dt, 0.0)
    state["success"] = state["hold"] >= hold_time
    return state["success"]


def high_step_progress(env):
    return _state(env)["progress"] / env.step_dt


def high_step_foot_landing(env):
    # A one-off reward per foot per episode, not a reward for hanging at the edge.
    return _state(env)["landings"] / env.step_dt


def high_step_completion(env):
    # RewardManager already multiplies by dt: cancelling here creates a fixed
    # terminal-event bonus (not another per-second reward).
    return _state(env)["success"].float() / env.step_dt


def high_step_failure(env):
    return (env.reset_terminated & ~_state(env)["success"]).float() / env.step_dt


def high_step_roll(env):
    # Allow pitching forward/back to climb, while discouraging lateral collapse.
    return env.scene["robot"].data.projected_gravity_b[:, 1].square()


def high_step_out_of_lane(env):
    position = env.scene["robot"].data.root_pos_w - env.scene.env_origins
    return (position[:, 1].abs() > 1.0) | (position[:, 0] < -0.8) | (
        env.scene["robot"].data.projected_gravity_b[:, 2] > -0.15
    )


def high_step_curriculum(env, env_ids, successes_to_promote=3, failures_to_demote=3, promote=True):
    """Promote by completed climbs, never by distance walked or jumping past a wall."""
    state = _state(env)
    terrain = env.scene.terrain
    state["last_success"][env_ids] = state["success"][env_ids]
    # Initial reset is not an attempt; very short randomized initial timeouts
    # are not allowed to affect the training curriculum.
    valid = env_ids[state["age"][env_ids] >= 25]
    level = terrain.terrain_levels[valid].clone()
    success = state["success"][valid]
    state["attempts"] += torch.bincount(level, minlength=len(state["heights"]))
    state["wins"] += torch.bincount(level[success], minlength=len(state["heights"]))
    state["win_streak"][valid] = torch.where(success, state["win_streak"][valid] + 1, 0)
    state["loss_streak"][valid] = torch.where(success, 0, state["loss_streak"][valid] + 1)
    if promote:
        up = state["win_streak"][valid] >= successes_to_promote
        down = state["loss_streak"][valid] >= failures_to_demote
        # Do NOT use TerrainImporter.update_env_origins: its last-level success
        # wraps back to a random level. Keep training at the 28 cm target instead.
        terrain.terrain_levels[valid] = (level + up.long() - down.long()).clamp(
            0, len(state["heights"]) - 1
        )
        terrain.env_origins[valid] = terrain.terrain_origins[
            terrain.terrain_levels[valid], terrain.terrain_types[valid]
        ]
        state["win_streak"][valid[up | down]] = 0
        state["loss_streak"][valid[up | down]] = 0
    heights = state["heights"][terrain.terrain_levels]
    metrics = {
        "mean_height_cm": heights.mean() * 100,
        "max_height_cm": heights.max() * 100,
        "fraction_at_28cm": (heights >= 0.279).float().mean(),
    }
    _, target_heights = _target_geometry(env.cfg.scene.terrain.terrain_generator.sub_terrains["single_step"])
    for index, height in enumerate(target_heights):
        label = f"{height * 100:02.0f}cm"
        metrics[f"attempts_{label}"] = state["attempts"][index].clone()
        metrics[f"success_cumulative_{label}"] = (
            state["wins"][index] / state["attempts"][index].clamp(min=1)
        )
    return metrics
