"""Stage 2: climb first, then stop and stand with four loaded feet.

No observation/action dimensions change. Reference stance is the Go2 default
joint pose, with analytical FK for its 0.213 m thigh/calf links and hip offsets.
Pose rewards are gated by the post-climb settling phase, not applied to climbing.
"""

import math

import torch

from isaaclab.envs.mdp.events import reset_root_state_uniform
from isaaclab.utils.math import quat_apply_inverse

from .high_step import (
    HighStepVelocityCommand,
    _state,
    _target_geometry,
    high_step_curriculum,
    high_step_success,
    reset_high_step_state,
)


def nominal_go2_feet(default_joint_pos, joint_names, foot_names):
    """Body-frame foot-link positions at the configured nominal joint angles.

    Geometry agrees with the local Unitree Go2 model: hips at (+/-0.1934,
    +/-0.0465, 0), thigh lateral offset +/-0.0955, two links of 0.213 m.
    Foot-link origin is the centre of the 0.022 m foot sphere, not its sole.
    """
    indices = {name: index for index, name in enumerate(joint_names)}
    feet = []
    for name in foot_names:
        prefix = name.split("_")[0]
        if prefix not in ("FL", "FR", "RL", "RR"):
            raise ValueError(f"Unexpected Go2 foot name: {name}")
        side = 1.0 if prefix.endswith("L") else -1.0
        front = 1.0 if prefix.startswith("F") else -1.0
        hip = default_joint_pos[indices[f"{prefix}_hip_joint"]]
        thigh = default_joint_pos[indices[f"{prefix}_thigh_joint"]]
        calf = default_joint_pos[indices[f"{prefix}_calf_joint"]]
        x = front * 0.1934 - 0.213 * (torch.sin(thigh) + torch.sin(thigh + calf))
        leg_z = -0.213 * (torch.cos(thigh) + torch.cos(thigh + calf))
        y = side * 0.0465 + torch.cos(hip) * (side * 0.0955) - torch.sin(hip) * leg_z
        z = torch.sin(hip) * (side * 0.0955) + torch.cos(hip) * leg_z
        feet.append(torch.stack((x, y, z)))
    return torch.stack(feet)


def _stand_state(env):
    if not hasattr(env, "_high_step_stand_state"):
        robot = env.scene["robot"]
        _, names = robot.find_bodies(".*_foot")
        nominal = nominal_go2_feet(robot.data.default_joint_pos[0], robot.joint_names, names)
        hip_ids = [i for i, name in enumerate(robot.joint_names) if "_hip_joint" in name]
        leg_ids = [i for i, name in enumerate(robot.joint_names) if "_hip_joint" not in name]
        zeros = lambda: torch.zeros(env.num_envs, device=env.device)
        env._high_step_stand_state = {
            "nominal_feet": nominal,
            "nominal_height": -nominal[:, 2].mean() + 0.022,
            "hip_ids": hip_ids,
            "leg_ids": leg_ids,
            "settling": zeros().bool(),
            "climbed": zeros().bool(),
            "hold": zeros(),
            "last_climbed": zeros().bool(),
            "last_contact_count": zeros(),
            "last_hip_error": zeros(),
            "last_leg_error": zeros(),
            "last_foot_error": zeros(),
            "last_tilt_deg": zeros(),
            "last_speed": zeros(),
            "last_max_hold": zeros(),
            "max_hold": zeros(),
            "totals": torch.zeros(5, device=env.device),
            "features": {},
            "last_step": -1,
        }
    return env._high_step_stand_state


class HighStepStandVelocityCommand(HighStepVelocityCommand):
    def _update_command(self):
        super()._update_command()
        if hasattr(self._env, "_high_step_stand_state"):
            # A true zero command: no creep, yaw correction, or lateral motion
            # while practising the final static stance.
            self.vel_command_b[self._env._high_step_stand_state["settling"]] = 0.0


def reset_high_step_stand_base(env, env_ids, pose_range, velocity_range, platform_probability=0.25):
    """Optional standing-only practice at reset; NEVER used by the PLAY task."""
    reset_root_state_uniform(env, env_ids, pose_range=pose_range, velocity_range=velocity_range)
    if not hasattr(env, "_stand_platform_start"):
        env._stand_platform_start = torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)
    chosen = torch.rand(len(env_ids), device=env.device) < platform_probability
    env._stand_platform_start[env_ids] = chosen
    ids = env_ids[chosen]
    if len(ids):
        cfg = env.cfg.scene.terrain.terrain_generator.sub_terrains["single_step"]
        final_edge, target_heights = _target_geometry(cfg)
        robot = env.scene["robot"]
        root = robot.data.root_state_w[ids].clone()
        root[:, 0] = env.scene.env_origins[ids, 0] + final_edge + 0.9
        heights = torch.tensor(target_heights, device=env.device)[env.scene.terrain.terrain_levels[ids]]
        root[:, 2] = env.scene.env_origins[ids, 2] + heights + robot.data.default_root_state[ids, 2]
        robot.write_root_pose_to_sim(root[:, :7], env_ids=ids)


def reset_high_step_stand_state(env, env_ids):
    reset_high_step_state(env, env_ids)
    stand = _stand_state(env)
    for name in ("settling", "climbed", "hold", "max_hold"):
        stand[name][env_ids] = 0
    # Current-step reward features are overwritten by the termination pass
    # before any reward computation; completed snapshots survive this reset.


def high_step_stand_success(
    env, hold_time=2.0, hip_tolerance=0.12, leg_tolerance=0.35,
    foot_tolerance=0.05, height_tolerance=0.05, linear_speed_max=0.08,
    angular_speed_max=0.20, tilt_max_deg=10.0, min_support_force=5.0,
):
    stand = _stand_state(env)
    state = _state(env)
    if stand["last_step"] == env.common_step_counter:
        return state["success"]
    stand["last_step"] = env.common_step_counter
    # Maintain the original progress/landing accounting, but do not terminate
    # an episode when its old, permissive climb criterion first becomes true.
    stand["climbed"] |= high_step_success(env).clone()
    robot = env.scene["robot"]
    data = robot.data
    origins = env.scene.env_origins
    position = data.root_pos_w - origins
    feet_w = data.body_pos_w[:, state["feet"], :]
    feet = feet_w - origins[:, None, :]
    height = state["heights"][env.scene.terrain.terrain_levels]
    beyond_edge = feet[:, :, 0] > state["edge"] + 0.10
    stage = (position[:, 0] > state["edge"] + 0.70) & (position[:, 2] > height + 0.20)
    stand["settling"] |= stage
    # Only upward normal force counts as support, not a foot pushing a wall.
    normal_force = env.scene["contact_forces"].data.net_forces_w[:, state["contact_feet"], 2]
    foot_height_ok = (feet[:, :, 2] > height[:, None] - 0.025) & (
        feet[:, :, 2] < height[:, None] + 0.06
    )
    support = (normal_force > min_support_force) & foot_height_ok & beyond_edge
    feet_b = quat_apply_inverse(
        data.root_quat_w[:, None, :].expand(-1, 4, -1), feet_w - data.root_pos_w[:, None, :]
    )
    foot_error = torch.linalg.vector_norm(feet_b[:, :, :2] - stand["nominal_feet"][None, :, :2], dim=-1)
    joint_error = data.joint_pos - data.default_joint_pos
    hip_error = joint_error[:, stand["hip_ids"]]
    leg_error = joint_error[:, stand["leg_ids"]]
    tilt = torch.linalg.vector_norm(data.projected_gravity_b[:, :2], dim=-1)
    height_error = position[:, 2] - height - stand["nominal_height"]
    speed = torch.linalg.vector_norm(data.root_lin_vel_w, dim=-1)
    angular_speed = torch.linalg.vector_norm(data.root_ang_vel_b, dim=-1)
    command_stopped = torch.linalg.vector_norm(env.command_manager.get_command("base_velocity"), dim=-1) < 0.01
    stable = (
        stand["settling"] & support.all(dim=1) & command_stopped
        & (position[:, 1].abs() < 0.65)
        & (speed < linear_speed_max) & (angular_speed < angular_speed_max)
        & (data.projected_gravity_b[:, 2] < 0)
        & (tilt < math.sin(math.radians(tilt_max_deg)))
        & (height_error.abs() < height_tolerance)
        & (hip_error.abs().amax(dim=1) < hip_tolerance)
        & (leg_error.abs().amax(dim=1) < leg_tolerance)
        & (foot_error.amax(dim=1) < foot_tolerance)
    )
    stand["hold"] = torch.where(stable, stand["hold"] + env.step_dt, 0.0)
    stand["max_hold"] = torch.maximum(stand["max_hold"], stand["hold"])
    state["success"] = stand["hold"] >= hold_time
    stand["features"] = {
        "gate": stand["settling"].float(), "support": support.float(),
        "hip_error": hip_error, "leg_error": leg_error, "foot_error": foot_error,
        "height_error": height_error, "tilt": tilt, "speed": speed,
        "angular_speed": angular_speed,
        "tilt_deg": torch.rad2deg(torch.acos((-data.projected_gravity_b[:, 2]).clamp(-1, 1))),
    }
    return state["success"]


def stand_tracking_scale(env, unsupported_scale=0.1):
    """Keep climbing rewards intact; reduce post-climb tracking without four supports."""
    if not 0.0 <= unsupported_scale <= 1.0:
        raise ValueError("unsupported_scale must be between 0 and 1")
    f = _stand_state(env)["features"]
    incomplete_stance = (f["gate"] > 0) & ~f["support"].bool().all(dim=1)
    return torch.where(incomplete_stance, unsupported_scale, torch.ones_like(f["gate"]))


def stand_track_lin_vel_xy_exp(env, std, command_name, unsupported_scale=0.1):
    """Original body-frame XY tracking, scaled only during incomplete settling."""
    error = (
        env.command_manager.get_command(command_name)[:, :2]
        - env.scene["robot"].data.root_lin_vel_b[:, :2]
    ).square().sum(dim=1)
    return torch.exp(-error / std**2) * stand_tracking_scale(env, unsupported_scale)


def stand_track_ang_vel_z_exp(env, std, command_name, unsupported_scale=0.1):
    """Original body-frame yaw tracking, scaled only during incomplete settling."""
    error = (
        env.command_manager.get_command(command_name)[:, 2]
        - env.scene["robot"].data.root_ang_vel_b[:, 2]
    ).square()
    return torch.exp(-error / std**2) * stand_tracking_scale(env, unsupported_scale)


def stand_support_reward(env):
    f = _stand_state(env)["features"]
    # Three supporting feet must not receive 75% of a four-foot stance reward.
    return f["gate"] * f["support"].bool().all(dim=1).float()


def stand_pose_penalty(env):
    f = _stand_state(env)["features"]
    return f["gate"] * (f["hip_error"].square().mean(dim=1) + 0.25 * f["leg_error"].square().mean(dim=1))


def stand_foot_placement_penalty(env):
    f = _stand_state(env)["features"]
    return f["gate"] * f["foot_error"].square().mean(dim=1)


def stand_orientation_penalty(env):
    f = _stand_state(env)["features"]
    return f["gate"] * f["tilt"].square()


def stand_height_penalty(env):
    f = _stand_state(env)["features"]
    return f["gate"] * f["height_error"].square()


def stand_motion_penalty(env):
    f = _stand_state(env)["features"]
    return f["gate"] * (f["speed"].square() + 0.1 * f["angular_speed"].square())


def stand_quality_reward(env):
    f = _stand_state(env)["features"]
    error = (
        f["hip_error"].square().mean(dim=1) / 0.12**2
        + f["leg_error"].square().mean(dim=1) / 0.35**2
        + f["foot_error"].square().mean(dim=1) / 0.05**2
        + f["tilt"].square() / 0.17**2 + f["speed"].square() / 0.08**2
    )
    return f["gate"] * f["support"].bool().all(dim=1).float() * torch.exp(-error)


def high_step_stand_curriculum(env, env_ids):
    stand = _stand_state(env)
    state = _state(env)
    stand["last_climbed"][env_ids] = stand["climbed"][env_ids]
    stand["last_max_hold"][env_ids] = stand["max_hold"][env_ids]
    f = stand["features"]
    if f:
        for key, value in (
            ("contact_count", f["support"].sum(dim=1)),
            ("hip_error", f["hip_error"].abs().amax(dim=1)),
            ("leg_error", f["leg_error"].abs().amax(dim=1)),
            ("foot_error", f["foot_error"].amax(dim=1)),
            ("tilt_deg", f["tilt_deg"]), ("speed", f["speed"]),
        ):
            stand[f"last_{key}"][env_ids] = value[env_ids]
    valid = env_ids[state["age"][env_ids] >= 25]
    platform = env._stand_platform_start[valid] if hasattr(env, "_stand_platform_start") else torch.zeros(
        len(valid), device=env.device, dtype=torch.bool
    )
    full = valid[~platform]
    practice = valid[platform]
    stand["totals"] += torch.stack((
        torch.tensor(len(full), device=env.device), state["success"][full].sum(),
        stand["climbed"][full].sum(), torch.tensor(len(practice), device=env.device),
        state["success"][practice].sum(),
    ))
    combined = high_step_curriculum(env, env_ids, promote=False)
    # Do not expose mixed reset-type successes as a misleading climbing rate.
    result = {
        key: value for key, value in combined.items()
        if not key.startswith(("attempts_", "success_cumulative_"))
    }
    totals = stand["totals"]
    result.update({
        "from_below_attempts": totals[0].clone(),
        "from_below_stands": totals[1].clone(),
        "platform_practice_attempts": totals[3].clone(),
        "climb_from_below_rate": totals[2] / totals[0].clamp(min=1),
        "strict_stand_from_below_rate": totals[1] / totals[0].clamp(min=1),
        "platform_practice_stand_rate": totals[4] / totals[3].clamp(min=1),
        "settling_fraction": stand["settling"].float().mean(),
        "platform_start_fraction": env._stand_platform_start.float().mean() if hasattr(
            env, "_stand_platform_start"
        ) else torch.tensor(0.0, device=env.device),
    })
    return result
