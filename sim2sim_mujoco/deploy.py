"""Run an Isaac Lab Go2 TorchScript policy in the official Unitree MuJoCo model."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import mujoco
import numpy as np
import torch

from keyboard_control import KeyboardController


ROOT = Path(__file__).resolve().parent
DEFAULT_MODEL = ROOT / "unitree_mujoco" / "unitree_robots" / "go2" / "scene_policy_flat.xml"
STAIRS_MODEL = DEFAULT_MODEL.with_name("scene_policy_stairs.xml")
PLAYGROUND_MODEL = DEFAULT_MODEL.with_name("scene_policy_playground.xml")
DEFAULT_POLICY = (
    ROOT.parent
    / "logs"
    / "rsl_rl"
    / "go2_velocity_stairs_height_transfer"
    / "2026-10-01_10-58-57_stairs_lateral_speed"
    / "exported"
    / "policy.pt"
)

ISAAC_JOINT_ORDER = (
    "FL_hip_joint",
    "FR_hip_joint",
    "RL_hip_joint",
    "RR_hip_joint",
    "FL_thigh_joint",
    "FR_thigh_joint",
    "RL_thigh_joint",
    "RR_thigh_joint",
    "FL_calf_joint",
    "FR_calf_joint",
    "RL_calf_joint",
    "RR_calf_joint",
)

DEFAULT_JOINT_POS = np.array(
    [0.1, -0.1, 0.1, -0.1, 0.8, 0.8, 1.0, 1.0, -1.5, -1.5, -1.5, -1.5], dtype=np.float64
)


class Go2Sim2Sim:
    """MuJoCo state adapter for the Isaac Lab Go2 policy."""

    def __init__(
        self,
        model_path: Path,
        policy_path: Path,
        command: np.ndarray,
        kp: float = 25.0,
        kd: float = 0.5,
        action_scale: float = 0.25,
        torque_limit: float = 23.5,
        constant_height: float | None = None,
        step_height: float | None = None,
        joint_dynamics: str = "isaac",
    ):
        if not model_path.is_file():
            raise FileNotFoundError(f"MuJoCo scene not found: {model_path}")
        if not policy_path.is_file():
            raise FileNotFoundError(f"TorchScript policy not found: {policy_path}")

        self.model = mujoco.MjModel.from_xml_path(str(model_path))
        if step_height is not None:
            if not 0.01 <= step_height <= 0.15:
                raise ValueError("--step-height must be between 0.01 and 0.15 meters.")
            for name in [f"up_{i}" for i in range(1, 7)] + ["platform"] + [f"down_{i}" for i in range(1, 6)]:
                geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, name)
                if geom_id < 0:
                    raise ValueError("--step-height requires the provided stairs or playground scene.")
                self.model.geom_pos[geom_id, 2] *= step_height / 0.05
                self.model.geom_size[geom_id, 2] *= step_height / 0.05
        self.data = mujoco.MjData(self.model)
        self.model.opt.timestep = 0.005
        self.command = np.asarray(command, dtype=np.float32)
        self.kp = kp
        self.kd = kd
        self.action_scale = action_scale
        self.torque_limit = torque_limit
        self.constant_height = constant_height
        self.policy = torch.jit.load(str(policy_path), map_location="cpu").eval()
        actor_weights = [parameter for name, parameter in self.policy.named_parameters() if name == "actor.0.weight"]
        if len(actor_weights) != 1:
            raise RuntimeError("Could not infer policy input width from actor.0.weight.")
        self.policy_obs_dim = int(actor_weights[0].shape[1])
        if self.policy_obs_dim not in (48, 235):
            raise RuntimeError(f"Unsupported policy observation width: {self.policy_obs_dim}")
        self.last_action = np.zeros(12, dtype=np.float32)
        self.action = np.zeros(12, dtype=np.float32)

        self.base_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "base_link")
        if self.base_body_id < 0:
            raise RuntimeError("Body base_link not found in MuJoCo scene.")
        self.joint_qpos_adr = np.array([self._joint_address(name, qpos=True) for name in ISAAC_JOINT_ORDER])
        self.joint_dof_adr = np.array([self._joint_address(name, qpos=False) for name in ISAAC_JOINT_ORDER])
        self.actuator_ids = np.array(
            [
                mujoco.mj_name2id(
                    self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, joint_name.removesuffix("_joint")
                )
                for joint_name in ISAAC_JOINT_ORDER
            ]
        )
        if np.any(self.actuator_ids < 0):
            raise RuntimeError("Failed to resolve one or more MuJoCo actuators.")

        # The trained USD uses explicit DCMotor PD, zero joint friction and no
        # added armature. MuJoCo's stock passive damping/friction would double
        # count damping and change the learned actuator dynamics.
        if joint_dynamics == "isaac":
            self.model.dof_damping[self.joint_dof_adr] = 0.0
            self.model.dof_frictionloss[self.joint_dof_adr] = 0.0
            self.model.dof_armature[self.joint_dof_adr] = 0.0
        elif joint_dynamics != "official":
            raise ValueError(f"Unknown joint dynamics profile: {joint_dynamics}")
        self.joint_dynamics = joint_dynamics

        # Isaac disabled robot self-collisions. Keep robot/terrain contacts and
        # visual meshes intact, using separate collision masks for each side.
        robot_geoms = self.model.body_rootid[self.model.geom_bodyid] == self.base_body_id
        colliding = (self.model.geom_contype != 0) | (self.model.geom_conaffinity != 0)
        self.model.geom_contype[robot_geoms & colliding] = 2
        self.model.geom_conaffinity[robot_geoms & colliding] = 1
        self.model.geom_contype[~robot_geoms & colliding] = 1
        self.model.geom_conaffinity[~robot_geoms & colliding] = 2
        self.model.geom_friction[:, 0] = 0.6

        xs = np.linspace(-0.8, 0.8, 17)
        ys = np.linspace(-0.5, 0.5, 11)
        self.scan_points_b = np.array([[x, y, 0.0] for y in ys for x in xs], dtype=np.float64)
        self.ray_geom_group = np.array([1, 1, 0, 0, 0, 0], dtype=np.uint8)

        self.reset()

    def _joint_address(self, name: str, qpos: bool) -> int:
        joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if joint_id < 0:
            raise RuntimeError(f"Joint not found in MuJoCo model: {name}")
        return int(self.model.jnt_qposadr[joint_id] if qpos else self.model.jnt_dofadr[joint_id])

    def reset(self) -> None:
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[:3] = (0.0, 0.0, 0.40)
        self.data.qpos[3:7] = (1.0, 0.0, 0.0, 0.0)
        self.data.qpos[self.joint_qpos_adr] = DEFAULT_JOINT_POS
        self.data.qvel[:] = 0.0
        self.data.ctrl[:] = 0.0
        self.last_action[:] = 0.0
        self.action[:] = 0.0
        mujoco.mj_forward(self.model, self.data)

    def body_rotation(self) -> np.ndarray:
        matrix = np.empty(9, dtype=np.float64)
        mujoco.mju_quat2Mat(matrix, self.data.xquat[self.base_body_id])
        return matrix.reshape(3, 3)

    def body_velocity(self) -> tuple[np.ndarray, np.ndarray]:
        velocity = np.zeros(6, dtype=np.float64)
        mujoco.mj_objectVelocity(
            self.model, self.data, mujoco.mjtObj.mjOBJ_BODY, self.base_body_id, velocity, 0
        )
        # mjOBJ_BODY with flg_local=1 uses the body's inertial/principal axes,
        # which differ from Go2's link frame. Isaac uses COM velocities rotated
        # into the root LINK frame. Read world-oriented COM velocity first.
        world_to_link = self.body_rotation().T
        return world_to_link @ velocity[3:], world_to_link @ velocity[:3]

    def height_scan(self) -> np.ndarray:
        if self.constant_height is not None:
            return np.full(187, self.constant_height, dtype=np.float32)
        rotation = self.body_rotation()
        yaw = np.arctan2(rotation[1, 0], rotation[0, 0])
        cos_yaw, sin_yaw = np.cos(yaw), np.sin(yaw)
        yaw_rotation = np.array(
            [[cos_yaw, -sin_yaw, 0.0], [sin_yaw, cos_yaw, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64
        )
        base_pos = self.data.xpos[self.base_body_id]
        heights = np.empty(187, dtype=np.float32)
        direction = np.array([0.0, 0.0, -1.0], dtype=np.float64)
        geom_id = np.array([-1], dtype=np.int32)

        for index, point_b in enumerate(self.scan_points_b):
            point_w = base_pos + yaw_rotation @ point_b
            origin = point_w.copy()
            origin[2] = base_pos[2] + 2.0
            distance = mujoco.mj_ray(
                self.model,
                self.data,
                origin,
                direction,
                self.ray_geom_group,
                True,
                -1,
                geom_id,
            )
            if distance < 0.0:
                raise RuntimeError(f"Height ray {index} missed the terrain; check the scene's ground geoms.")
            else:
                hit_z = origin[2] - distance
                heights[index] = np.clip(base_pos[2] - hit_z - 0.5, -1.0, 1.0)
        return heights

    def observation(self) -> np.ndarray:
        lin_vel_b, ang_vel_b = self.body_velocity()
        projected_gravity = self.body_rotation().T @ np.array([0.0, 0.0, -1.0])
        joint_pos = self.data.qpos[self.joint_qpos_adr]
        joint_vel = self.data.qvel[self.joint_dof_adr]
        observation = np.concatenate(
            (
                lin_vel_b,
                ang_vel_b,
                projected_gravity,
                self.command,
                joint_pos - DEFAULT_JOINT_POS,
                joint_vel,
                self.last_action,
                self.height_scan(),
            )
        ).astype(np.float32)
        if observation.shape != (235,):
            raise RuntimeError(f"Unexpected observation shape: {observation.shape}")
        return observation

    def infer(self) -> None:
        # mj_step integrates qpos/qvel after computing Cartesian quantities.
        # Refresh them so joint and base observations refer to the same instant.
        mujoco.mj_forward(self.model, self.data)
        obs = torch.from_numpy(self.observation()[: self.policy_obs_dim]).unsqueeze(0)
        with torch.inference_mode():
            action = self.policy(obs).squeeze(0).cpu().numpy()
        if action.shape != (12,):
            raise RuntimeError(f"Policy must return 12 actions, got {action.shape}.")
        if not np.isfinite(action).all():
            raise FloatingPointError("Policy returned a non-finite action.")
        # Isaac's last_action observation is the action just executed during
        # the preceding policy interval, not the action from two intervals ago.
        # Keep the raw action: the training task has no action clipping.
        self.action[:] = action
        self.last_action[:] = action

    def apply_pd(self, use_policy: bool) -> float:
        if use_policy:
            target = DEFAULT_JOINT_POS + self.action_scale * self.action
        else:
            target = DEFAULT_JOINT_POS
        joint_pos = self.data.qpos[self.joint_qpos_adr]
        joint_vel = self.data.qvel[self.joint_dof_adr]
        torque = self.kp * (target - joint_pos) - self.kd * joint_vel
        # Match Isaac Lab's DCMotor._clip_effort (stall torque=23.5 Nm,
        # no-load speed=30 rad/s), including reduced torque at high speed.
        stall_torque = self.torque_limit
        speed = np.clip(joint_vel, -60.0, 60.0)
        max_effort = np.minimum(stall_torque * (1.0 - speed / 30.0), self.torque_limit)
        min_effort = np.maximum(stall_torque * (-1.0 - speed / 30.0), -self.torque_limit)
        torque = np.clip(torque, min_effort, max_effort)
        self.data.ctrl[self.actuator_ids] = torque
        return float(np.max(np.abs(torque)))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", choices=("flat", "stairs", "playground"), default="flat")
    parser.add_argument("--model", type=Path, default=None, help="Override the scene with a custom MJCF.")
    parser.add_argument("--step-height", type=float, default=None, help="Riser height in stairs/playground (default 0.05 m).")
    parser.add_argument("--policy", type=Path, default=DEFAULT_POLICY)
    parser.add_argument("--vx", type=float, default=0.5)
    parser.add_argument("--vy", type=float, default=0.0)
    parser.add_argument("--wz", type=float, default=0.0)
    parser.add_argument("--warmup", type=float, default=0.0, help="Seconds to hold the default pose before inference.")
    parser.add_argument("--duration", type=float, default=0.0, help="Seconds to run; zero runs until the viewer closes.")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--keyboard", action="store_true", help="Arrow keys move, QE turn, Space stop, R reset, M map.")
    parser.add_argument("--move-speed", type=float, default=0.5, help="Up/Down command magnitude in m/s.")
    parser.add_argument("--strafe-speed", type=float, default=0.15, help="Left/Right command magnitude in m/s.")
    parser.add_argument("--turn-speed", type=float, default=0.2, help="Q/E command magnitude in rad/s.")
    parser.add_argument("--kp", type=float, default=25.0)
    parser.add_argument("--kd", type=float, default=0.5)
    parser.add_argument("--action-scale", type=float, default=0.25)
    parser.add_argument("--torque-limit", type=float, default=23.5)
    parser.add_argument("--joint-dynamics", choices=("isaac", "official"), default="isaac")
    parser.add_argument(
        "--constant-height",
        type=float,
        default=None,
        help="Replace all height rays with a constant value for observation-parity diagnostics.",
    )
    parser.add_argument("--report", type=Path, help="Save rollout metrics as JSON.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.keyboard and args.headless:
        raise ValueError("--keyboard needs a viewer; remove --headless.")
    torch.set_num_threads(1)
    keyboard = KeyboardController(args.move_speed, args.strafe_speed, args.turn_speed) if args.keyboard else None
    command = np.zeros(3, dtype=np.float32) if keyboard else np.array([args.vx, args.vy, args.wz], dtype=np.float32)
    scenes = {"flat": DEFAULT_MODEL, "stairs": STAIRS_MODEL, "playground": PLAYGROUND_MODEL}
    model_path = args.model or scenes[args.scene]
    sim = Go2Sim2Sim(
        model_path.resolve(),
        args.policy.resolve(),
        command,
        args.kp,
        args.kd,
        args.action_scale,
        args.torque_limit,
        args.constant_height,
        args.step_height,
        args.joint_dynamics,
    )
    policy_decimation = 4
    start_wall = time.perf_counter()
    next_report = 0.0
    max_steps = int(args.duration / sim.model.opt.timestep) if args.duration > 0.0 else None
    if args.headless and max_steps is None:
        raise ValueError("Headless evaluation requires a positive --duration.")
    base_positions = []
    base_upright = []
    linear_velocities = []
    angular_velocities = []
    fall_time = None
    reset_count = 0
    map_camera = False

    viewer_context = None
    if not args.headless:
        from mujoco import viewer as mj_viewer

        viewer_context = mj_viewer.launch_passive(
            sim.model, sim.data, key_callback=keyboard.on_key if keyboard else None
        )
        with viewer_context.lock():
            viewer_context.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
            viewer_context.cam.trackbodyid = sim.base_body_id
            viewer_context.cam.distance = 2.5
            viewer_context.cam.elevation = -20.0
            viewer_context.cam.azimuth = 135.0

    if keyboard:
        print(f"Keyboard mode: {keyboard.mode}. Focus MuJoCo viewer. Arrows move; QE turn; Space stop; R reset; M map.")

    try:
        step = 0
        while (max_steps is None or step < max_steps) and (
            viewer_context is None or viewer_context.is_running()
        ):
            sim_time = step * sim.model.opt.timestep
            if keyboard:
                command, request_reset, request_camera = keyboard.poll(sim.model.opt.timestep)
                if request_reset:
                    with viewer_context.lock():
                        sim.reset()
                    reset_count += 1
                    print("Robot reset to the start pad.")
                sim.command[:] = command
                if request_camera:
                    map_camera = not map_camera
                    with viewer_context.lock():
                        if map_camera:
                            viewer_context.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
                            viewer_context.cam.lookat[:] = (6.0, 0.0, 0.25)
                            viewer_context.cam.distance = 24.0
                            viewer_context.cam.elevation = -65.0
                            viewer_context.cam.azimuth = 90.0
                        else:
                            viewer_context.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
                            viewer_context.cam.trackbodyid = sim.base_body_id
                            viewer_context.cam.distance = 2.5
                            viewer_context.cam.elevation = -20.0
                            viewer_context.cam.azimuth = 135.0
            use_policy = sim_time >= args.warmup
            if use_policy and step % policy_decimation == 0:
                sim.infer()
            max_torque = sim.apply_pd(use_policy)
            mujoco.mj_step(sim.model, sim.data)

            if step % policy_decimation == 0:
                mujoco.mj_forward(sim.model, sim.data)
                position = sim.data.xpos[sim.base_body_id].copy()
                upright = float(sim.body_rotation()[2, 2])
                if (position[2] < 0.15 or upright < 0.5) and fall_time is None:
                    fall_time = sim_time
                base_positions.append(position)
                base_upright.append(upright)
                linear_velocity, angular_velocity = sim.body_velocity()
                linear_velocities.append(linear_velocity)
                angular_velocities.append(angular_velocity)

            if viewer_context is not None:
                if keyboard and step % policy_decimation == 0 and hasattr(viewer_context, "set_texts"):
                    info = keyboard.overlay()
                    if model_path.resolve() == PLAYGROUND_MODEL.resolve():
                        info += (
                            "\n+X routes: y=0 stairs/bumps; y=+4.5 ramp/rubble; y=-4.5 narrow stairs"
                        )
                    viewer_context.set_texts((None, None, "Go2 controls", info))
                viewer_context.sync()
                # Keep pacing monotonic even when R resets MuJoCo data.time.
                target_wall = start_wall + (step + 1) * sim.model.opt.timestep
                sleep_time = target_wall - time.perf_counter()
                if sleep_time > 0.0:
                    time.sleep(sleep_time)

            if sim_time >= next_report:
                print(
                    f"t={sim_time:6.2f}s "
                    f"base_xyz={sim.data.xpos[sim.base_body_id].round(3).tolist()} "
                    f"cmd={sim.command.tolist()} action_abs={np.mean(np.abs(sim.action)):.3f} "
                    f"max_tau={max_torque:.2f}"
                )
                next_report += 1.0
            step += 1
    finally:
        if viewer_context is not None:
            viewer_context.close()

    if base_positions:
        positions = np.asarray(base_positions)
        velocities = np.asarray(linear_velocities)
        is_stair_course = model_path.resolve() in (STAIRS_MODEL.resolve(), PLAYGROUND_MODEL.resolve())
        riser_height = args.step_height if args.step_height is not None else 0.05
        on_platform = (positions[:, 0] >= 3.3) & (positions[:, 0] <= 5.1) & (np.abs(positions[:, 1]) < 2.0)
        stairs_cleared = (
            fall_time is None
            and positions[-1, 0] > 7.0
            and np.any(on_platform & (positions[:, 2] > 6.0 * riser_height + 0.2))
        ) if is_stair_course else None
        summary = {
            "policy": str(args.policy.resolve()),
            "scene": str(model_path.resolve()),
            "step_height_m": riser_height if is_stair_course else None,
            "stairs_course_cleared": bool(stairs_cleared) if is_stair_course else None,
            "observation_dim": sim.policy_obs_dim,
            "duration_s": float(step * sim.model.opt.timestep),
            "control_mode": keyboard.mode if keyboard else "fixed",
            "reset_count": reset_count,
            "command": sim.command.tolist(),
            "kp": sim.kp,
            "kd": sim.kd,
            "action_scale": sim.action_scale,
            "joint_dynamics": sim.joint_dynamics,
            "first_fall_s": fall_time,
            "final_position": positions[-1].tolist(),
            "distance_xy_m": float(np.linalg.norm(positions[-1, :2] - positions[0, :2])),
            "min_base_z_m": float(positions[:, 2].min()),
            "max_base_z_m": float(positions[:, 2].max()),
            "min_upright_cos": float(min(base_upright)),
            "mean_body_velocity": velocities[len(velocities) // 10 :].mean(axis=0).tolist(),
            "mean_body_angular_velocity": np.asarray(angular_velocities)[len(velocities) // 10 :].mean(axis=0).tolist(),
        }
        print("RESULT " + json.dumps(summary))
        if args.report is not None:
            args.report.parent.mkdir(parents=True, exist_ok=True)
            args.report.write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
