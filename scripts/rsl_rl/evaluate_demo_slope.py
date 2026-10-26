"""Evaluate and optionally record a fixed ramp without changing/training a policy."""

import argparse
import json
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--direction", choices=("uphill", "downhill"), required=True)
parser.add_argument("--slope_degrees", type=float, default=20.0)
parser.add_argument("--num_envs", type=int, default=16)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--duration", type=float, default=18.0)
parser.add_argument("--output", required=True)
parser.add_argument("--video_folder")
parser.add_argument("--video_name", default="go2_slope")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if not 0 < args.slope_degrees < 30 or args.num_envs < 1 or not 1 < args.duration < 20:
    parser.error("Require 0 < slope < 30 degrees, positive environments, 1 < duration < 20 seconds")
if args.video_folder and args.num_envs != 1:
    parser.error("Record one robot; run multi-environment evaluations separately")
if args.video_folder:
    args.enable_cameras = True
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import gymnasium as gym
import torch
from rsl_rl.runners import OnPolicyRunner

import go2_demo.tasks  # noqa: F401
from isaaclab_tasks.utils import load_cfg_from_registry
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper


def main():
    task = "Go2-Demo-" + ("Uphill" if args.direction == "uphill" else "Downhill") + "-Play-v0"
    cfg = load_cfg_from_registry(task, "env_cfg_entry_point")
    agent = load_cfg_from_registry(task, "rsl_rl_cfg_entry_point")
    cfg.seed = agent.seed = args.seed
    cfg.scene.num_envs = args.num_envs
    cfg.sim.device = args.device
    terrain = cfg.scene.terrain.terrain_generator.sub_terrains["demo"]
    terrain.slope_degrees = args.slope_degrees
    # Stay within the middle half of the 4 m ramp: +/-1 m leaves a full
    # metre of edge margin. Lateral drift is not equivalent to falling.
    lateral_limit = cfg.scene.terrain.terrain_generator.size[1] / 4
    torch.manual_seed(args.seed)
    environment = gym.make(task, cfg=cfg, render_mode="rgb_array" if args.video_folder else None)
    dt = environment.unwrapped.step_dt
    steps = round(args.duration / dt)
    if args.video_folder:
        environment = gym.wrappers.RecordVideo(
            environment, video_folder=str(Path(args.video_folder).resolve()),
            name_prefix=args.video_name, step_trigger=lambda step: step == 0,
            video_length=steps, disable_logger=True,
        )
    env = RslRlVecEnvWrapper(environment, clip_actions=agent.clip_actions)
    try:
        runner = OnPolicyRunner(env, agent.to_dict(), device=env.unwrapped.device)
        runner.load(str(Path(args.checkpoint).resolve()), load_optimizer=False)
        policy = runner.get_inference_policy(device=env.unwrapped.device)
        robot = env.unwrapped.scene["robot"]
        initial = robot.data.root_pos_w.clone()
        device = initial.device
        count = args.num_envs
        active = torch.ones(count, device=device, dtype=torch.bool)
        success = torch.zeros_like(active)
        terminated = torch.zeros_like(active)
        unsafe = torch.zeros_like(active)
        reset_count = torch.zeros(count, device=device, dtype=torch.long)
        whole_run_tilt = torch.zeros(count, device=device)
        crossed = torch.full((count,), -1., device=device)
        elapsed = torch.zeros(count, device=device)
        progress = torch.zeros(count, device=device)
        lateral = torch.zeros(count, device=device)
        tilt = torch.zeros(count, device=device)
        traces = []
        obs = env.get_observations()
        for step in range(steps):
            with torch.inference_mode():
                obs, _, done, _ = env.step(policy(obs))
                runner.alg.policy.reset(done)
                valid = active & ~done.bool()
                elapsed[active] += dt
                relative = robot.data.root_pos_w - initial
                angle = torch.rad2deg(torch.acos((-robot.data.projected_gravity_b[:, 2]).clamp(-1, 1)))
                reset_count += done.long()
                whole_run_tilt = torch.maximum(whole_run_tilt, angle)
                progress[valid] = relative[valid, 0]
                lateral[valid] = torch.maximum(lateral[valid], relative[valid, 1].abs())
                tilt[valid] = torch.maximum(tilt[valid], angle[valid])
                # Ramp starts 1 m ahead and ends 5 m ahead of the spawn point.
                # Require the body 0.6 m beyond it, then another second upright.
                newly_crossed = valid & (crossed < 0) & (relative[:, 0] >= 5.6)
                crossed[newly_crossed] = elapsed[newly_crossed]
                bad = valid & ((angle >= 45) | (relative[:, 1].abs() >= lateral_limit))
                unsafe |= bad
                terminated |= active & done.bool()
                passed = valid & ~unsafe & (crossed >= 0) & (elapsed - crossed >= 1.) & (relative[:, 0] >= 5.3)
                success |= passed
                active &= ~(done.bool() | bad | passed)
                if step % 10 == 0 and count == 1:
                    traces.append({"time_s": round((step + 1) * dt, 3),
                                   "x_from_spawn_m": float(relative[0, 0]),
                                   "z_world_m": float(robot.data.root_pos_w[0, 2]),
                                   "tilt_deg": float(angle[0]), "done": bool(done[0])})
            if not args.video_folder and not active.any():
                break
        trials = [{"passed": bool(success[i]), "terminated_before_pass": bool(terminated[i]),
                   "unsafe_before_pass": bool(unsafe[i]), "x_progress_m": float(progress[i]),
                   "elapsed_s": float(elapsed[i]), "crossed_at_s": float(crossed[i]),
                   "max_tilt_deg": float(tilt[i]), "max_lateral_drift_m": float(lateral[i])}
                  for i in range(count)]
        result = {"task": task, "checkpoint": str(Path(args.checkpoint).resolve()),
                  "slope_degrees": args.slope_degrees, "direction": args.direction,
                  "seed": args.seed, "ramp_length_m": terrain.ramp_length,
                  "duration_budget_s": args.duration, "environment_count": count,
                  "passed": int(success.sum()), "trials": trials, "trajectory": traces,
                  "whole_run_reset_count": reset_count.cpu().tolist(),
                  "whole_run_max_tilt_deg": whole_run_tilt.cpu().tolist(),
                  "lateral_limit_m": lateral_limit,
                  "criterion": "advance 5.6m beyond spawn and stay upright for 1s; no reset, tilt>=45deg or drift outside middle half of ramp before pass",
                  "video": str(Path(args.video_folder).resolve() / (args.video_name + "-step-0.mp4")) if args.video_folder else None}
        output = Path(args.output).resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(f"[SLOPE RESULT] {args.direction} {args.slope_degrees:g}deg: {int(success.sum())}/{count} passed; report={output}", flush=True)
    finally:
        env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
