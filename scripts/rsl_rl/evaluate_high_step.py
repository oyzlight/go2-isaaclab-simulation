"""Deterministic, fixed-height trials. Does not train or overwrite the source model."""

import argparse
import json
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--task", default="Go2-High-Step-28cm-Play-v0")
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--step_height", type=float, default=0.28, help="Vertical ledge height in metres")
parser.add_argument("--num_envs", type=int, default=32)
parser.add_argument("--episodes_per_env", type=int, default=4)
parser.add_argument("--output", required=True)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if not 0.01 <= args.step_height <= 0.30 or args.episodes_per_env < 1 or args.num_envs < 1:
    parser.error("Use height 0.01..0.30 m, and positive environment/trial counts")
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import gymnasium as gym
import torch
from rsl_rl.runners import OnPolicyRunner

import go2_demo.tasks  # noqa: F401
from isaaclab_tasks.utils import load_cfg_from_registry
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper


def main():
    task = args.task
    env_cfg = load_cfg_from_registry(task, "env_cfg_entry_point")
    agent_cfg = load_cfg_from_registry(task, "rsl_rl_cfg_entry_point")
    env_cfg.scene.num_envs = args.num_envs
    env_cfg.scene.terrain.terrain_generator.sub_terrains["single_step"].height_levels = (args.step_height,)
    env_cfg.seed = args.seed
    env_cfg.sim.device = args.device
    env = RslRlVecEnvWrapper(gym.make(task, cfg=env_cfg), clip_actions=agent_cfg.clip_actions)
    try:
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), device=env.unwrapped.device)
        runner.load(str(Path(args.checkpoint).resolve()), load_optimizer=False)
        policy = runner.get_inference_policy(device=env.unwrapped.device)
        obs = env.get_observations()
        shape = tuple(obs["policy"].shape)
        if shape != (args.num_envs, 235):
            raise RuntimeError(f"Expected checkpoint-compatible 235-D policy observations, got {shape}")
        completed = torch.zeros(args.num_envs, device=env.unwrapped.device, dtype=torch.long)
        wins = torch.zeros_like(completed)
        times = torch.zeros(args.num_envs, device=env.unwrapped.device)
        elapsed = torch.zeros_like(times)
        climbed = torch.zeros_like(completed)
        strict_stance = hasattr(env.unwrapped, "_high_step_stand_state")
        if strict_stance and env_cfg.events.reset_base.params["platform_probability"] != 0:
            raise ValueError("Strict evaluation must start below the ledge, never on the platform")
        stand_metrics = {key: [] for key in ("contact_count", "hip_error", "leg_error", "foot_error", "tilt_deg", "speed")}
        terminal_stand_metrics = {key: [] for key in stand_metrics}
        maximum_stable_holds = []
        initial_fk_error = None
        if strict_stance:
            from isaaclab.utils.math import quat_apply_inverse
            robot = env.unwrapped.scene["robot"]
            state = env.unwrapped._high_step_state
            stand = env.unwrapped._high_step_stand_state
            actual = quat_apply_inverse(
                robot.data.root_quat_w[:, None, :].expand(-1, 4, -1),
                robot.data.body_pos_w[:, state["feet"], :] - robot.data.root_pos_w[:, None, :],
            )
            initial_fk_error = float(torch.linalg.vector_norm(actual - stand["nominal_feet"][None], dim=-1).max())
            if initial_fk_error > 0.015:
                raise RuntimeError(f"Nominal stance FK differs from Isaac geometry by {initial_fk_error:.4f} m")
        # Every lane gets exactly the requested number of attempts; do not bias
        # the result toward lanes that reset/fail more frequently.
        budget = (env.unwrapped.max_episode_length + 10) * args.episodes_per_env
        for _ in range(budget):
            with torch.inference_mode():
                obs, _, done, _ = env.step(policy(obs))
            elapsed += env.unwrapped.step_dt
            ids = (done.bool() & (completed < args.episodes_per_env)).nonzero().flatten()
            success = env.unwrapped._high_step_state["last_success"][ids]
            wins[ids] += success.long()
            if strict_stance:
                stand = env.unwrapped._high_step_stand_state
                climbed[ids] += stand["last_climbed"][ids].long()
                success_ids = ids[success]
                for key in stand_metrics:
                    stand_metrics[key].extend(stand[f"last_{key}"][success_ids].cpu().tolist())
                    terminal_stand_metrics[key].extend(stand[f"last_{key}"][ids].cpu().tolist())
                maximum_stable_holds.extend(stand["last_max_hold"][ids].cpu().tolist())
            times[ids] += elapsed[ids] * success
            completed[ids] += 1
            elapsed[done.bool()] = 0
            runner.alg.policy.reset(done)
            if bool((completed >= args.episodes_per_env).all()):
                break
        result = {
            "checkpoint": str(Path(args.checkpoint).resolve()),
            "task": task,
            "height_m": args.step_height,
            "policy_observations": shape[1],
            "deterministic_actions": True,
            "seed": env_cfg.seed,
            "episodes": int(completed.sum()),
            "successes": int(wins.sum()),
            "success_rate": float(wins.sum() / completed.sum().clamp(min=1)),
            "mean_success_time_s": float(times.sum() / wins.sum()) if wins.sum() > 0 else None,
            "complete": bool((completed >= args.episodes_per_env).all()),
            "criterion": "All 4 feet past ledge, >=2 supporting contacts, upright, slow, held 0.6 s",
        }
        if strict_stance:
            result["criterion"] = {
                "description": "Climb FROM BELOW, zero command, FOUR upward-load contacts and nominal stance continuously held",
                **env_cfg.terminations.step_success.params,
            }
            result["initial_nominal_fk_error_m"] = initial_fk_error
            result["climb_successes"] = int(climbed.sum())
            result["climb_success_rate"] = float(climbed.sum() / completed.sum().clamp(min=1))
            result["strict_stand_success_rate"] = result["success_rate"]
            result["successful_stance_metrics"] = {
                key: {"mean": sum(values) / len(values), "max": max(values)} if values else None
                for key, values in stand_metrics.items()
            }
            result["terminal_stance_metrics_all_attempts"] = {
                key: {"mean": sum(values) / len(values), "max": max(values)} if values else None
                for key, values in terminal_stand_metrics.items()
            }
            result["best_continuous_stand_hold_s"] = max(maximum_stable_holds) if maximum_stable_holds else 0.0
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(json.dumps(result, indent=2), flush=True)
        if not result["complete"]:
            raise RuntimeError("Evaluation ended before every lane completed its trials")
    finally:
        env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
