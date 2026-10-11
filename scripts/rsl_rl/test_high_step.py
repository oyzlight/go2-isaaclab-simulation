"""Small simulator-free tests for single-step geometry and curriculum logic."""

import importlib.util
import sys
import unittest
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
TASK = ROOT / "source/go2_demo/go2_demo/tasks/manager_based/go2_demo"


def load_file(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


geometry = load_file("high_step_geometry_test", TASK / "high_step_geometry.py")
# Only the command base class/math helper need simulator imports. Mock them for
# unit testing; the real classes are separately checked by the Isaac smoke run.
command_stub = ModuleType("isaaclab.envs.mdp.commands")
command_stub.UniformVelocityCommand = object
math_stub = ModuleType("isaaclab.utils.math")
math_stub.wrap_to_pi = lambda x: (x + torch.pi) % (2 * torch.pi) - torch.pi
math_stub.quat_apply_inverse = lambda quat, vec: vec.clone()  # fixtures use identity quaternion
event_stub = ModuleType("isaaclab.envs.mdp.events")
event_stub.reset_root_state_uniform = lambda *args, **kwargs: None
with patch.dict(sys.modules, {command_stub.__name__: command_stub, math_stub.__name__: math_stub}):
    mdp = load_file("high_step_mdp_test", TASK / "mdp/high_step.py")
stand_package = ModuleType("stand_test_package")
stand_package.__path__ = []
with patch.dict(sys.modules, {
    command_stub.__name__: command_stub, math_stub.__name__: math_stub,
    event_stub.__name__: event_stub, "stand_test_package": stand_package,
    "stand_test_package.high_step": mdp,
}):
    stand_mdp = load_file("stand_test_package.high_step_stand", TASK / "mdp/high_step_stand.py")


class HighStepTests(unittest.TestCase):
    def test_two_step_height_spacing_and_no_overlap(self):
        cfg = SimpleNamespace(
            size=(7.0, 4.0), height_levels=(0.28,), approach_length=2.0,
            ledge_distance=0.9, step_count=2, step_depth=0.7,
        )
        meshes, origin = geometry.two_step_terrain(0.5, cfg)
        self.assertEqual(len(meshes), 3)
        for mesh, height in zip(meshes, (0.0, 0.28, 0.56)):
            self.assertAlmostEqual(mesh.bounds[1, 2], height)
            self.assertTrue(mesh.is_watertight)
        self.assertAlmostEqual(meshes[1].bounds[0, 0] - origin[0], 0.9)
        self.assertAlmostEqual(meshes[2].bounds[0, 0] - origin[0], 1.6)
        self.assertAlmostEqual(meshes[1].bounds[1, 0], meshes[2].bounds[0, 0])
        self.assertEqual(mdp._target_geometry(cfg), (1.6, (0.56,)))
        cfg.step_depth = 4.0
        with self.assertRaises(ValueError):
            geometry.two_step_terrain(0.5, cfg)

    def test_single_step_goal_geometry_is_unchanged(self):
        cfg = SimpleNamespace(ledge_distance=0.9, height_levels=(0.08, 0.28))
        self.assertEqual(mdp._target_geometry(cfg), (0.9, (0.08, 0.28)))

    def test_geometry_every_row(self):
        heights = tuple(round(0.06 + 0.02 * i, 2) for i in range(12))
        cfg = SimpleNamespace(size=(7.0, 4.0), height_levels=heights, approach_length=2.0, ledge_distance=0.9)
        for index, height in enumerate(heights):
            for jitter in (0.0, 0.5, 0.999):
                meshes, origin = geometry.single_step_terrain((index + jitter) / 12, cfg)
                self.assertAlmostEqual(meshes[0].bounds[1, 2], 0.0)
                self.assertAlmostEqual(meshes[1].bounds[1, 2], height)
                self.assertAlmostEqual(meshes[1].bounds[0, 0] - origin[0], 0.9)
                self.assertAlmostEqual(meshes[0].bounds[1, 0], meshes[1].bounds[0, 0])
                np.testing.assert_allclose(origin, (2.0, 2.0, 0.0))
        self.assertAlmostEqual(geometry.single_step_terrain(1.0, cfg)[0][1].bounds[1, 2], 0.28)

    def test_success_needs_all_feet_and_hold(self):
        env = self.fake_env()
        # Merely moving the root beyond the ledge cannot count as a climb.
        env.scene["robot"].data.body_pos_w[:, 3, 0] = 0.5
        for _ in range(40):
            env.common_step_counter += 1
            self.assertFalse(bool(mdp.high_step_success(env).any()))
        env.scene["robot"].data.body_pos_w[:, 3, 0] = 1.6
        for _ in range(29):
            env.common_step_counter += 1
            self.assertFalse(bool(mdp.high_step_success(env).any()))
        for _ in range(2):
            env.common_step_counter += 1
            mdp.high_step_success(env)
        self.assertTrue(bool(mdp.high_step_success(env).all()))
        age = env._high_step_state["age"].clone()
        mdp.high_step_success(env)
        torch.testing.assert_close(age, env._high_step_state["age"])

    def test_curriculum_and_no_final_wrap(self):
        env = self.fake_env()
        state = mdp._state(env)
        ids = torch.arange(2)
        env.scene.terrain.terrain_levels[:] = torch.tensor([0, 1])
        state["age"][:] = 100
        state["success"][:] = True
        for _ in range(3):
            mdp.high_step_curriculum(env, ids)
        torch.testing.assert_close(env.scene.terrain.terrain_levels, torch.tensor([1, 1]))
        state["success"][:] = False
        for _ in range(3):
            mdp.high_step_curriculum(env, ids)
        torch.testing.assert_close(env.scene.terrain.terrain_levels, torch.tensor([0, 0]))
        state["success"][:] = True
        state["age"][:] = 0
        attempts = state["attempts"].clone()
        mdp.high_step_curriculum(env, ids)
        torch.testing.assert_close(attempts, state["attempts"])

    @staticmethod
    def fake_env():
        cfg = SimpleNamespace(height_levels=(0.08, 0.28), ledge_distance=0.9)
        data = SimpleNamespace(
            root_pos_w=torch.tensor([[1.7, 0, 0.6], [1.7, 0, 0.6]]),
            body_pos_w=torch.tensor([[[1.6, 0, 0.31]] * 4] * 2),
            projected_gravity_b=torch.tensor([[0., 0., -1.]] * 2),
            root_lin_vel_w=torch.zeros(2, 3), root_ang_vel_b=torch.zeros(2, 3),
        )
        robot = SimpleNamespace(data=data, find_bodies=lambda _: (list(range(4)), ["a", "b", "c", "d"]))
        sensor = SimpleNamespace(
            data=SimpleNamespace(net_forces_w=torch.tensor([[[0., 0., 30.]] * 4] * 2)),
            find_bodies=lambda *args, **kwargs: (list(range(4)), []),
        )
        class Scene(dict):
            pass
        scene = Scene(robot=robot, contact_forces=sensor)
        scene.env_origins = torch.zeros(2, 3)
        scene.terrain = SimpleNamespace(
            terrain_levels=torch.ones(2, dtype=torch.long), terrain_types=torch.zeros(2, dtype=torch.long),
            terrain_origins=torch.zeros(2, 1, 3), env_origins=scene.env_origins,
        )
        return SimpleNamespace(
            scene=scene, num_envs=2, device="cpu", step_dt=0.02, common_step_counter=0,
            cfg=SimpleNamespace(scene=SimpleNamespace(terrain=SimpleNamespace(
                terrain_generator=SimpleNamespace(sub_terrains={"single_step": cfg})
            ))),
        )


class HighStepStandTests(unittest.TestCase):
    @staticmethod
    def fake_env():
        env = HighStepTests.fake_env()
        robot = env.scene["robot"]
        prefixes = ("FL", "FR", "RL", "RR")
        robot.joint_names = [f"{prefix}_{joint}_joint" for joint in ("hip", "thigh", "calf") for prefix in prefixes]
        robot.find_bodies = lambda _: (list(range(4)), [f"{prefix}_foot" for prefix in prefixes])
        robot.data.default_joint_pos = torch.tensor([[0.1, -0.1, 0.1, -0.1, 0.8, 0.8, 1.0, 1.0, -1.5, -1.5, -1.5, -1.5]] * 2)
        robot.data.joint_pos = robot.data.default_joint_pos.clone()
        robot.data.root_lin_vel_b = robot.data.root_lin_vel_w  # identity-quaternion fixtures
        robot.data.root_quat_w = torch.tensor([[1., 0., 0., 0.]] * 2)
        nominal = stand_mdp.nominal_go2_feet(robot.data.default_joint_pos[0], robot.joint_names, [f"{p}_foot" for p in prefixes])
        robot.data.root_pos_w[:, 0] = 1.8
        robot.data.root_pos_w[:, 2] = 0.28 - nominal[:, 2].mean() + 0.022
        robot.data.body_pos_w = robot.data.root_pos_w[:, None, :] + nominal[None]
        env.command_manager = SimpleNamespace(get_command=lambda _: torch.zeros(2, 3))
        env._stand_platform_start = torch.zeros(2, dtype=torch.bool)
        return env

    def test_nominal_reference_matches_local_mjcf(self):
        try:
            import mujoco
        except ImportError:
            self.skipTest("MuJoCo not installed")
        model = mujoco.MjModel.from_xml_path(str(ROOT / "sim2sim_mujoco/unitree_mujoco/unitree_robots/go2/go2.xml"))
        data = mujoco.MjData(model)
        env = self.fake_env()
        robot = env.scene["robot"]
        for index, name in enumerate(robot.joint_names):
            jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
            data.qpos[model.jnt_qposadr[jid]] = float(robot.data.default_joint_pos[0, index])
        mujoco.mj_forward(model, data)
        base_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base_link")
        _, names = robot.find_bodies(".*_foot")
        actual = np.array([data.xpos[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, n)] - data.xpos[base_id] for n in names])
        nominal = stand_mdp._stand_state(env)["nominal_feet"].numpy()
        np.testing.assert_allclose(nominal, actual, atol=1e-6)

    def test_two_step_does_not_settle_on_first_tread(self):
        env = self.fake_env()
        cfg = env.cfg.scene.terrain.terrain_generator.sub_terrains["single_step"]
        cfg.step_count = 2
        cfg.step_depth = 0.7
        env.common_step_counter += 1
        self.assertFalse(bool(stand_mdp.high_step_stand_success(env).any()))
        self.assertFalse(bool(env._high_step_stand_state["settling"].any()))
        self.assertAlmostEqual(env._high_step_state["edge"], 1.6)
        self.assertAlmostEqual(float(env._high_step_state["heights"][1]), 0.56, places=6)
        robot = env.scene["robot"]
        nominal = env._high_step_stand_state["nominal_feet"]
        robot.data.root_pos_w[:, 0] = 2.5
        robot.data.root_pos_w[:, 2] = 0.56 + env._high_step_stand_state["nominal_height"]
        robot.data.body_pos_w = robot.data.root_pos_w[:, None, :] + nominal[None]
        for _ in range(102):
            env.common_step_counter += 1
            stand_mdp.high_step_stand_success(env)
        self.assertTrue(bool(env._high_step_state["success"].all()))

    def test_correct_stance_holds_two_seconds(self):
        env = self.fake_env()
        for _ in range(99):
            env.common_step_counter += 1
            self.assertFalse(bool(stand_mdp.high_step_stand_success(env).any()))
        for _ in range(3):
            env.common_step_counter += 1
            stand_mdp.high_step_stand_success(env)
        self.assertTrue(bool(env._high_step_state["success"].all()))
        hold = env._high_step_stand_state["hold"].clone()
        stand_mdp.high_step_stand_success(env)
        torch.testing.assert_close(hold, env._high_step_stand_state["hold"])

    def test_hanging_leg_wall_contact_inward_hips_and_creep_fail(self):
        for problem in ("unloaded", "hanging", "wall", "hips", "placement", "creep", "command"):
            with self.subTest(problem=problem):
                env = self.fake_env()
                robot = env.scene["robot"]
                if problem == "unloaded":
                    env.scene["contact_forces"].data.net_forces_w[:, 3, :] = 0
                elif problem == "hanging":
                    robot.data.body_pos_w[:, 3, 2] += 0.20
                elif problem == "wall":
                    env.scene["contact_forces"].data.net_forces_w[:, 3, :] = torch.tensor([30., 0., 0.])
                elif problem == "hips":
                    robot.data.joint_pos[:, :4] += 0.3
                elif problem == "placement":
                    robot.data.body_pos_w[:, 3, 1] += 0.1
                elif problem == "creep":
                    robot.data.root_lin_vel_w[:, 0] = 0.15
                else:
                    env.command_manager.get_command = lambda _: torch.tensor([[0.1, 0., 0.]] * 2)
                for _ in range(110):
                    env.common_step_counter += 1
                    self.assertFalse(bool(stand_mdp.high_step_stand_success(env).any()))

    def test_pose_rewards_are_off_during_approach(self):
        env = self.fake_env()
        env.scene["robot"].data.root_pos_w[:, 0] = 0.0
        env.scene["robot"].data.joint_pos += 1
        env.common_step_counter += 1
        stand_mdp.high_step_stand_success(env)
        for fn in (stand_mdp.stand_pose_penalty, stand_mdp.stand_foot_placement_penalty,
                   stand_mdp.stand_orientation_penalty, stand_mdp.stand_quality_reward):
            torch.testing.assert_close(fn(env), torch.zeros(2))

    def test_curriculum_snapshots_survive_reset(self):
        env = self.fake_env()
        for _ in range(102):
            env.common_step_counter += 1
            stand_mdp.high_step_stand_success(env)
        ids = torch.arange(2)
        stand_mdp.high_step_stand_curriculum(env, ids)
        stand_mdp.reset_high_step_stand_state(env, ids)
        self.assertTrue(bool(env._high_step_state["last_success"].all()))
        self.assertTrue(bool(env._high_step_stand_state["last_climbed"].all()))
        torch.testing.assert_close(env._high_step_stand_state["last_contact_count"], torch.tensor([4., 4.]))
        self.assertFalse(bool(env._high_step_state["success"].any()))

    def test_platform_practice_resets_only_training_probability(self):
        env = self.fake_env()
        robot = env.scene["robot"]
        robot.data.root_state_w = torch.zeros(2, 13)
        robot.data.default_root_state = torch.zeros(2, 13)
        robot.data.default_root_state[:, 2] = 0.4
        robot.write_root_pose_to_sim = lambda pose, env_ids: robot.data.root_pos_w.__setitem__(env_ids, pose[:, :3])
        ids = torch.arange(2)
        stand_mdp.reset_high_step_stand_base(env, ids, {}, {}, platform_probability=1.0)
        torch.testing.assert_close(robot.data.root_pos_w[:, 0], torch.tensor([1.8, 1.8]))
        torch.testing.assert_close(robot.data.root_pos_w[:, 2], torch.tensor([0.68, 0.68]))
        self.assertTrue(bool(env._stand_platform_start.all()))
        stand_mdp.reset_high_step_stand_base(env, ids, {}, {}, platform_probability=0.0)
        self.assertFalse(bool(env._stand_platform_start.any()))

    def test_three_feet_get_only_ten_percent_tracking_and_no_stance_bonus(self):
        for missing_foot in range(4):
            with self.subTest(missing_foot=missing_foot):
                env = self.fake_env()
                env.scene["contact_forces"].data.net_forces_w[:, missing_foot, :] = 0
                env.common_step_counter += 1
                stand_mdp.high_step_stand_success(env)
                torch.testing.assert_close(stand_mdp.stand_support_reward(env), torch.zeros(2))
                torch.testing.assert_close(stand_mdp.stand_quality_reward(env), torch.zeros(2))
                lin = stand_mdp.stand_track_lin_vel_xy_exp(env, std=0.5, command_name="base_velocity")
                yaw = stand_mdp.stand_track_ang_vel_z_exp(env, std=0.5, command_name="base_velocity")
                torch.testing.assert_close(lin, torch.full((2,), 0.1))
                torch.testing.assert_close(yaw, torch.full((2,), 0.1))
                # Weighted velocity rewards: 3.0 + 1.5 -> 0.3 + 0.15 per second.
                torch.testing.assert_close(3.0 * lin + 1.5 * yaw, torch.full((2,), 0.45))

    def test_four_feet_keep_original_positive_rewards(self):
        env = self.fake_env()
        env.common_step_counter += 1
        stand_mdp.high_step_stand_success(env)
        for reward in (
            stand_mdp.stand_support_reward(env), stand_mdp.stand_quality_reward(env),
            stand_mdp.stand_track_lin_vel_xy_exp(env, std=0.5, command_name="base_velocity"),
            stand_mdp.stand_track_ang_vel_z_exp(env, std=0.5, command_name="base_velocity"),
        ):
            torch.testing.assert_close(reward, torch.ones(2))

    def test_climbing_tracking_is_unchanged_without_four_supports(self):
        env = self.fake_env()
        env.scene["robot"].data.root_pos_w[:, 0] = 0.0
        env.scene["contact_forces"].data.net_forces_w[:, 3, :] = 0
        env.command_manager.get_command = lambda _: torch.tensor([[0.4, 0., 0.2]] * 2)
        env.common_step_counter += 1
        stand_mdp.high_step_stand_success(env)
        torch.testing.assert_close(stand_mdp.stand_tracking_scale(env), torch.ones(2))
        lin = stand_mdp.stand_track_lin_vel_xy_exp(env, std=0.5, command_name="base_velocity")
        yaw = stand_mdp.stand_track_ang_vel_z_exp(env, std=0.5, command_name="base_velocity")
        torch.testing.assert_close(lin, torch.exp(torch.full((2,), -0.4**2 / 0.5**2)))
        torch.testing.assert_close(yaw, torch.exp(torch.full((2,), -0.2**2 / 0.5**2)))

    def test_reduction_is_per_environment_and_configurable(self):
        env = self.fake_env()
        env.scene["contact_forces"].data.net_forces_w[0, 3, :] = 0
        env.common_step_counter += 1
        stand_mdp.high_step_stand_success(env)
        torch.testing.assert_close(stand_mdp.stand_tracking_scale(env), torch.tensor([0.1, 1.0]))
        torch.testing.assert_close(stand_mdp.stand_tracking_scale(env, 0.0), torch.tensor([0.0, 1.0]))
        for invalid in (-0.1, 1.1):
            with self.assertRaises(ValueError):
                stand_mdp.stand_tracking_scale(env, invalid)


if __name__ == "__main__":
    unittest.main()
