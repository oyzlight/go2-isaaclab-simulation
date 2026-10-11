"""Regression checks for Isaac Lab/MuJoCo observation and motor semantics."""

import unittest

import mujoco
import numpy as np
import torch

from deploy import DEFAULT_MODEL, DEFAULT_POLICY, PLAYGROUND_MODEL, STAIRS_MODEL, Go2Sim2Sim


class Sim2SimInterfaceTest(unittest.TestCase):
    def setUp(self):
        self.sim = Go2Sim2Sim(DEFAULT_MODEL, DEFAULT_POLICY, np.array([0.5, 0.0, 0.0]))

    def test_velocity_is_in_link_frame_not_principal_inertia_frame(self):
        # Go2's inertial frame is strongly rotated from its link axes.
        self.sim.data.qvel[:3] = [0.4, -0.2, 0.1]
        self.sim.data.qvel[3:6] = [1.0, 2.0, 3.0]
        mujoco.mj_forward(self.sim.model, self.sim.data)
        lin, ang = self.sim.body_velocity()
        expected_ang = np.array([1.0, 2.0, 3.0])
        expected_lin = self.sim.data.qvel[:3] + np.cross(
            expected_ang, self.sim.model.body_ipos[self.sim.base_body_id]
        )
        np.testing.assert_allclose(ang, expected_ang, atol=1e-10)
        np.testing.assert_allclose(lin, expected_lin, atol=1e-10)

    def test_last_action_is_most_recent_action_and_is_not_clipped(self):
        class ProbePolicy:
            def __init__(self):
                self.inputs = []

            def __call__(self, obs):
                self.inputs.append(obs.clone())
                return torch.full((1, 12), 6.0 if len(self.inputs) == 1 else 2.0)

        probe = ProbePolicy()
        self.sim.policy = probe
        self.sim.infer()
        self.sim.infer()
        np.testing.assert_array_equal(probe.inputs[0][0, 36:48].numpy(), np.zeros(12))
        np.testing.assert_array_equal(probe.inputs[1][0, 36:48].numpy(), np.full(12, 6.0))
        np.testing.assert_array_equal(self.sim.observation()[36:48], np.full(12, 2.0))

    def test_dc_motor_torque_decreases_at_high_speed(self):
        self.sim.data.qpos[self.sim.joint_qpos_adr] = -10.0
        self.sim.data.qvel[self.sim.joint_dof_adr] = 15.0
        self.sim.apply_pd(False)
        np.testing.assert_allclose(self.sim.data.ctrl[self.sim.actuator_ids], 11.75)
        self.sim.data.qpos[self.sim.joint_qpos_adr] = 10.0
        self.sim.data.qvel[self.sim.joint_dof_adr] = -15.0
        self.sim.apply_pd(False)
        np.testing.assert_allclose(self.sim.data.ctrl[self.sim.actuator_ids], -11.75)

    def test_height_scan_ignores_robot_and_matches_flat_reference(self):
        np.testing.assert_allclose(self.sim.height_scan(), np.full(187, -0.1), atol=1e-6)
        self.sim.data.qpos[:3] = [2.0, -1.0, 0.6]
        self.sim.data.qpos[3:7] = [np.cos(0.35), 0.0, 0.0, np.sin(0.35)]
        mujoco.mj_forward(self.sim.model, self.sim.data)
        np.testing.assert_allclose(self.sim.height_scan(), np.full(187, 0.1), atol=1e-6)

    def test_stair_height_scan_layout_is_y_rows_and_x_columns(self):
        sim = Go2Sim2Sim(STAIRS_MODEL, DEFAULT_POLICY, np.array([0.5, 0.0, 0.0]))
        sim.data.qpos[:3] = [1.7, 0.0, 0.7]
        mujoco.mj_forward(sim.model, sim.data)
        grid = sim.height_scan().reshape(11, 17)
        # Straight stairs are uniform in y: all 11 rows must match, while
        # scanning forward in x encounters higher treads (smaller height obs).
        np.testing.assert_allclose(grid, np.repeat(grid[5:6], 11, axis=0), atol=1e-6)
        self.assertGreater(float(grid[5, 0]), float(grid[5, -1]))

    def test_playground_height_rays_see_bumps_but_ignore_pad_markings(self):
        sim = Go2Sim2Sim(PLAYGROUND_MODEL, DEFAULT_POLICY, np.array([0.5, 0.0, 0.0]))
        np.testing.assert_allclose(sim.height_scan(), np.full(187, -0.1), atol=1e-6)
        # At the low ridge the rays must hit both the raised pad (1 cm) and
        # the ridge top (3 cm), not the visual-only ground markings.
        sim.data.qpos[:3] = [8.5, 0.0, 0.5]
        mujoco.mj_forward(sim.model, sim.data)
        heights = sim.height_scan()
        self.assertLessEqual(float(heights.min()), -0.029)
        self.assertGreaterEqual(float(heights.max()), -0.011)


if __name__ == "__main__":
    unittest.main()
