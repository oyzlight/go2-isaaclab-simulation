"""Check keyboard command sequences independently of the OS viewer."""

import unittest

import numpy as np

from keyboard_control import (
    DOWN, E, LEFT, M, MINUS, PLUS, Q, R, RIGHT, SPACE, UP, WINDOWS_VIRTUAL_KEYS, KeyboardController,
)


class KeyboardControlTest(unittest.TestCase):
    def test_holding_multiple_keys_release_and_stop(self):
        controls = KeyboardController()
        for _ in range(100):
            command, _, _ = controls.update({UP, LEFT, Q}, 0.01)
        np.testing.assert_allclose(command, [0.5, 0.15, 0.2], atol=1e-6)
        for _ in range(100):
            command, _, _ = controls.update(set(), 0.01)
        np.testing.assert_allclose(command, [0.0, 0.0, 0.0], atol=1e-6)
        for _ in range(100):
            command, _, _ = controls.update({DOWN, RIGHT, E}, 0.01)
        np.testing.assert_allclose(command, [-0.5, -0.15, -0.2], atol=1e-6)
        command, _, _ = controls.update({UP, SPACE}, 0.01)
        np.testing.assert_array_equal(command, [0.0, 0.0, 0.0])

    def test_focus_loss_clears_motion_and_does_not_request_reset(self):
        controls = KeyboardController()
        controls.update({UP}, 1.0)
        command, reset, camera = controls.update({R, M, UP}, 0.01, focused=False)
        np.testing.assert_array_equal(command, [0.0, 0.0, 0.0])
        self.assertFalse(reset)
        self.assertFalse(camera)

    def test_reset_and_camera_are_edge_triggered(self):
        controls = KeyboardController()
        _, reset, camera = controls.update({R, M}, 0.01)
        self.assertTrue(reset)
        self.assertTrue(camera)
        _, reset, camera = controls.update({R, M}, 0.01)
        self.assertFalse(reset)
        self.assertFalse(camera)

    def test_speed_changes_once_per_press_and_stays_bounded(self):
        controls = KeyboardController()
        for _ in range(20):
            controls.update({PLUS}, 0.01)
        self.assertAlmostEqual(controls.move_speed, 0.6)
        for _ in range(20):
            controls.update(set(), 0.01)
            controls.update({PLUS}, 0.01)
        self.assertEqual(controls.move_speed, 1.0)
        for _ in range(20):
            controls.update(set(), 0.01)
            controls.update({MINUS}, 0.01)
        self.assertEqual(controls.move_speed, 0.1)

    def test_glfw_arrow_codes_are_translated_to_windows_virtual_keys(self):
        self.assertEqual([WINDOWS_VIRTUAL_KEYS[key] for key in (UP, DOWN, LEFT, RIGHT)], [38, 40, 37, 39])


if __name__ == "__main__":
    unittest.main()
