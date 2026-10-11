"""Simulator-free tests for GUI request handling and GPU mouse protection."""

import unittest
from types import SimpleNamespace

from playback_controls import PlaybackControls, add_playback_panel


class Settings(dict):
    def set_bool(self, key, value):
        self[key] = value


class PlaybackControlTests(unittest.TestCase):
    def test_gpu_mouse_path_disabled_and_reenable_is_blocked(self):
        settings = Settings({PlaybackControls.MOUSE_INTERACTION_SETTING: True, "gpu_physics": True})
        controls = PlaybackControls("cuda:0", settings)
        self.assertTrue(controls.enforce_gpu_mouse_safety())
        self.assertFalse(settings[controls.MOUSE_INTERACTION_SETTING])
        self.assertTrue(settings["gpu_physics"])
        self.assertFalse(controls.enforce_gpu_mouse_safety())
        settings[controls.MOUSE_INTERACTION_SETTING] = True
        self.assertTrue(controls.enforce_gpu_mouse_safety())

    def test_cpu_mouse_settings_are_untouched(self):
        settings = Settings({PlaybackControls.MOUSE_INTERACTION_SETTING: True})
        controls = PlaybackControls("cpu", settings)
        self.assertFalse(controls.enforce_gpu_mouse_safety())
        self.assertTrue(settings[controls.MOUSE_INTERACTION_SETTING])

    def test_unset_gpu_mouse_setting_is_disabled(self):
        controls = PlaybackControls("cuda", Settings())
        self.assertTrue(controls.enforce_gpu_mouse_safety())
        self.assertFalse(controls.settings[controls.MOUSE_INTERACTION_SETTING])

    def test_reset_is_queued_and_policy_state_cleared_once(self):
        calls = []
        obs = {"policy": "fresh_observations"}
        env = SimpleNamespace(reset=lambda: (calls.append("env_reset") or obs, {}))
        controls = PlaybackControls("cuda:0", Settings())
        controls.request_reset()
        self.assertEqual(calls, [])
        result = controls.apply_pending_reset(env, lambda: calls.append("policy_reset"))
        self.assertIs(result, obs)
        self.assertEqual(calls, ["env_reset", "policy_reset"])
        self.assertIsNone(controls.apply_pending_reset(env, lambda: calls.append("unexpected")))

    def test_paused_reset_restores_pause_after_reset(self):
        calls = []
        env = SimpleNamespace(
            reset=lambda: (calls.append("env_reset") or {"policy": "fresh"}, {}),
            unwrapped=SimpleNamespace(sim=SimpleNamespace(pause=lambda: calls.append("pause"))),
        )
        controls = PlaybackControls("cuda:0", Settings())
        controls.request_reset(pause_after=True)
        controls.request_reset(pause_after=False)
        controls.apply_pending_reset(env, lambda: calls.append("policy_reset"))
        self.assertEqual(calls, ["env_reset", "policy_reset", "pause"])

    def test_no_gui_has_no_panel(self):
        env = SimpleNamespace(unwrapped=SimpleNamespace(window=None))
        self.assertFalse(add_playback_panel(env, PlaybackControls("cuda", Settings())))


if __name__ == "__main__":
    unittest.main()
