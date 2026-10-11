"""Small, testable playback controls; no native PhysX addForce/addTorque calls."""


class PlaybackControls:
    MOUSE_INTERACTION_SETTING = "/physics/mouseInteractionEnabled"

    def __init__(self, device, settings):
        self.uses_gpu_physics = str(device).startswith("cuda")
        self.settings = settings
        self._reset_requested = False
        self._pause_after_reset = False

    def enforce_gpu_mouse_safety(self):
        """Disable the CPU mouse-force path, not GPU physics or error reporting."""
        if self.uses_gpu_physics and self.settings.get(self.MOUSE_INTERACTION_SETTING) is not False:
            self.settings.set_bool(self.MOUSE_INTERACTION_SETTING, False)
            return True
        return False

    def request_reset(self, pause_after=False):
        # UI callbacks only queue a reset. Physics/managers are reset by the
        # playback loop after an in-flight environment step has finished.
        self._reset_requested = True
        self._pause_after_reset |= pause_after

    def apply_pending_reset(self, env, reset_policy):
        if not self._reset_requested:
            return None
        pause_after = self._pause_after_reset
        self._reset_requested = False
        self._pause_after_reset = False
        observations, _ = env.reset()
        reset_policy()
        if pause_after:
            env.unwrapped.sim.pause()
        return observations


def add_playback_panel(env, controls):
    """Append safe controls to the existing IsaacLab Simulation Settings panel."""
    window = getattr(env.unwrapped, "window", None)
    if window is None or "sim_vstack" not in window.ui_window_elements:
        return False
    import omni.ui as ui

    def request_reset():
        sim = env.unwrapped.sim
        paused = not sim.is_playing()
        controls.request_reset(pause_after=paused)
        if paused:
            # Release SimulationContext.step's internal pause/render loop so
            # the queued reset can be performed safely by the outer play loop.
            sim.play()

    with window.ui_window_elements["sim_vstack"]:
        if controls.uses_gpu_physics:
            window.ui_window_elements["gpu_mouse_note"] = ui.Label(
                "GPU mouse dragging is disabled. Use reset below.", height=24
            )
        window.ui_window_elements["restart_episode_button"] = ui.Button(
            "Reset robot to start",
            height=32,
            tooltip="Restart all robots and clear action/observation/policy histories. Keeps the current model.",
            clicked_fn=request_reset,
        )
    return True
