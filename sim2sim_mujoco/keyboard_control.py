"""Focused-window keyboard commands for the passive MuJoCo viewer.

Windows uses physical key state so key release is observable (the MuJoCo
key_callback only supplies press events). Other platforms use press-to-toggle
commands through that callback, with Space to clear them. No extra dependency
or global keyboard hook is required.
"""

from __future__ import annotations

import ctypes
import os
import threading

import numpy as np


# GLFW codes used by MuJoCo key_callback; Windows virtual keys differ.
UP, DOWN, LEFT, RIGHT = 265, 264, 263, 262
Q, E = ord("Q"), ord("E")
SPACE, R, M, PLUS, MINUS = 32, ord("R"), ord("M"), ord("="), ord("-")
KEYS = (UP, DOWN, LEFT, RIGHT, Q, E, SPACE, R, M, PLUS, MINUS)
WINDOWS_VIRTUAL_KEYS = {UP: 0x26, DOWN: 0x28, LEFT: 0x25, RIGHT: 0x27, PLUS: 0xBB, MINUS: 0xBD}


class WindowsKeyState:
    """Read key state only while a window of this process has focus."""

    def __init__(self):
        from ctypes import wintypes

        self.user32 = ctypes.WinDLL("user32", use_last_error=True)
        self.user32.GetForegroundWindow.argtypes = []
        self.user32.GetForegroundWindow.restype = wintypes.HWND
        self.user32.GetWindowThreadProcessId.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.DWORD)]
        self.user32.GetWindowThreadProcessId.restype = wintypes.DWORD
        self.user32.GetAsyncKeyState.argtypes = [ctypes.c_int]
        self.user32.GetAsyncKeyState.restype = ctypes.c_short
        self.process_id = os.getpid()

    def sample(self) -> tuple[set[int], bool]:
        from ctypes import wintypes

        process_id = wintypes.DWORD()
        window = self.user32.GetForegroundWindow()
        self.user32.GetWindowThreadProcessId(window, ctypes.byref(process_id))
        if process_id.value != self.process_id:
            return set(), False
        # GetAsyncKeyState takes Windows virtual keys, not GLFW key codes.
        pressed = {
            key for key in KEYS
            if self.user32.GetAsyncKeyState(WINDOWS_VIRTUAL_KEYS.get(key, key)) & 0x8000
        }
        return pressed, True


class KeyboardController:
    """Convert arrow/QE input to smoothly changing body-frame commands."""

    def __init__(self, move_speed: float = 0.5, strafe_speed: float = 0.15, turn_speed: float = 0.2):
        if not (0.0 < move_speed <= 1.0 and 0.0 < strafe_speed <= 0.3 and 0.0 < turn_speed <= 0.5):
            raise ValueError("Keyboard speeds must be vx in (0,1], vy in (0,0.3], wz in (0,0.5].")
        self.move_speed = move_speed
        self.strafe_speed = strafe_speed
        self.turn_speed = turn_speed
        self.command = np.zeros(3, dtype=np.float32)
        self.acceleration = np.array([1.0, 0.5, 1.0], dtype=np.float32)
        self.previous_keys: set[int] = set()
        self.reader = WindowsKeyState() if os.name == "nt" else None
        self.mode = "hold" if self.reader is not None else "toggle"
        self.focused = False
        self._latched: set[int] = set()
        self._events: set[int] = set()
        self._lock = threading.Lock()

    def on_key(self, keycode: int) -> None:
        """MuJoCo press-event callback, used for the portable toggle fallback."""
        if self.reader is not None:
            return
        with self._lock:
            if keycode == SPACE:
                self._latched.clear()
                self._events.add(SPACE)
            elif keycode in (R, M, PLUS, MINUS):
                self._events.add(keycode)
                if keycode == R:
                    self._latched.clear()
            elif keycode in (UP, DOWN, LEFT, RIGHT, Q, E):
                opposite = {UP: DOWN, DOWN: UP, LEFT: RIGHT, RIGHT: LEFT, Q: E, E: Q}[keycode]
                if keycode in self._latched:
                    self._latched.remove(keycode)
                else:
                    self._latched.discard(opposite)
                    self._latched.add(keycode)

    def poll(self, dt: float) -> tuple[np.ndarray, bool, bool]:
        if self.reader is not None:
            keys, focused = self.reader.sample()
        else:
            with self._lock:
                keys = self._latched | self._events
                self._events.clear()
            focused = True
        return self.update(keys, dt, focused)

    def update(self, keys: set[int], dt: float, focused: bool = True) -> tuple[np.ndarray, bool, bool]:
        """Apply a snapshot; return command, reset request, camera request."""
        self.focused = focused
        if not focused:
            self.previous_keys.clear()
            self.command[:] = 0.0
            return self.command.copy(), False, False

        new_keys = keys - self.previous_keys
        self.previous_keys = keys.copy()
        if PLUS in new_keys:
            self.move_speed = min(1.0, round(self.move_speed + 0.1, 2))
        if MINUS in new_keys:
            self.move_speed = max(0.1, round(self.move_speed - 0.1, 2))
        reset = R in new_keys
        toggle_camera = M in new_keys
        if SPACE in keys or reset:
            self.command[:] = 0.0
        else:
            target = np.array(
                [
                    (int(UP in keys) - int(DOWN in keys)) * self.move_speed,
                    (int(LEFT in keys) - int(RIGHT in keys)) * self.strafe_speed,
                    (int(Q in keys) - int(E in keys)) * self.turn_speed,
                ],
                dtype=np.float32,
            )
            max_change = self.acceleration * dt
            self.command += np.clip(target - self.command, -max_change, max_change)
        return self.command.copy(), reset, toggle_camera

    def overlay(self) -> str:
        focus = "active" if self.focused else "click viewer to control"
        return (
            f"Keyboard {self.mode}: {focus}\n"
            "Up/Down: forward/back  Left/Right: strafe\n"
            "Q/E: turn  Space: stop  R: reset\n"
            "M: map/follow camera  +/-: forward speed\n"
            f"speed={self.move_speed:.2f}  cmd vx={self.command[0]:+.2f} "
            f"vy={self.command[1]:+.2f} wz={self.command[2]:+.2f}"
        )
