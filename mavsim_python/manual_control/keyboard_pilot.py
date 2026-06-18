"""
Keyboard pilot for manual MAV control.
Uses pynput in a background thread; safe to call update() from the sim loop.
"""
from pynput import keyboard

from message_types.msg_delta import MsgDelta


class KeyboardPilot:
    """Map held keys to control-surface commands relative to trim."""

    def __init__(self, trim_delta=None):
        if trim_delta is None:
            trim_delta = MsgDelta(
                elevator=-0.1248,
                aileron=0.001836,
                rudder=-0.0003026,
                throttle=0.6768,
            )
        self._trim = trim_delta
        self.delta = MsgDelta(
            elevator=trim_delta.elevator,
            aileron=trim_delta.aileron,
            rudder=trim_delta.rudder,
            throttle=trim_delta.throttle,
        )

        self._pressed = set()
        self.quit_requested = False

        # radians / second and throttle / second
        self.elevator_rate = 0.35
        self.aileron_rate = 0.35
        self.rudder_rate = 0.25
        self.throttle_rate = 0.40

        self.surface_limit = 0.52
        self.throttle_min = 0.0
        self.throttle_max = 1.0

        self._listener = keyboard.Listener(
            on_press=self._on_press,
            on_release=self._on_release,
        )
        self._listener.start()

    def _key_id(self, key):
        try:
            if hasattr(key, "char") and key.char is not None:
                return key.char.lower()
        except AttributeError:
            pass
        return key

    def _on_press(self, key):
        kid = self._key_id(key)
        if key == keyboard.Key.esc:
            self.quit_requested = True
            return False
        if kid == "r":
            self.reset_to_trim()
            return
        self._pressed.add(kid)

    def _on_release(self, key):
        kid = self._key_id(key)
        self._pressed.discard(kid)

    def reset_to_trim(self):
        self.delta.elevator = self._trim.elevator
        self.delta.aileron = self._trim.aileron
        self.delta.rudder = self._trim.rudder
        self.delta.throttle = self._trim.throttle

    @staticmethod
    def _clamp(value, low, high):
        return max(low, min(high, value))

    def update(self, dt):
        """Advance controls according to keys held this frame."""
        if "w" in self._pressed:
            self.delta.elevator -= self.elevator_rate * dt
        if "s" in self._pressed:
            self.delta.elevator += self.elevator_rate * dt
        if "a" in self._pressed:
            self.delta.aileron -= self.aileron_rate * dt
        if "d" in self._pressed:
            self.delta.aileron += self.aileron_rate * dt
        if "q" in self._pressed:
            self.delta.rudder -= self.rudder_rate * dt
        if "e" in self._pressed:
            self.delta.rudder += self.rudder_rate * dt
        if keyboard.Key.up in self._pressed or "=" in self._pressed or "+" in self._pressed:
            self.delta.throttle += self.throttle_rate * dt
        if keyboard.Key.down in self._pressed or "-" in self._pressed:
            self.delta.throttle -= self.throttle_rate * dt

        self.delta.elevator = self._clamp(
            self.delta.elevator, -self.surface_limit, self.surface_limit)
        self.delta.aileron = self._clamp(
            self.delta.aileron, -self.surface_limit, self.surface_limit)
        self.delta.rudder = self._clamp(
            self.delta.rudder, -self.surface_limit, self.surface_limit)
        self.delta.throttle = self._clamp(
            self.delta.throttle, self.throttle_min, self.throttle_max)
        return self.delta

    def stop(self):
        self._listener.stop()

    @staticmethod
    def print_help():
        print(
            "\nManual flight controls (focus this terminal while flying):\n"
            "  W / S ........ pitch up / down (elevator)\n"
            "  A / D ........ roll left / right (aileron)\n"
            "  Q / E ........ yaw left / right (rudder)\n"
            "  Up / Down .... throttle up / down  (+ / - also work)\n"
            "  R ............ reset to trim\n"
            "  Esc .......... quit\n"
        )
