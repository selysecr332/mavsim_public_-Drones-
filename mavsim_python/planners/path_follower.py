import numpy as np
from math import sin, cos
from message_types.msg_state import MsgState
from message_types.msg_path import MsgPath
from message_types.msg_autopilot import MsgAutopilot
from tools.wrap import wrap
import parameters.aerosonde_parameters as MAV


class PathFollower:
    def __init__(self):
        self.chi_inf = np.radians(45.0)  # approach angle for large distance from straight-line path
        self.k_path = 0.05  # path gain for straight-line path following
        self.k_orbit = 10.0  # path gain for orbit following
        self.gravity = MAV.gravity
        self.autopilot_commands = MsgAutopilot()  # message sent to autopilot

    def update(self,
               path: MsgPath,
               state: MsgState) -> MsgAutopilot:
        if path.type == 'line':
            self._follow_straight_line(path, state)
        elif path.type == 'orbit':
            self._follow_orbit(path, state)
        elif path.type == 'helix':
            self._follow_orbit(path, state)
        return self.autopilot_commands

    def _follow_straight_line(self,
                              path: MsgPath,
                              state: MsgState):
        r = path.line_origin
        q = path.line_direction
        p = np.array([[state.north], [state.east], [-state.altitude]])

        e_py = np.cross(q.flatten(), (p - r).flatten()).reshape(3, 1)

        chi_q = np.arctan2(q.item(1), q.item(0))
        chi_r = np.arctan2(-e_py.item(0), e_py.item(1))
        chi_r = wrap(chi_r, state.chi)

        chi_c = chi_q - np.arctan2(self.k_path * e_py.item(2), self.chi_inf)

        s = float(np.dot((p - r).flatten(), q.flatten()))
        h_c = -(r.item(2) + s * q.item(2))

        self.autopilot_commands.airspeed_command = path.airspeed
        self.autopilot_commands.course_command = chi_c
        self.autopilot_commands.altitude_command = h_c
        self.autopilot_commands.phi_feedforward = 0.0

    def _follow_orbit(self,
                      path: MsgPath,
                      state: MsgState):
        if path.orbit_direction == 'CW':
            lam = 1.0
        else:
            lam = -1.0

        c = path.orbit_center
        rho = path.orbit_radius
        pn = state.north
        pe = state.east

        d = np.sqrt((pn - c.item(0)) ** 2 + (pe - c.item(1)) ** 2)
        varphi = np.arctan2(pe - c.item(1), pn - c.item(0))
        varphi = wrap(varphi, state.chi)

        orbit_error = d - rho
        chi_c = varphi + lam * (np.pi / 2 + np.arctan2(self.k_orbit * orbit_error, np.pi / 2))

        h_c = -c.item(2)

        if abs(orbit_error) < 10.0:
            phi_ff = lam * np.arctan(path.airspeed ** 2 / (self.gravity * rho))
        else:
            phi_ff = 0.0

        self.autopilot_commands.airspeed_command = path.airspeed
        self.autopilot_commands.course_command = chi_c
        self.autopilot_commands.altitude_command = h_c
        self.autopilot_commands.phi_feedforward = phi_ff
