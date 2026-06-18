"""
mavsim_python: path manager
    - Beard & McLain, PUP, 2012
"""

import numpy as np
from planners.dubins_parameters import DubinsParameters
from message_types.msg_state import MsgState
from message_types.msg_path import MsgPath
from message_types.msg_waypoints import MsgWaypoints


class PathManager:
    def __init__(self):
        self._path = MsgPath()
        self._num_waypoints = 0
        self._ptr_previous = 0
        self._ptr_current = 1
        self._ptr_next = 2
        self._halfspace_n = np.inf * np.ones((3, 1))
        self._halfspace_r = np.inf * np.ones((3, 1))
        self._manager_state = 1
        self.manager_requests_waypoints = True
        self.dubins_path = DubinsParameters()

    def update(self,
               waypoints: MsgWaypoints,
               radius: float,
               state: MsgState) -> MsgPath:
        if waypoints.num_waypoints == 0:
            self.manager_requests_waypoints = True
        if self.manager_requests_waypoints is True \
                and waypoints.flag_waypoints_changed is True:
            self.manager_requests_waypoints = False
        if waypoints.type == 'straight_line':
            self._line_manager(waypoints, state)
        elif waypoints.type == 'fillet':
            self._fillet_manager(waypoints, radius, state)
        elif waypoints.type == 'dubins':
            self._dubins_manager(waypoints, radius, state)
        else:
            print('Error in Path Manager: Undefined waypoint type.')
        return self._path

    def _line_manager(self,
                      waypoints: MsgWaypoints,
                      state: MsgState):
        if waypoints.num_waypoints < 3:
            return

        mav_pos = np.array([[state.north], [state.east], [-state.altitude]])

        if waypoints.flag_waypoints_changed:
            self._num_waypoints = waypoints.num_waypoints
            waypoints.flag_waypoints_changed = False
            self._initialize_pointers()
            self._construct_line(waypoints)
            return

        if self._inHalfSpace(mav_pos):
            self._increment_pointers()
            if self._ptr_next >= self._num_waypoints:
                self.manager_requests_waypoints = True
            else:
                self._construct_line(waypoints)

    def _fillet_manager(self,
                        waypoints: MsgWaypoints,
                        radius: float,
                        state: MsgState):
        if waypoints.num_waypoints < 3:
            return

        mav_pos = np.array([[state.north], [state.east], [-state.altitude]])

        if waypoints.flag_waypoints_changed:
            self._num_waypoints = waypoints.num_waypoints
            waypoints.flag_waypoints_changed = False
            self._initialize_pointers()
            self._manager_state = 1
            self._construct_fillet_line(waypoints, radius)
            return

        if self._manager_state == 1:
            if self._inHalfSpace(mav_pos):
                self._manager_state = 2
                self._construct_fillet_circle(waypoints, radius)
        elif self._manager_state == 2:
            if self._inHalfSpace(mav_pos):
                self._increment_pointers()
                if self._ptr_next >= self._num_waypoints:
                    self.manager_requests_waypoints = True
                else:
                    self._manager_state = 1
                    self._construct_fillet_line(waypoints, radius)

    def _dubins_manager(self,
                        waypoints: MsgWaypoints,
                        radius: float,
                        state: MsgState):
        if waypoints.num_waypoints < 3:
            return

        mav_pos = np.array([[state.north], [state.east], [-state.altitude]])

        if waypoints.flag_waypoints_changed:
            self._num_waypoints = waypoints.num_waypoints
            waypoints.flag_waypoints_changed = False
            self._initialize_pointers()
            self._manager_state = 1
            self._setup_dubins_segment(waypoints, radius)
            self._construct_dubins_circle_start(waypoints, self.dubins_path)
            return

        if self._manager_state == 1:
            self._construct_dubins_circle_start(waypoints, self.dubins_path)
            if self._inHalfSpace(mav_pos):
                self._manager_state = 2
        elif self._manager_state == 2:
            self._construct_dubins_line(waypoints, self.dubins_path)
            if self._inHalfSpace(mav_pos):
                self._manager_state = 3
        elif self._manager_state == 3:
            self._construct_dubins_circle_end(waypoints, self.dubins_path)
            if self._inHalfSpace(mav_pos):
                self._increment_pointers()
                if self._ptr_next >= self._num_waypoints:
                    self.manager_requests_waypoints = True
                else:
                    self._manager_state = 1
                    self._setup_dubins_segment(waypoints, radius)

    def _setup_dubins_segment(self, waypoints: MsgWaypoints, radius: float):
        ps = waypoints.ned[:, self._ptr_previous:self._ptr_previous + 1]
        pe = waypoints.ned[:, self._ptr_current:self._ptr_current + 1]
        chis = waypoints.course[self._ptr_previous]
        chie = waypoints.course[self._ptr_current]
        self.dubins_path.update(ps, chis, pe, chie, radius)

    def _initialize_pointers(self):
        if self._num_waypoints >= 3:
            self._ptr_previous = 0
            self._ptr_current = 1
            self._ptr_next = 2
        else:
            print('Error Path Manager: need at least three waypoints')

    def _increment_pointers(self):
        self._ptr_previous += 1
        self._ptr_current += 1
        self._ptr_next += 1

    def _construct_line(self,
                        waypoints: MsgWaypoints):
        if (waypoints.num_waypoints == 0
                or self._ptr_current >= waypoints.num_waypoints):
            return

        previous = waypoints.ned[:, self._ptr_previous:self._ptr_previous + 1]
        current = waypoints.ned[:, self._ptr_current:self._ptr_current + 1]
        q = current - previous
        q = q / np.linalg.norm(q)

        self._halfspace_n = q
        self._halfspace_r = current
        self._path.type = 'line'
        self._path.line_origin = previous
        self._path.line_direction = q
        self._path.airspeed = waypoints.airspeed[self._ptr_current]
        self._path.plot_updated = False

    def _construct_fillet_line(self,
                               waypoints: MsgWaypoints,
                               radius: float):
        previous = waypoints.ned[:, self._ptr_previous:self._ptr_previous + 1]
        current = waypoints.ned[:, self._ptr_current:self._ptr_current + 1]
        next_wp = waypoints.ned[:, self._ptr_next:self._ptr_next + 1]

        q1 = current - previous
        q1 = q1 / np.linalg.norm(q1)
        q2 = next_wp - current
        q2 = q2 / np.linalg.norm(q2)

        angle = np.arccos(np.clip(float(q1.T @ q2), -1., 1.))
        if np.sin(angle) < 1e-6:
            self._construct_line(waypoints)
            return

        d = radius / np.tan(angle / 2.)
        r = current - d * q1

        self._halfspace_n = q1
        self._halfspace_r = r
        self._path.type = 'line'
        self._path.line_origin = previous
        self._path.line_direction = q1
        self._path.airspeed = waypoints.airspeed[self._ptr_current]
        self._path.plot_updated = False

    def _construct_fillet_circle(self,
                                 waypoints: MsgWaypoints,
                                 radius: float):
        previous = waypoints.ned[:, self._ptr_previous:self._ptr_previous + 1]
        current = waypoints.ned[:, self._ptr_current:self._ptr_current + 1]
        next_wp = waypoints.ned[:, self._ptr_next:self._ptr_next + 1]

        q1 = current - previous
        q1 = q1 / np.linalg.norm(q1)
        q2 = next_wp - current
        q2 = q2 / np.linalg.norm(q2)

        angle = np.arccos(np.clip(float(q1.T @ q2), -1., 1.))
        d = radius / np.tan(angle / 2.)
        r = current - d * q1
        r2 = current + d * q2

        q1_flat = q1.flatten()
        turn = np.sign(q1_flat[0] * q2.item(1) - q1_flat[1] * q2.item(0))
        n_perp = np.array([[-q1_flat[1]], [q1_flat[0]], [0.]])
        n_perp = n_perp / np.linalg.norm(n_perp)
        center = r + turn * radius * n_perp

        self._halfspace_n = q2
        self._halfspace_r = r2
        self._path.type = 'orbit'
        self._path.orbit_center = center
        self._path.orbit_radius = radius
        self._path.orbit_direction = 'CCW' if turn > 0 else 'CW'
        self._path.airspeed = waypoints.airspeed[self._ptr_current]
        self._path.plot_updated = False

    def _construct_dubins_circle_start(self,
                                       waypoints: MsgWaypoints,
                                       dubins_path: DubinsParameters):
        self._halfspace_n = dubins_path.n1
        self._halfspace_r = dubins_path.r1
        self._path.type = 'orbit'
        self._path.orbit_center = dubins_path.center_s
        self._path.orbit_radius = dubins_path.radius
        self._path.orbit_direction = 'CW' if dubins_path.dir_s > 0 else 'CCW'
        self._path.airspeed = waypoints.airspeed[self._ptr_current]
        self._path.plot_updated = False

    def _construct_dubins_line(self,
                               waypoints: MsgWaypoints,
                               dubins_path: DubinsParameters):
        self._halfspace_n = dubins_path.n1
        self._halfspace_r = dubins_path.r2
        self._path.type = 'line'
        self._path.line_origin = dubins_path.r1
        self._path.line_direction = dubins_path.n1
        self._path.airspeed = waypoints.airspeed[self._ptr_current]
        self._path.plot_updated = False

    def _construct_dubins_circle_end(self,
                                     waypoints: MsgWaypoints,
                                     dubins_path: DubinsParameters):
        self._halfspace_n = dubins_path.n3
        self._halfspace_r = dubins_path.r3
        self._path.type = 'orbit'
        self._path.orbit_center = dubins_path.center_e
        self._path.orbit_radius = dubins_path.radius
        self._path.orbit_direction = 'CW' if dubins_path.dir_e > 0 else 'CCW'
        self._path.airspeed = waypoints.airspeed[self._ptr_current]
        self._path.plot_updated = False

    def _inHalfSpace(self,
                     pos: np.ndarray) -> bool:
        '''Is pos in the half space defined by r and n?'''
        if (pos - self._halfspace_r).T @ self._halfspace_n >= 0:
            return True
        return False
