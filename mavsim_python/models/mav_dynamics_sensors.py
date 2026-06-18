"""
mavDynamics 
    - this file implements the dynamic equations of motion for MAV
    - use unit quaternion for the attitude state
    
mavsim_python
    - Beard & McLain, PUP, 2012
    - Update history:  
        2/24/2020 - RWB
"""
import numpy as np
from message_types.msg_sensors import MsgSensors
import parameters.aerosonde_parameters as MAV
import parameters.sensor_parameters as SENSOR
from models.mav_dynamics_control import MavDynamics as MavDynamicsNoSensors
from tools.rotations import quaternion_to_rotation, quaternion_to_euler, euler_to_rotation

class MavDynamics(MavDynamicsNoSensors):
    def __init__(self, Ts):
        super().__init__(Ts)
        # initialize the sensors message
        self._sensors = MsgSensors()
        # random walk parameters for GPS
        self._gps_eta_n = 0.
        self._gps_eta_e = 0.
        self._gps_eta_h = 0.
        # timer so that gps only updates every ts_gps seconds
        self._t_gps = 999.  # large value ensures gps updates at initial time.

    def sensors(self):
        "Return value of sensors on MAV: gyros, accels, absolute_pressure, dynamic_pressure, GPS"

        phi, theta, psi = quaternion_to_euler(self._state[6:10])
        pdot = quaternion_to_rotation(self._state[6:10]) @ self._state[3:6]
        p = self._state.item(10)
        q = self._state.item(11)
        r = self._state.item(12)

        # simulate rate gyros (units are rad / sec)
        self._sensors.gyro_x = (p + SENSOR.gyro_x_bias
                                + np.random.normal(0, SENSOR.gyro_sigma))
        self._sensors.gyro_y = (q + SENSOR.gyro_y_bias
                                + np.random.normal(0, SENSOR.gyro_sigma))
        self._sensors.gyro_z = (r + SENSOR.gyro_z_bias
                                + np.random.normal(0, SENSOR.gyro_sigma))

        # simulate accelerometers (units of g)
        R = quaternion_to_rotation(self._state[6:10])
        f_body = self._forces / MAV.mass
        g_body = R.T @ np.array([[0.], [0.], [MAV.gravity]])
        accel_noise = SENSOR.accel_sigma / MAV.gravity
        self._sensors.accel_x = ((f_body.item(0) - g_body.item(0)) / MAV.gravity
                                 + np.random.normal(0, accel_noise))
        self._sensors.accel_y = ((f_body.item(1) - g_body.item(1)) / MAV.gravity
                                 + np.random.normal(0, accel_noise))
        self._sensors.accel_z = ((f_body.item(2) - g_body.item(2)) / MAV.gravity
                                 + np.random.normal(0, accel_noise))

        # simulate magnetometers
        decl = np.radians(12.5)
        incl = np.radians(66.0)
        mag_inertial = np.array([
            [np.cos(decl) * np.cos(incl)],
            [np.sin(decl) * np.cos(incl)],
            [np.sin(incl)],
        ])
        mag_body = R.T @ mag_inertial
        self._sensors.mag_x = mag_body.item(0) + np.random.normal(0, SENSOR.mag_sigma)
        self._sensors.mag_y = mag_body.item(1) + np.random.normal(0, SENSOR.mag_sigma)
        self._sensors.mag_z = mag_body.item(2) + np.random.normal(0, SENSOR.mag_sigma)

        # simulate pressure sensors
        altitude = -self._state.item(2)
        self._sensors.abs_pressure = (MAV.rho * MAV.gravity * altitude
                                        + np.random.normal(0, SENSOR.abs_pres_sigma))
        self._sensors.diff_pressure = (0.5 * MAV.rho * self._Va ** 2
                                       + np.random.normal(0, SENSOR.diff_pres_sigma))

        # simulate GPS sensor
        if self._t_gps >= SENSOR.ts_gps:
            self._gps_eta_n += np.random.normal(0, np.sqrt(SENSOR.gps_k * SENSOR.ts_gps))
            self._gps_eta_e += np.random.normal(0, np.sqrt(SENSOR.gps_k * SENSOR.ts_gps))
            self._gps_eta_h += np.random.normal(0, np.sqrt(SENSOR.gps_k * SENSOR.ts_gps))
            self._sensors.gps_n = (self._state.item(0) + self._gps_eta_n
                                   + np.random.normal(0, SENSOR.gps_n_sigma))
            self._sensors.gps_e = (self._state.item(1) + self._gps_eta_e
                                   + np.random.normal(0, SENSOR.gps_e_sigma))
            self._sensors.gps_h = (-self._state.item(2) + self._gps_eta_h
                                   + np.random.normal(0, SENSOR.gps_h_sigma))
            self._sensors.gps_Vg = (np.linalg.norm(pdot[0:2])
                                    + np.random.normal(0, SENSOR.gps_Vg_sigma))
            self._sensors.gps_course = (np.arctan2(pdot.item(1), pdot.item(0))
                                        + np.random.normal(0, SENSOR.gps_course_sigma))
            self._t_gps = 0.
        else:
            self._t_gps += self._ts_simulation
        return self._sensors

    def external_set_state(self, new_state):
        self._state = new_state

    def _update_true_state(self):
        # update the class structure for the true state:
        #   [pn, pe, h, Va, alpha, beta, phi, theta, chi, p, q, r, Vg, wn, we, psi, gyro_bx, gyro_by, gyro_bz]
        phi, theta, psi = quaternion_to_euler(self._state[6:10])
        pdot = quaternion_to_rotation(self._state[6:10]) @ self._state[3:6]
        self.true_state.north = self._state.item(0)
        self.true_state.east = self._state.item(1)
        self.true_state.altitude = -self._state.item(2)
        self.true_state.Va = self._Va
        self.true_state.alpha = self._alpha
        self.true_state.beta = self._beta
        self.true_state.phi = phi
        self.true_state.theta = theta
        self.true_state.psi = psi
        self.true_state.Vg = np.linalg.norm(pdot)
        self.true_state.gamma = np.arcsin(pdot.item(2) / self.true_state.Vg)
        self.true_state.chi = np.arctan2(pdot.item(1), pdot.item(0))
        self.true_state.p = self._state.item(10)
        self.true_state.q = self._state.item(11)
        self.true_state.r = self._state.item(12)
        self.true_state.wn = self._wind.item(0)
        self.true_state.we = self._wind.item(1)
        self.true_state.bx = SENSOR.gyro_x_bias
        self.true_state.by = SENSOR.gyro_y_bias
        self.true_state.bz = SENSOR.gyro_z_bias
        self.true_state.gimbal_az = self._state.item(13)
        self.true_state.gimbal_el = self._state.item(14)