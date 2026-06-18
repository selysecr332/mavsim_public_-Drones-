"""
observer
    - Beard & McLain, PUP, 2012
    - Last Update:
        3/2/2019 - RWB
        3/4/2024 - RWB
"""
import numpy as np
import parameters.simulation_parameters as SIM
import parameters.sensor_parameters as SENSOR
import parameters.aerosonde_parameters as MAV
from tools.wrap import wrap
from message_types.msg_state import MsgState
from message_types.msg_sensors import MsgSensors
from estimators.filters import AlphaFilter, ExtendedKalmanFilterContinuousDiscrete


class Observer:
    def __init__(self, ts: float, initial_measurements: MsgSensors=MsgSensors()):
        self.Ts = ts  # sample rate of observer
        # initialized estimated state message
        self.estimated_state = MsgState()

        tau_gyro = 0.02
        tau_accel = 0.02
        tau_abs = 0.1
        tau_diff = 0.02
        alpha_gyro = SIM.ts_control / (SIM.ts_control + tau_gyro)
        alpha_accel = SIM.ts_control / (SIM.ts_control + tau_accel)
        alpha_abs = SIM.ts_control / (SIM.ts_control + tau_abs)
        alpha_diff = SIM.ts_control / (SIM.ts_control + tau_diff)

        self.lpf_gyro_x = AlphaFilter(alpha=alpha_gyro, y0=initial_measurements.gyro_x)
        self.lpf_gyro_y = AlphaFilter(alpha=alpha_gyro, y0=initial_measurements.gyro_y)
        self.lpf_gyro_z = AlphaFilter(alpha=alpha_gyro, y0=initial_measurements.gyro_z)
        self.lpf_accel_x = AlphaFilter(alpha=alpha_accel, y0=initial_measurements.accel_x)
        self.lpf_accel_y = AlphaFilter(alpha=alpha_accel, y0=initial_measurements.accel_y)
        self.lpf_accel_z = AlphaFilter(alpha=alpha_accel, y0=initial_measurements.accel_z)
        self.lpf_abs = AlphaFilter(alpha=alpha_abs, y0=initial_measurements.abs_pressure)
        self.lpf_diff = AlphaFilter(alpha=alpha_diff, y0=initial_measurements.diff_pressure)

        self.attitude_ekf = ExtendedKalmanFilterContinuousDiscrete(
            f=self.f_attitude,
            Q=np.diag([
                np.radians(0.5) ** 2,
                np.radians(0.5) ** 2,
            ]),
            P0=np.diag([
                np.radians(30.) ** 2,
                np.radians(30.) ** 2,
            ]),
            xhat0=np.array([
                [MAV.phi0],
                [MAV.theta0],
            ]),
            Qu=np.diag([
                SENSOR.gyro_sigma ** 2,
                SENSOR.gyro_sigma ** 2,
                SENSOR.gyro_sigma ** 2,
                SENSOR.diff_pres_sigma ** 2,
            ]),
            Ts=ts,
            N=5,
        )

        self.position_ekf = ExtendedKalmanFilterContinuousDiscrete(
            f=self.f_smooth,
            Q=np.diag([
                (0.1) ** 2,
                (0.1) ** 2,
                (0.1) ** 2,
                np.radians(0.5) ** 2,
                (0.01) ** 2,
                (0.01) ** 2,
                np.radians(0.5) ** 2,
            ]),
            P0=np.diag([
                (10.) ** 2,
                (10.) ** 2,
                (10.) ** 2,
                np.radians(45.) ** 2,
                (5.) ** 2,
                (5.) ** 2,
                np.radians(45.) ** 2,
            ]),
            xhat0=np.array([
                [MAV.north0],
                [MAV.east0],
                [MAV.Va0],
                [0.0],
                [0.0],
                [0.0],
                [MAV.psi0],
            ]),
            Qu=np.diag([
                SENSOR.gyro_sigma ** 2,
                SENSOR.gyro_sigma ** 2,
                SENSOR.diff_pres_sigma ** 2,
                np.radians(1.0) ** 2,
                np.radians(1.0) ** 2,
            ]),
            Ts=ts,
            N=10,
        )

        self.R_accel = np.diag([
            SENSOR.accel_sigma ** 2,
            SENSOR.accel_sigma ** 2,
            SENSOR.accel_sigma ** 2,
        ])
        self.R_pseudo = np.diag([
            (1.0) ** 2,
            (1.0) ** 2,
        ])
        self.R_gps = np.diag([
            SENSOR.gps_n_sigma ** 2,
            SENSOR.gps_e_sigma ** 2,
            SENSOR.gps_Vg_sigma ** 2,
            SENSOR.gps_course_sigma ** 2,
        ])
        self.gps_n_old = 9999
        self.gps_e_old = 9999
        self.gps_Vg_old = 9999
        self.gps_course_old = 9999

    def update(self, measurement: MsgSensors) -> MsgState:
        # estimates for p, q, r are low pass filter of gyro minus bias estimate
        self.estimated_state.p = self.lpf_gyro_x.update(measurement.gyro_x)
        self.estimated_state.q = self.lpf_gyro_y.update(measurement.gyro_y)
        self.estimated_state.r = self.lpf_gyro_z.update(measurement.gyro_z)

        # invert sensor model to get altitude and airspeed
        abs_pressure = self.lpf_abs.update(measurement.abs_pressure)
        diff_pressure = self.lpf_diff.update(measurement.diff_pressure)
        self.estimated_state.altitude = abs_pressure / (MAV.rho * MAV.gravity)
        self.estimated_state.Va = np.sqrt(max(2.0 * diff_pressure / MAV.rho, 0.0))

        # estimate phi and theta with ekf
        u_attitude = np.array([
            [self.estimated_state.p],
            [self.estimated_state.q],
            [self.estimated_state.r],
            [self.estimated_state.Va],
        ])
        xhat_attitude, P_attitude = self.attitude_ekf.propagate_model(u_attitude)
        y_accel = np.array([
            [measurement.accel_x],
            [measurement.accel_y],
            [measurement.accel_z],
        ])
        xhat_attitude, P_attitude = self.attitude_ekf.measurement_update(
            y=y_accel,
            u=u_attitude,
            h=self.h_accel,
            R=self.R_accel,
        )
        self.estimated_state.phi = xhat_attitude.item(0)
        self.estimated_state.theta = xhat_attitude.item(1)

        # estimate pn, pe, Vg, chi, wn, we, psi with ekf
        u_smooth = np.array([
            [self.estimated_state.q],
            [self.estimated_state.r],
            [self.estimated_state.Va],
            [self.estimated_state.phi],
            [self.estimated_state.theta],
        ])
        xhat_position, P_position = self.position_ekf.propagate_model(u_smooth)
        y_pseudo = np.array([[0.], [0.]])
        xhat_position, P_position = self.position_ekf.measurement_update(
            y=y_pseudo,
            u=u_smooth,
            h=self.h_pseudo,
            R=self.R_pseudo,
        )

        # only update GPS when one of the signals changes
        if (measurement.gps_n != self.gps_n_old) \
            or (measurement.gps_e != self.gps_e_old) \
            or (measurement.gps_Vg != self.gps_Vg_old) \
            or (measurement.gps_course != self.gps_course_old):
            y_gps = np.array([
                [measurement.gps_n],
                [measurement.gps_e],
                [measurement.gps_Vg],
                [wrap(measurement.gps_course, xhat_position.item(3))],
            ])
            xhat_position, P_position = self.position_ekf.measurement_update(
                y=y_gps,
                u=u_smooth,
                h=self.h_gps,
                R=self.R_gps,
            )
            self.gps_n_old = measurement.gps_n
            self.gps_e_old = measurement.gps_e
            self.gps_Vg_old = measurement.gps_Vg
            self.gps_course_old = measurement.gps_course

        self.estimated_state.north = xhat_position.item(0)
        self.estimated_state.east = xhat_position.item(1)
        self.estimated_state.Vg = xhat_position.item(2)
        self.estimated_state.chi = xhat_position.item(3)
        self.estimated_state.wn = xhat_position.item(4)
        self.estimated_state.we = xhat_position.item(5)
        self.estimated_state.psi = xhat_position.item(6)

        self.estimated_state.alpha = self.estimated_state.theta
        self.estimated_state.beta = 0.0
        self.estimated_state.bx = 0.0
        self.estimated_state.by = 0.0
        self.estimated_state.bz = 0.0
        return self.estimated_state

    def f_attitude(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        '''
            system dynamics for propagation model: xdot = f(x, u)
                x = [phi, theta].T
                u = [p, q, r, Va].T
        '''
        phi = x.item(0)
        theta = x.item(1)
        p = u.item(0)
        q = u.item(1)
        r = u.item(2)

        phi_dot = p + np.sin(phi) * np.tan(theta) * q + np.cos(phi) * np.tan(theta) * r
        theta_dot = np.cos(phi) * q - np.sin(phi) * r
        xdot = np.array([[phi_dot], [theta_dot]])
        return xdot

    def h_accel(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        '''
            measurement model y=h(x,u) for accelerometers
                x = [phi, theta].T
                u = [p, q, r, Va].T
        '''
        phi = x.item(0)
        theta = x.item(1)
        p = u.item(0)
        q = u.item(1)
        Va = u.item(3)
        g = MAV.gravity

        ax = np.sin(theta) * np.cos(phi) + q * Va * np.sin(theta) / g
        ay = -np.cos(theta) * np.sin(phi) - p * Va / g
        az = -np.cos(theta) * np.cos(phi) - q * Va * np.cos(theta) / g
        y = np.array([[ax], [ay], [az]])
        return y

    def f_smooth(self, x, u):
        '''
            system dynamics for propagation model: xdot = f(x, u)
                x = [pn, pe, Vg, chi, wn, we, psi].T
                u = [q, r, Va, phi, theta].T
        '''
        Vg = x.item(2)
        chi = x.item(3)
        r = u.item(1)
        Va = u.item(2)
        phi = u.item(3)

        pn_dot = Vg * np.cos(chi)
        pe_dot = Vg * np.sin(chi)
        Vg_dot = 0.0
        chi_dot = MAV.gravity / Va * np.tan(phi)
        wn_dot = 0.0
        we_dot = 0.0
        psi_dot = r
        xdot = np.array([[pn_dot], [pe_dot], [Vg_dot], [chi_dot],
                         [wn_dot], [we_dot], [psi_dot]])
        return xdot

    def h_pseudo(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        '''
            measurement model for wind triangle pseudo measurement: y=y(x, u)
                x = [pn, pe, Vg, chi, wn, we, psi].T
                u = [q, r, Va, phi, theta].T
        '''
        Vg = x.item(2)
        chi = x.item(3)
        wn = x.item(4)
        we = x.item(5)
        psi = x.item(6)
        Va = u.item(2)
        theta = u.item(4)

        u_body = Va * np.cos(theta)
        y = np.array([
            [Vg * np.cos(chi) - u_body * np.cos(psi) - wn],
            [Vg * np.sin(chi) - u_body * np.sin(psi) - we],
        ])
        return y

    def h_gps(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        '''
            measurement model for gps measurements: y=y(x, u)
                x = [pn, pe, Vg, chi, wn, we, psi].T
            returns
                y = [pn, pe, Vg, chi]
        '''
        y = np.array([
            [x.item(0)],
            [x.item(1)],
            [x.item(2)],
            [x.item(3)],
        ])
        return y
