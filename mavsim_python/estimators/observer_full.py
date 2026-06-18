"""
observer
    - Beard & McLain, PUP, 2012
    - Last Update:
        3/4/2019 - RWB
        3/6/2024 - RWB
"""
import numpy as np
from scipy import stats
import parameters.control_parameters as CTRL
import parameters.simulation_parameters as SIM
import parameters.sensor_parameters as SENSOR
import parameters.aerosonde_parameters as MAV
from tools.rotations import euler_to_rotation
from tools.wrap import wrap
from message_types.msg_state import MsgState
from message_types.msg_sensors import MsgSensors
from estimators.filters import AlphaFilter, ExtendedKalmanFilterContinuousDiscrete


class Observer:
    def __init__(self, ts):
        # initialized estimated state message
        tau_gyro = 1.0
        alpha_gyro = SIM.ts_control / (SIM.ts_control + tau_gyro)
        self.ekf = ExtendedKalmanFilterContinuousDiscrete(
            f=self.f,
            Q=np.diag([
                (0.1) ** 2, (0.1) ** 2, (0.1) ** 2,
                (1.0) ** 2, (1.0) ** 2, (1.0) ** 2,
                np.radians(1.0) ** 2, np.radians(1.0) ** 2, np.radians(1.0) ** 2,
                np.radians(0.01) ** 2, np.radians(0.01) ** 2, np.radians(0.01) ** 2,
                (0.1) ** 2, (0.1) ** 2,
            ]),
            P0=np.diag([
                (10.) ** 2, (10.) ** 2, (10.) ** 2,
                (10.) ** 2, (10.) ** 2, (10.) ** 2,
                np.radians(30.) ** 2, np.radians(30.) ** 2, np.radians(30.) ** 2,
                np.radians(0.01) ** 2, np.radians(0.01) ** 2, np.radians(0.01) ** 2,
                (5.) ** 2, (5.) ** 2,
            ]),
            xhat0=np.array([[
                MAV.north0,
                MAV.east0,
                MAV.down0,
                MAV.Va0,
                0,
                0,
                0,
                0,
                MAV.psi0,
                0,
                0,
                0,
                0,
                0,
            ]]).T,
            Qu=np.diag([
                SENSOR.gyro_sigma ** 2,
                SENSOR.gyro_sigma ** 2,
                SENSOR.gyro_sigma ** 2,
                SENSOR.accel_sigma ** 2,
                SENSOR.accel_sigma ** 2,
                SENSOR.accel_sigma ** 2,
            ]),
            Ts=ts,
            N=10,
        )
        self.R_analog = np.diag([
            SENSOR.abs_pres_sigma ** 2,
            SENSOR.diff_pres_sigma ** 2,
            (0.01) ** 2,
        ])
        self.R_gps = np.diag([
            SENSOR.gps_n_sigma ** 2,
            SENSOR.gps_e_sigma ** 2,
            SENSOR.gps_Vg_sigma ** 2,
            SENSOR.gps_course_sigma ** 2,
        ])
        self.R_pseudo = np.diag([
            (1.0) ** 2,
            (1.0) ** 2,
        ])
        initial_measurements = MsgSensors()
        self.lpf_gyro_x = AlphaFilter(alpha=alpha_gyro, y0=initial_measurements.gyro_x)
        self.lpf_gyro_y = AlphaFilter(alpha=alpha_gyro, y0=initial_measurements.gyro_y)
        self.lpf_gyro_z = AlphaFilter(alpha=alpha_gyro, y0=initial_measurements.gyro_z)
        self.analog_threshold = stats.chi2.isf(q=0.01, df=3)
        self.pseudo_threshold = stats.chi2.isf(q=0.01, df=2)
        self.gps_n_old = 9999
        self.gps_e_old = 9999
        self.gps_Vg_old = 9999
        self.gps_course_old = 9999
        self.estimated_state = MsgState()
        self.elapsed_time = 0

    def update(self, measurement):
        # system input
        u = np.array([[
            measurement.gyro_x, 
            measurement.gyro_y, 
            measurement.gyro_z,
            measurement.accel_x, 
            measurement.accel_y, 
            measurement.accel_z,
            ]]).T
        xhat, P = self.ekf.propagate_model(u)
        # update with analog measurement
        y_analog = np.array([
            [measurement.abs_pressure],
            [measurement.diff_pressure],
            [0.0], # sideslip pseudo measurement
            ])
        xhat, P = self.ekf.measurement_update(
            y=y_analog, 
            u=u,
            h=self.h_analog,
            R=self.R_analog)
        # update with wind triangle pseudo measurement
        y_pseudo = np.array([
            [0.],
            [0.], 
            ])
        xhat, P = self.ekf.measurement_update(
            y=y_pseudo, 
            u=u,
            h=self.h_pseudo,
            R=self.R_pseudo)
        # only update GPS when one of the signals changes
        if (measurement.gps_n != self.gps_n_old) \
            or (measurement.gps_e != self.gps_e_old) \
            or (measurement.gps_Vg != self.gps_Vg_old) \
            or (measurement.gps_course != self.gps_course_old):
            state = to_MsgState(xhat) 
                # need to do this to get the current chi to wrap meaurement
            y_chi = wrap(measurement.gps_course, state.chi)
            y_gps = np.array([
                [measurement.gps_n], 
                [measurement.gps_e], 
                [measurement.gps_Vg], 
                [y_chi]])
            xhat, P = self.ekf.measurement_update(
                y=y_gps, 
                u=u,
                h=self.h_gps,
                R=self.R_gps)
            # update stored GPS signals
            self.gps_n_old = measurement.gps_n
            self.gps_e_old = measurement.gps_e
            self.gps_Vg_old = measurement.gps_Vg
            self.gps_course_old = measurement.gps_course
        # convert internal xhat to MsgState format
        self.estimated_state = to_MsgState(xhat)
        self.estimated_state.p = self.lpf_gyro_x.update(measurement.gyro_x) \
            - self.estimated_state.bx
        self.estimated_state.q = self.lpf_gyro_y.update(measurement.gyro_y) \
            - self.estimated_state.by
        self.estimated_state.r = self.lpf_gyro_z.update(measurement.gyro_z) \
            - self.estimated_state.bz
        self.elapsed_time += SIM.ts_control
        return self.estimated_state

    def f(self, x:np.ndarray, u:np.ndarray)->np.ndarray:
        # system dynamics for propagation model: xdot = f(x, u)
        vel = x[3:6]
        Theta = x[6:9]
        bias = x[9:12]
        y_gyro = u[0:3]
        y_accel = u[3:6]

        R = euler_to_rotation(Theta.item(0), Theta.item(1), Theta.item(2))
        pos_dot = R @ vel

        omega = y_gyro - bias
        vel_dot = (cross(omega) @ vel
                   + MAV.gravity * (R.T @ np.array([[0.], [0.], [1.]]) + y_accel))

        Theta_dot = S(Theta) @ omega
        bias_dot = np.zeros((3, 1))
        wind_dot = np.zeros((2, 1))
        xdot = np.concatenate((pos_dot, vel_dot, Theta_dot, bias_dot, wind_dot), axis=0)
        return xdot

    def h_analog(self, x:np.ndarray, u:np.ndarray)->np.ndarray:
        # analog sensor measurements and pseudo measurements
        pos = x[0:3]
        vel = x[3:6]
        Theta = x[6:9]

        R = euler_to_rotation(Theta.item(0), Theta.item(1), Theta.item(2))
        wind_world = np.array([[x.item(12)], [x.item(13)], [0]])
        wind_body = R.T @ wind_world
        vel_rel = vel - wind_body
        Va = np.linalg.norm(vel_rel)

        altitude = -pos.item(2)
        abs_pres = MAV.rho * MAV.gravity * altitude
        diff_pres = 0.5 * MAV.rho * Va ** 2
        if Va > 1e-6:
            sideslip = vel_rel.item(1) / Va
        else:
            sideslip = 0.0

        y = np.array([[abs_pres], [diff_pres], [sideslip]])
        return y

    def h_gps(self, x:np.ndarray, u:np.ndarray)->np.ndarray:
        # measurement model for gps measurements
        pos = x[0:3]
        vel = x[3:6]
        Theta = x[6:9]

        R = euler_to_rotation(Theta.item(0), Theta.item(1), Theta.item(2))
        vel_world = R @ vel
        pn = pos.item(0)
        pe = pos.item(1)
        Vg = np.sqrt(vel_world.item(0) ** 2 + vel_world.item(1) ** 2)
        chi = np.arctan2(vel_world.item(1), vel_world.item(0))

        y = np.array([[pn], [pe], [Vg], [chi]])
        return y

    def h_pseudo(self, x:np.ndarray, u:np.ndarray)->np.ndarray:
        # measurement model for wind triangle pseudo measurement
        vel = x[3:6]
        Theta = x[6:9]

        R = euler_to_rotation(Theta.item(0), Theta.item(1), Theta.item(2))
        vel_world = R @ vel
        Vg = np.sqrt(vel_world.item(0) ** 2 + vel_world.item(1) ** 2)
        chi = np.arctan2(vel_world.item(1), vel_world.item(0))
        wn = x.item(12)
        we = x.item(13)
        psi = Theta.item(2)
        u_body = vel.item(0)
        v_body = vel.item(1)

        y = np.array([
            [Vg * np.cos(chi) - (u_body * np.cos(psi) - v_body * np.sin(psi)) - wn],
            [Vg * np.sin(chi) - (u_body * np.sin(psi) + v_body * np.cos(psi)) - we],
        ])
        return y


def to_MsgState(x: np.ndarray) -> MsgState:
    state = MsgState()
    state.north = x.item(0)
    state.east = x.item(1)
    state.altitude = -x.item(2)
    vel_body = x[3:6]
    state.phi = x.item(6)
    state.theta = x.item(7)
    state.psi = x.item(8)
    state.bx = x.item(9)
    state.by = x.item(10)
    state.bz = x.item(11)
    state.wn = x.item(12)
    state.we = x.item(13)
    # estimate needed quantities that are not part of state
    R = euler_to_rotation(
        state.phi,
        state.theta,
        state.psi)
    vel_world = R @ vel_body
    wind_world = np.array([[state.wn], [state.we], [0]])
    wind_body = R.T @ wind_world
    vel_rel = vel_body - wind_body
    state.Va = np.linalg.norm(vel_rel)
    state.alpha = np.arctan(vel_rel.item(2) / vel_rel.item(0))
    state.beta = np.arcsin(vel_rel.item(1) / state.Va)
    state.Vg = np.linalg.norm(vel_world)
    state.chi = np.arctan2(vel_world.item(1), vel_world.item(0))
    return state


def cross(vec: np.ndarray)->np.ndarray:
    return np.array([[0, -vec.item(2), vec.item(1)],
                     [vec.item(2), 0, -vec.item(0)],
                     [-vec.item(1), vec.item(0), 0]])


def S(Theta:np.ndarray)->np.ndarray:
    return np.array([[1,
                      np.sin(Theta.item(0)) * np.tan(Theta.item(1)),
                      np.cos(Theta.item(0)) * np.tan(Theta.item(1))],
                     [0,
                      np.cos(Theta.item(0)),
                      -np.sin(Theta.item(0))],
                     [0,
                      (np.sin(Theta.item(0)) / np.cos(Theta.item(1))),
                      (np.cos(Theta.item(0)) / np.cos(Theta.item(1)))]
                     ])
