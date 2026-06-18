"""
target geolocation algorithm
    - Beard & McLain, PUP, 2012
    - Updated:
        4/1/2022 - RWB
        4/6/2022 - RWB
        7/13/2023 - RWB
        4/7/2025 - TWM
"""
import numpy as np
import parameters.simulation_parameters as SIM
import parameters.camera_parameters as CAM
from tools.rotations import euler_to_rotation
from estimators.filters import ExtendedKalmanFilterContinuousDiscrete


# Note that state equations assume a constant-velocity model for the target
class Geolocation:
    def __init__(self, ts: float=0.01):
        self.ekf = ExtendedKalmanFilterContinuousDiscrete(
            f=self.f,
            Q=0.01 * np.diag([
                (1.) ** 2,
                (1.) ** 2,
                (1.) ** 2,
                (10.) ** 2,
                (10.) ** 2,
                (10.) ** 2,
                (3.) ** 2,
            ]),
            P0=0.1 * np.diag([
                10 ** 2,
                10 ** 2,
                10 ** 2,
                10 ** 2,
                10 ** 2,
                10 ** 2,
                10 ** 2,
            ]),
            xhat0=np.array([[
                0.,
                0.,
                0.,
                0.,
                0.,
                0.,
                100.,
            ]]).T,
            Qu=0.01 * np.diag([
                1 ** 2,
                1 ** 2,
                1 ** 2,
                1 ** 2,
                1 ** 2,
                1 ** 2,
            ]),
            Ts=ts,
            N=10,
        )
        self.R = 0.1 * np.diag([1.0, 1.0, 1.0, 1.0])

    def update(self, mav, pixels):
        R = euler_to_rotation(mav.phi, mav.theta, mav.psi)
        vel_ned = R @ np.array([[mav.Va], [0.], [0.]])
        u = np.array([
            [mav.north],
            [mav.east],
            [-mav.altitude],
            [vel_ned.item(0)],
            [vel_ned.item(1)],
            [vel_ned.item(2)],
        ])
        xhat, P = self.ekf.propagate_model(u)
        y = self.process_measurements(mav, pixels)
        xhat, P = self.ekf.measurement_update(
            y=y,
            u=u,
            h=self.h,
            R=self.R,
        )
        return xhat[0:3, :]

    def f(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        p_t = x[0:3]
        v_t = x[3:6]
        L = x.item(6)
        p_m = u[0:3]
        v_m = u[3:6]

        target_position_dot = v_t
        target_velocity_dot = np.zeros((3, 1))

        if abs(L) < 1e-3:
            L = 1e-3
        e = (p_t - p_m) / L
        L_dot = np.array([[float(e.T @ (v_t - v_m))]])

        xdot = np.concatenate((target_position_dot, target_velocity_dot, L_dot), axis=0)
        return xdot

    def h(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        target_position = x[0:3]
        L = x[6:7]
        y = np.concatenate((target_position, L), axis=0)
        return y

    def process_measurements(self, mav, pixels):
        mav_position = np.array([[mav.north], [mav.east], [-mav.altitude]])
        ell = np.array([[pixels.pixel_x], [pixels.pixel_y], [CAM.f]])
        ell_c = ell / np.linalg.norm(ell)

        R_b_i = euler_to_rotation(mav.phi, mav.theta, mav.psi)
        R_g_b = euler_to_rotation(0, mav.gimbal_el, mav.gimbal_az)
        R_c_g = np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]])
        R_i2c = R_c_g @ R_g_b.T @ R_b_i.T
        ell_i = R_i2c.T @ ell_c

        if abs(ell_i.item(2)) > 1e-6:
            L = -mav_position.item(2) / ell_i.item(2)
        else:
            L = self.ekf.xhat.item(6)

        target_position = mav_position + L * ell_i
        y = np.concatenate((target_position, np.array([[L]])), axis=0)
        return y
