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
from scipy.optimize import fsolve
from models.mav_dynamics import MavDynamics as MavDynamicsForces
# load message types
from message_types.msg_state import MsgState
from message_types.msg_delta import MsgDelta
import parameters.aerosonde_parameters as MAV
from tools.rotations import quaternion_to_rotation, quaternion_to_euler


class MavDynamics(MavDynamicsForces):
    def __init__(self, Ts):
        super().__init__(Ts)
        # store wind data for fast recall since it is used at various points in simulation
        self._wind = np.array([[0.], [0.], [0.]])  # wind in NED frame in meters/sec
        # store forces to avoid recalculation in the sensors function
        self._forces = np.array([[0.], [0.], [0.]])
        self._Va = MAV.u0
        self._alpha = 0
        self._beta = 0
        # update velocity data and forces and moments
        self._update_velocity_data()
        self._forces_moments(delta=MsgDelta())
        # update the message class for the true state
        self._update_true_state()


    ###################################
    # public functions
    def update(self, delta, wind):
        '''
            Integrate the differential equations defining dynamics, update sensors
            delta = (delta_a, delta_e, delta_r, delta_t) are the control inputs
            wind is the wind vector in inertial coordinates
            Ts is the time step between function calls.
        '''
        # get forces and moments acting on rigid bod
        forces_moments = self._forces_moments(delta)
        super()._rk4_step(forces_moments)
        # update the airspeed, angle of attack, and side slip angles using new state
        self._update_velocity_data(wind)
        # update the message class for the true state
        self._update_true_state()

    ###################################
    # private functions
    def _update_velocity_data(self, wind=np.zeros((6,1))):
        steady_state = wind[0:3]
        gust = wind[3:6]

        R_bi = quaternion_to_rotation(self._state[6:10])
        wind_body = R_bi.T @ steady_state
        wind_body += gust
        self._wind = R_bi @ wind_body

        u = self._state.item(3)
        v = self._state.item(4)
        w = self._state.item(5)
        ur = u - wind_body.item(0)
        vr = v - wind_body.item(1)
        wr = w - wind_body.item(2)

        self._Va = np.sqrt(ur ** 2 + vr ** 2 + wr ** 2)
        if ur == 0:
            self._alpha = np.sign(wr) * np.pi / 2
        else:
            self._alpha = np.arctan2(wr, ur)
        if self._Va == 0:
            self._beta = 0
        else:
            self._beta = np.arcsin(vr / self._Va)

    def _forces_moments(self, delta):
        """
        return the forces on the UAV based on the state, wind, and control surfaces
        :param delta: np.matrix(delta_a, delta_e, delta_r, delta_t)
        :return: Forces and Moments on the UAV np.matrix(Fx, Fy, Fz, Ml, Mn, Mm)
        """
        phi, theta, psi = quaternion_to_euler(self._state[6:10])
        p = self._state.item(10)
        q = self._state.item(11)
        r = self._state.item(12)

        c = MAV.c
        b = MAV.b
        qbar = 0.5 * MAV.rho * self._Va ** 2 * MAV.S_wing

        CL = (MAV.C_L_0 + MAV.C_L_alpha * self._alpha
              + MAV.C_L_q * c / (2 * self._Va) * q
              + MAV.C_L_delta_e * delta.elevator)
        CD = (MAV.C_D_p + MAV.C_D_alpha * self._alpha
              + MAV.C_D_q * c / (2 * self._Va) * q
              + MAV.C_D_delta_e * abs(delta.elevator))

        F_lift = qbar * CL
        F_drag = qbar * CD

        thrust_prop, torque_prop = self._motor_thrust_torque(self._Va, delta.throttle)

        ca = np.cos(self._alpha)
        sa = np.sin(self._alpha)
        fx = (-MAV.mass * MAV.gravity * np.sin(theta)
              + thrust_prop - F_drag * ca + F_lift * sa)
        fy = (MAV.mass * MAV.gravity * np.cos(theta) * np.sin(phi)
              + qbar * (MAV.C_Y_0 + MAV.C_Y_beta * self._beta
                        + MAV.C_Y_p * b / (2 * self._Va) * p
                        + MAV.C_Y_r * b / (2 * self._Va) * r
                        + MAV.C_Y_delta_a * delta.aileron
                        + MAV.C_Y_delta_r * delta.rudder))
        fz = (MAV.mass * MAV.gravity * np.cos(theta) * np.cos(phi)
              - F_drag * sa - F_lift * ca)

        C_m = (MAV.C_m_0 + MAV.C_m_alpha * self._alpha
               + MAV.C_m_q * c / (2 * self._Va) * q
               + MAV.C_m_delta_e * delta.elevator)
        C_ell = (MAV.C_ell_0 + MAV.C_ell_beta * self._beta
                 + MAV.C_ell_p * b / (2 * self._Va) * p
                 + MAV.C_ell_r * b / (2 * self._Va) * r
                 + MAV.C_ell_delta_a * delta.aileron
                 + MAV.C_ell_delta_r * delta.rudder)
        C_n = (MAV.C_n_0 + MAV.C_n_beta * self._beta
               + MAV.C_n_p * b / (2 * self._Va) * p
               + MAV.C_n_r * b / (2 * self._Va) * r
               + MAV.C_n_delta_a * delta.aileron
               + MAV.C_n_delta_r * delta.rudder)

        My = qbar * c * C_m
        Mx = qbar * b * C_ell - torque_prop
        Mz = qbar * b * C_n

        self._forces[0] = fx
        self._forces[1] = fy
        self._forces[2] = fz
        return np.array([[fx, fy, fz, Mx, My, Mz]]).T

    def _motor_thrust_torque(self, Va, delta_t):
        # compute thrust and torque due to propeller (McLain addendum)
        V_in = MAV.V_max * delta_t

        def torque_balance(omega_p):
            if omega_p == 0:
                return 1e6
            J_p = 2 * np.pi * Va / (omega_p * MAV.D_prop)
            C_Q_prop = MAV.C_Q0 + MAV.C_Q1 * J_p + MAV.C_Q2 * J_p ** 2
            Q_aero = MAV.rho * MAV.D_prop ** 5 * (omega_p / (2 * np.pi)) ** 2 * C_Q_prop
            i = (V_in - MAV.KV * omega_p) / MAV.R_motor
            Q_motor = MAV.KQ * i
            return Q_aero - Q_motor

        omega_p = fsolve(torque_balance, 300.0)[0]
        J_p = 2 * np.pi * Va / (omega_p * MAV.D_prop)
        C_T_prop = MAV.C_T0 + MAV.C_T1 * J_p + MAV.C_T2 * J_p ** 2
        C_Q_prop = MAV.C_Q0 + MAV.C_Q1 * J_p + MAV.C_Q2 * J_p ** 2
        n = omega_p / (2 * np.pi)
        thrust_prop = MAV.rho * MAV.D_prop ** 4 * n ** 2 * C_T_prop
        torque_prop = MAV.rho * MAV.D_prop ** 5 * n ** 2 * C_Q_prop
        return thrust_prop, torque_prop

    def _update_true_state(self):
        # rewrite this function because we now have more information
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
        self.true_state.bx = 0
        self.true_state.by = 0
        self.true_state.bz = 0
        self.true_state.gimbal_az = 0
        self.true_state.gimbal_el = 0
