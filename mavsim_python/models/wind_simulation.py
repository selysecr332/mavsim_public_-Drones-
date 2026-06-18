"""
Class to determine wind velocity at any given moment,
calculates a steady wind speed and uses a stochastic
process to represent wind gusts. (Follows section 4.4 in uav book)
"""
from tools.transfer_function import TransferFunction
import numpy as np


class WindSimulation:
    def __init__(self, Ts, gust_flag=True, steady_state=np.array([[0., 0., 0.]]).T):
        # steady state wind defined in the inertial frame
        self._steady_state = steady_state

        # Dryden gust model parameters (section 4.4 UAV book)
        Va = 25
        Lu = 200
        Lv = 200
        Lw = 50
        if gust_flag:
            sigma_u = 0.05
            sigma_v = 0.05
            sigma_w = 0.02
        else:
            sigma_u = 0.0
            sigma_v = 0.0
            sigma_w = 0.0

        num_u = np.array([[sigma_u * np.sqrt(2 * Va / Lu)]])
        den_u = np.array([[1, Va / Lu]])
        self.u_w = TransferFunction(num=num_u, den=den_u, Ts=Ts)

        num_v = np.array([[sigma_v * np.sqrt(3 * Va / Lv),
                           sigma_v * np.sqrt(3 * Va / Lv) * Va / (np.sqrt(3) * Lv)]])
        den_v = np.array([[1, 2 * Va / Lv, Va ** 2 / Lv ** 2]])
        self.v_w = TransferFunction(num=num_v, den=den_v, Ts=Ts)

        num_w = np.array([[sigma_w * np.sqrt(3 * Va / Lw),
                           sigma_w * np.sqrt(3 * Va / Lw) * Va / (np.sqrt(3) * Lw)]])
        den_w = np.array([[1, 2 * Va / Lw, Va ** 2 / Lw ** 2]])
        self.w_w = TransferFunction(num=num_w, den=den_w, Ts=Ts)
        self._Ts = Ts

    def update(self):
        # returns a six vector.
        #   The first three elements are the steady state wind in the inertial frame
        #   The second three elements are the gust in the body frame
        gust = np.array([[self.u_w.update(np.random.randn())],
                         [self.v_w.update(np.random.randn())],
                         [self.w_w.update(np.random.randn())]])
        return np.concatenate((self._steady_state, gust))
