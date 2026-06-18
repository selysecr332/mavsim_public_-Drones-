"""
compute_trim 
    - Chapter 5 assignment for Beard & McLain, PUP, 2012
    - Update history:  
        12/29/2018 - RWB
"""
import numpy as np
from scipy.optimize import minimize
from tools.rotations import euler_to_quaternion
from message_types.msg_delta import MsgDelta
import time

def compute_trim(mav, Va, gamma):
    # define initial state and input

    # set the initial conditions of the optimization
    e = euler_to_quaternion(0., gamma, 0.)
    state0 = np.array([
        0.,  # pn
        0.,  # pe
        -100.,  # pd
        Va * np.cos(gamma),  # u
        0.,  # v
        Va * np.sin(gamma),  # w
        e.item(0),  # e0
        e.item(1),  # e1
        e.item(2),  # e2
        e.item(3),  # e3
        0.,  # p
        0.,  # q
        0.,  # r
    ])
    delta0 = np.array([0., 0., 0., 0.5])  # elevator, aileron, rudder, throttle
    x0 = np.concatenate((state0, delta0))
    # define equality constraints
    cons = ({'type': 'eq',
             'fun': lambda x: np.array([
                                x[3]**2 + x[4]**2 + x[5]**2 - Va**2,  # magnitude of velocity vector is Va
                                x[4],  # v=0, force side velocity to be zero
                                x[6]**2 + x[7]**2 + x[8]**2 + x[9]**2 - 1.,  # force quaternion to be unit length
                                x[7],  # e1=0  - forcing e1=e3=0 ensures zero roll and zero yaw in trim
                                x[9],  # e3=0
                                x[10],  # p=0  - angular rates should all be zero
                                x[11],  # q=0
                                x[12],  # r=0
                                ]),
             'jac': lambda x: np.array([
                                [0., 0., 0., 2*x[3], 2*x[4], 2*x[5], 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                                [0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                                [0., 0., 0., 0., 0., 0., 2*x[6], 2*x[7], 2*x[8], 2*x[9], 0., 0., 0., 0., 0., 0., 0.],
                                [0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                                [0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0.],
                                [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0.],
                                [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0.],
                                [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0.],
                                ])
             })
    # solve the minimization problem to find the trim states and inputs

    res = minimize(trim_objective_fun, x0, method='SLSQP', args=(mav, Va, gamma),
                   constraints=cons, 
                   options={'ftol': 1e-10, 'disp': True})
    # extract trim state and input and return
    trim_state = res.x[0:13].reshape(-1, 1)
    trim_input = MsgDelta(elevator=res.x.item(13),
                          aileron=res.x.item(14),
                          rudder=res.x.item(15),
                          throttle=res.x.item(16))
    trim_input.print()
    print('trim_state=', trim_state.T)
    return trim_state, trim_input


def trim_objective_fun(x, mav, Va, gamma):
    # objective function to be minimized
    state = x[0:13].reshape(-1, 1)
    delta = MsgDelta(
        elevator=x[13],
        aileron=x[14],
        rudder=x[15],
        throttle=x[16],
    )
    mav._state = state
    mav._update_velocity_data()
    forces_moments = mav._forces_moments(delta)
    x_dot = mav._f(state, forces_moments)
    J = (x_dot.item(3) ** 2 + x_dot.item(4) ** 2 + x_dot.item(5) ** 2
         + x_dot.item(10) ** 2 + x_dot.item(11) ** 2 + x_dot.item(12) ** 2)
    return J
