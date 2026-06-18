"""
Manual keyboard flight demo for mavsim_python.
Run from repo root:
    python mavsim_python/manual_control/mavsim_manual.py
"""
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, os.fspath(Path(__file__).parents[1]))

import numpy as np
import parameters.simulation_parameters as SIM
from models.mav_dynamics_control import MavDynamics
from models.wind_simulation import WindSimulation
from message_types.msg_delta import MsgDelta
from viewers.view_manager import ViewManager
from manual_control.keyboard_pilot import KeyboardPilot

try:
    import models.model_coef as MC
    trim_state = MC.x_trim.copy()
    trim_input = MsgDelta(
        elevator=MC.u_trim.item(0),
        aileron=MC.u_trim.item(1),
        rudder=MC.u_trim.item(2),
        throttle=MC.u_trim.item(3),
    )
except Exception:
    trim_state = None
    trim_input = None

wind = WindSimulation(SIM.ts_simulation, gust_flag=False)
mav = MavDynamics(SIM.ts_simulation)
if trim_state is not None:
    mav._state = trim_state

pilot = KeyboardPilot(trim_input)
viewers = ViewManager(mav=True, data=True, video=False, video_name="manual_flight.mp4")

KeyboardPilot.print_help()
print("Click the 3D viewer window to look around. Keep this terminal focused for keys.\n")

sim_time = SIM.start_time
end_time = 600.0
last_time = time.time()
no_wind = np.zeros((6, 1))

while sim_time < end_time and not pilot.quit_requested:
    now = time.time()
    dt = max(now - last_time, SIM.ts_simulation)
    last_time = now

    delta = pilot.update(dt)
    mav.update(delta, no_wind)

    viewers.update(
        sim_time,
        true_state=mav.true_state,
        delta=delta,
    )

    sim_time += SIM.ts_simulation
    time.sleep(0.002)

pilot.stop()
viewers.close(dataplot_name="manual_flight_plot")
