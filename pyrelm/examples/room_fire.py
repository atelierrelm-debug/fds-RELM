"""End-to-end demo: a sofa-sized fire in a room with an open door.

Builds the model in Python, runs the real FDS solver, then plots the
thermocouple tree, the heat release rate, and an animated temperature
slice through the fire.

Run from this directory:  python3 room_fire.py
"""

import os

from pyrelm import Model, plot

# ---------------------------------------------------------------- model

# A 5 m x 4 m x 2.7 m room inside a slightly larger domain, with a door
# in the +x wall venting to the outside.
m = Model("room_fire", title="Room fire with open door", t_end=40.0,
          dt_output=0.5)

m.mesh(xb=(-0.2, 6.0, -0.2, 4.2, 0.0, 3.0), cell=0.2)
m.open_boundary("xmax", "ymin", "ymax", "zmax")

m.reaction("PROPANE", soot_yield=0.01)

# Room shell (walls + ceiling), then a door carved into the +x wall.
m.room(xb=(0.0, 5.0, 0.0, 4.0, 0.0, 2.7), wall_thickness=0.2)
m.hole(xb=(4.9, 5.3, 1.6, 2.6, 0.0, 2.0))

# 300 kW fire (a burning armchair, roughly) near the back wall,
# ramping up over the first 10 seconds.
m.fire(xb=(0.8, 1.6, 1.6, 2.4), hrr_kw=300,
       ramp=[(0, 0), (10, 1), (40, 1)])

# Instrumentation: thermocouple tree mid-room, one TC over the door,
# and visibility at head height.
m.tc_tree(x=2.5, y=2.0, heights=(0.5, 1.0, 1.5, 2.0, 2.5))
m.device("TC-door", "THERMOCOUPLE", xyz=(5.1, 2.1, 1.8))
m.device("VIS-exit", "VISIBILITY", xyz=(4.5, 2.1, 1.8))

# Animated slices through the fire centerline.
m.slice2d("TEMPERATURE", y=2.0)
m.slice2d("VELOCITY", y=2.0, vector=True)

for w in m.check():
    print("warning:", w)

# ------------------------------------------------------------------ run

res = m.run(directory=os.path.join(os.path.dirname(__file__), "out"),
            n_threads=4)

# -------------------------------------------------------------- results

import matplotlib.pyplot as plt  # noqa: E402

outdir = os.path.join(os.path.dirname(__file__), "out")

plot.plot_devices(res, [c for c in res.devices.columns if c.startswith("TC")],
                  title="Thermocouple tree, room center")
plt.savefig(os.path.join(outdir, "temperatures.png"), dpi=130,
            bbox_inches="tight")

plot.plot_hrr(res)
plt.savefig(os.path.join(outdir, "hrr.png"), dpi=130, bbox_inches="tight")

slc = res.slice("TEMPERATURE")
plot.plot_slice(slc, t=res.devices.index[-1])
plt.savefig(os.path.join(outdir, "temp_slice.png"), dpi=130,
            bbox_inches="tight")

plot.animate_slice(slc, os.path.join(outdir, "temp_slice.gif"), fps=8)

print("wrote plots to", outdir)
