# pyrelm — a Pythonic interface for FDS

`pyrelm` is a lightweight front-end for the
[Fire Dynamics Simulator (FDS)](https://pages.nist.gov/fds-smv/). It replaces
hand-written namelist files and ad-hoc post-processing with a clean Python
workflow, while the physics stays **100% real FDS** — pyrelm generates a
standard `.fds` input file, runs the actual compiled solver, and reads its
native outputs back into numpy/pandas.

```
Python model  ──►  .fds input file  ──►  FDS solver (Fortran/MPI)  ──►  results
   Model()           m.write()              m.run()                  res.devices
                                                                     res.hrr
                                                                     res.slice()
```

## Quick start

```python
from pyrelm import Model, plot

m = Model("room_fire", t_end=40, dt_output=0.5)
m.mesh(xb=(-0.2, 6.0, -0.2, 4.2, 0.0, 3.0), cell=0.2)
m.open_boundary("xmax", "ymin", "ymax", "zmax")

m.reaction("PROPANE", soot_yield=0.01)
m.room(xb=(0, 5, 0, 4, 0, 2.7), wall_thickness=0.2)   # walls + ceiling
m.hole(xb=(4.9, 5.3, 1.6, 2.6, 0.0, 2.0))             # door

m.fire(xb=(0.8, 1.6, 1.6, 2.4), hrr_kw=300,           # burning armchair
       ramp=[(0, 0), (10, 1), (40, 1)])

m.tc_tree(x=2.5, y=2.0, heights=(0.5, 1.0, 1.5, 2.0, 2.5))
m.slice2d("TEMPERATURE", y=2.0)

print(m.check())          # sanity warnings (stretched cells, cell count…)
res = m.run(n_threads=4)  # runs the real FDS executable

res.devices               # pandas DataFrame, time-indexed
res.hrr                   # heat release rate / energy budget
slc = res.slice("TEMPERATURE")
plot.plot_slice(slc, t=40)
plot.animate_slice(slc, "temp.gif")
```

See `examples/room_fire.py` for the full runnable version.

## What's in the box

| Module | Purpose |
| --- | --- |
| `pyrelm.model` | `Model` builder: meshes, reactions, fires, geometry, devices, slices — plus `add()` as an escape hatch for any raw namelist record |
| `pyrelm.namelist` | Correct FDS namelist serialization (types, quoting, line wrapping) |
| `pyrelm.runner` | Finds the FDS executable, runs it (MPI/OpenMP aware), streams progress, fails loudly with the solver's error text |
| `pyrelm.results` | `Results`: device & HRR CSVs as DataFrames; binary slice files (`.sf`) decoded straight into numpy arrays via the `.smv` index |
| `pyrelm.plot` | One-call matplotlib helpers: device time histories, HRR curve, slice snapshots, slice GIF animation |

## Locating the solver

`pyrelm.runner.find_fds()` checks, in order:

1. the `FDS_EXE` environment variable,
2. `fds` on `PATH`,
3. binaries built in this repository (`Build/<target>/fds_*`).

To build FDS from this repo on Linux with GNU + OpenMPI:

```bash
sudo apt-get install gfortran libopenmpi-dev openmpi-bin
cd Build/ompi_gnu_linux && ./make_fds.sh
```

## Design principles

- **The physics is FDS.** pyrelm never reimplements the solver; every model
  it produces is a plain `.fds` file you can inspect, hand-edit, or run with
  a stock FDS / open in PyroSim or Smokeview.
- **Outputs are data, not screenshots.** Devices and HRR come back as
  pandas DataFrames; slices come back as `(time, i, j, k)` numpy arrays with
  real grid coordinates — ready for scripting, parameter sweeps, and CI.
- **Helpful guardrails.** `Model.check()` flags missing meshes, stretched
  cells, and very large grids before you burn CPU time.

## Status / roadmap

Early but functional (v0.1): meshes, reactions, burner fires with ramps,
obstructions/holes/vents, room helper, materials and surfaces, point
devices, 2D slices, boundary output, runner with progress, devc/hrr/slice
readers, matplotlib plots + GIF animation.

Not yet covered: multi-mesh decomposition helpers, HVAC, complex geometry
(`&GEOM`), particle/sprinkler helpers, 3D rendering, boundary-file reader.
All of these can be reached today through `Model.add(...)` raw records.
