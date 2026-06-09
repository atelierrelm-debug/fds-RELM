"""High-level model builder for FDS.

A :class:`Model` collects geometry, materials, fire sources, devices and
output requests, then serializes them into a valid FDS input file. The
generated file is plain FDS — anything built here runs on a stock FDS
executable and remains editable by hand.

Example
-------
>>> from pyrelm import Model
>>> m = Model("demo", t_end=30)
>>> m.mesh(xb=(0, 4, 0, 3, 0, 2.7), cell=0.2)
>>> m.reaction("PROPANE", soot_yield=0.01)
>>> m.fire(xb=(1.6, 2.4, 1.2, 1.8), hrr_kw=300)
>>> m.device("TC", "TEMPERATURE", xyz=(2, 1.5, 2.5))
>>> m.slice2d("TEMPERATURE", y=1.5)
>>> m.write("demo.fds")
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, field

from .namelist import Namelist

XB = tuple[float, float, float, float, float, float]


def _cells_from_size(xb: XB, cell) -> tuple[int, int, int]:
    if isinstance(cell, (int, float)):
        cell = (cell, cell, cell)
    return tuple(
        max(1, round((xb[2 * i + 1] - xb[2 * i]) / cell[i])) for i in range(3)
    )


@dataclass
class Mesh:
    xb: XB
    ijk: tuple[int, int, int]

    @property
    def cell_size(self) -> tuple[float, float, float]:
        return tuple(
            (self.xb[2 * i + 1] - self.xb[2 * i]) / self.ijk[i] for i in range(3)
        )

    @property
    def n_cells(self) -> int:
        return self.ijk[0] * self.ijk[1] * self.ijk[2]


@dataclass
class Model:
    chid: str
    title: str = ""
    t_end: float = 60.0
    dt_output: float | None = None

    meshes: list[Mesh] = field(default_factory=list)
    records: list[Namelist] = field(default_factory=list)
    _n_fires: int = 0

    # ------------------------------------------------------------------ core

    def add(self, group: str, **params) -> Namelist:
        """Escape hatch: add any raw namelist record."""
        rec = Namelist(group, **params)
        self.records.append(rec)
        return rec

    # ------------------------------------------------------------ domain

    def mesh(self, xb: XB, cell=None, ijk=None) -> Mesh:
        """Add a computational mesh covering ``xb``.

        Give either ``cell`` (target cell size in m, scalar or per-axis)
        or an explicit ``ijk`` cell count.
        """
        if (cell is None) == (ijk is None):
            raise ValueError("give exactly one of cell= or ijk=")
        if ijk is None:
            ijk = _cells_from_size(xb, cell)
        m = Mesh(tuple(xb), tuple(ijk))
        self.meshes.append(m)
        self.add("MESH", ijk=m.ijk, xb=m.xb)
        return m

    def open_boundary(self, *sides: str):
        """Open one or more exterior boundaries to ambient.

        ``sides`` are any of ``xmin xmax ymin ymax zmin zmax``.
        """
        mb = {
            "xmin": "XMIN", "xmax": "XMAX",
            "ymin": "YMIN", "ymax": "YMAX",
            "zmin": "ZMIN", "zmax": "ZMAX",
        }
        for side in sides:
            self.add("VENT", mb=mb[side.lower()], surf_id="OPEN")

    # ---------------------------------------------------------- combustion

    def reaction(self, fuel: str, soot_yield: float | None = None,
                 co_yield: float | None = None, heat_of_combustion: float | None = None,
                 **extra):
        """Define the gas-phase combustion reaction (one per model)."""
        self.add("REAC", fuel=fuel, soot_yield=soot_yield, co_yield=co_yield,
                 heat_of_combustion=heat_of_combustion, **extra)

    def fire(self, xb, hrr_kw: float | None = None, hrrpua: float | None = None,
             z: float | None = None, ramp: list[tuple[float, float]] | None = None,
             color: str = "RED") -> str:
        """Add a fire source as a burner patch.

        ``xb`` is either a horizontal rectangle ``(x1, x2, y1, y2)`` placed at
        height ``z`` (default: floor of the first mesh) or a full 6-tuple.
        Give the total heat release rate ``hrr_kw`` (kW) or ``hrrpua``
        (kW/m²) directly. ``ramp`` is an optional list of ``(time_s,
        fraction)`` pairs scaling the fire over time.

        Returns the SURF ID created for the burner.
        """
        if len(xb) == 4:
            if z is None:
                z = self.meshes[0].xb[4] if self.meshes else 0.0
            xb = (xb[0], xb[1], xb[2], xb[3], z, z)
        xb = tuple(xb)
        area = self._patch_area(xb)
        if (hrr_kw is None) == (hrrpua is None):
            raise ValueError("give exactly one of hrr_kw= or hrrpua=")
        if hrrpua is None:
            if area <= 0:
                raise ValueError("fire patch has zero area; pass hrrpua= instead")
            hrrpua = hrr_kw / area

        self._n_fires += 1
        surf_id = f"FIRE-{self._n_fires}"
        ramp_id = None
        if ramp:
            ramp_id = f"{surf_id}-RAMP"
            for t, f in ramp:
                self.add("RAMP", id=ramp_id, t=t, f=f)
        self.add("SURF", id=surf_id, hrrpua=hrrpua, color=color, ramp_q=ramp_id)
        self.add("VENT", xb=xb, surf_id=surf_id)
        return surf_id

    @staticmethod
    def _patch_area(xb: XB) -> float:
        dx, dy, dz = xb[1] - xb[0], xb[3] - xb[2], xb[5] - xb[4]
        dims = sorted((abs(dx), abs(dy), abs(dz)))
        return dims[1] * dims[2]

    # ------------------------------------------------------------ geometry

    def surface(self, id: str, **params) -> str:
        """Define a named surface (boundary condition), e.g. a hot wall."""
        self.add("SURF", id=id, **params)
        return id

    def material(self, id: str, density: float, conductivity: float,
                 specific_heat: float, **extra) -> str:
        """Define a solid material (kg/m³, W/m/K, kJ/kg/K)."""
        self.add("MATL", id=id, density=density, conductivity=conductivity,
                 specific_heat=specific_heat, **extra)
        return id

    def obstruction(self, xb: XB, surf_id: str | None = None, color: str | None = None,
                    **extra):
        """Add a solid block."""
        self.add("OBST", xb=tuple(xb), surf_id=surf_id, color=color, **extra)

    def hole(self, xb: XB):
        """Carve an opening (door, window) out of obstructions."""
        self.add("HOLE", xb=tuple(xb))

    def room(self, xb: XB, wall_thickness: float = 0.2,
             surf_id: str | None = None):
        """Add four walls and a ceiling enclosing ``xb`` (floor is the
        domain boundary). Walls grow outward from ``xb``; carve doors and
        windows afterwards with :meth:`hole`.
        """
        x1, x2, y1, y2, z1, z2 = xb
        t = wall_thickness
        self.obstruction((x1 - t, x2 + t, y1 - t, y1, z1, z2 + t), surf_id)
        self.obstruction((x1 - t, x2 + t, y2, y2 + t, z1, z2 + t), surf_id)
        self.obstruction((x1 - t, x1, y1, y2, z1, z2 + t), surf_id)
        self.obstruction((x2, x2 + t, y1, y2, z1, z2 + t), surf_id)
        self.obstruction((x1, x2, y1, y2, z2, z2 + t), surf_id)

    # ------------------------------------------------------- instrumentation

    def device(self, id: str, quantity: str, xyz=None, xb=None, **extra):
        """Add a point (or volume) measurement device.

        Common quantities: TEMPERATURE, THERMOCOUPLE, VELOCITY, VISIBILITY,
        'HEAT FLUX' (needs IOR), 'LAYER HEIGHT' (needs xb=).
        """
        self.add("DEVC", id=id, quantity=quantity,
                 xyz=tuple(xyz) if xyz else None,
                 xb=tuple(xb) if xb else None, **extra)

    def tc_tree(self, x: float, y: float, heights, prefix: str = "TC"):
        """Add a vertical thermocouple tree at (x, y)."""
        for z in heights:
            self.device(f"{prefix}-{z:g}m", "THERMOCOUPLE", xyz=(x, y, z))

    def slice2d(self, quantity: str, x: float | None = None,
                y: float | None = None, z: float | None = None,
                vector: bool = False):
        """Request an animated 2D slice of ``quantity`` on a plane."""
        if sum(v is not None for v in (x, y, z)) != 1:
            raise ValueError("give exactly one of x=, y=, z=")
        self.add("SLCF", pbx=x, pby=y, pbz=z, quantity=quantity,
                 vector=vector or None)

    def boundary_output(self, quantity: str):
        """Record ``quantity`` (e.g. 'WALL TEMPERATURE') on all surfaces."""
        self.add("BNDF", quantity=quantity)

    # ------------------------------------------------------------- output

    def render(self) -> str:
        head = [
            Namelist("HEAD", chid=self.chid, title=self.title or self.chid),
            Namelist("TIME", t_end=self.t_end),
        ]
        if self.dt_output is not None:
            head.append(Namelist("DUMP", dt_devc=self.dt_output,
                                 dt_slcf=self.dt_output, dt_hrr=self.dt_output))
        sections: dict[str, list[str]] = {}
        for rec in self.records:
            sections.setdefault(rec.group, []).append(rec.render())
        order = ["MESH", "REAC", "MATL", "SURF", "RAMP", "OBST", "HOLE",
                 "VENT", "DEVC", "SLCF", "BNDF"]
        out = [r.render() for r in head]
        for group in order:
            if group in sections:
                out.append("")
                out.extend(sections.pop(group))
        for group in sorted(sections):
            out.append("")
            out.extend(sections[group])
        out += ["", "&TAIL /", ""]
        return "\n".join(out)

    def write(self, path: str | None = None) -> str:
        """Write the FDS input file; returns its path."""
        path = path or f"{self.chid}.fds"
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w") as f:
            f.write(self.render())
        return path

    def run(self, directory: str | None = None, **kwargs):
        """Write the input file into ``directory`` and run FDS on it.

        Extra keyword arguments are passed to :func:`pyrelm.runner.run`.
        Returns a :class:`pyrelm.results.Results`.
        """
        from .runner import run

        directory = directory or self.chid
        path = self.write(os.path.join(directory, f"{self.chid}.fds"))
        return run(path, **kwargs)

    # -------------------------------------------------------------- checks

    def check(self) -> list[str]:
        """Return a list of human-readable warnings about the model."""
        warnings = []
        if not self.meshes:
            warnings.append("no MESH defined")
        if not any(r.group == "REAC" for r in self.records) and self._n_fires:
            warnings.append("fire defined but no REAC (FDS will default to propane)")
        for i, m in enumerate(self.meshes):
            dx, dy, dz = m.cell_size
            if max(dx, dy, dz) > 2.5 * min(dx, dy, dz):
                warnings.append(
                    f"mesh {i}: cells are stretched "
                    f"({dx:.3g} x {dy:.3g} x {dz:.3g} m); aim for near-cubic cells"
                )
        total = sum(m.n_cells for m in self.meshes)
        if total > 2_000_000:
            warnings.append(f"{total:,} cells — expect a long run time")
        return warnings
