"""Read FDS output: device CSVs, heat-release CSV, and slice files."""

from __future__ import annotations

import os
import re
import struct
from dataclasses import dataclass, field

import numpy as np
import pandas as pd


class Results:
    """Handle to one finished FDS run (``chid`` in ``directory``)."""

    def __init__(self, chid: str, directory: str = "."):
        self.chid = chid
        self.dir = os.path.abspath(directory)
        self._smv = None

    def _path(self, suffix: str) -> str:
        return os.path.join(self.dir, f"{self.chid}{suffix}")

    # ------------------------------------------------------------- tables

    @property
    def devices(self) -> pd.DataFrame:
        """Device time histories, indexed by time (s)."""
        return self._csv("_devc.csv")

    @property
    def hrr(self) -> pd.DataFrame:
        """Heat-release-rate and energy-budget time history."""
        return self._csv("_hrr.csv")

    def _csv(self, suffix: str) -> pd.DataFrame:
        df = pd.read_csv(self._path(suffix), header=1)
        return df.set_index(df.columns[0])

    @property
    def units(self) -> dict[str, str]:
        """Column units for the device CSV."""
        with open(self._path("_devc.csv")) as f:
            units = [u.strip().strip('"') for u in f.readline().split(",")]
            names = [n.strip().strip('"') for n in f.readline().split(",")]
        return dict(zip(names, units))

    # ------------------------------------------------------------- slices

    @property
    def slices(self) -> list["Slice"]:
        """All slice files recorded by the run."""
        return self._smv_info().slices

    def slice(self, quantity: str, mesh: int | None = None) -> "Slice":
        """First slice matching ``quantity`` (case-insensitive substring)."""
        hits = [s for s in self.slices
                if quantity.lower() in s.quantity.lower()
                and (mesh is None or s.mesh == mesh)]
        if not hits:
            have = ", ".join(sorted({s.quantity for s in self.slices})) or "none"
            raise KeyError(f"no slice matching {quantity!r}; available: {have}")
        return hits[0]

    def _smv_info(self) -> "SMVInfo":
        if self._smv is None:
            self._smv = parse_smv(self._path(".smv"), self.dir)
        return self._smv

    def __repr__(self):
        return f"Results({self.chid!r}, {self.dir!r})"


# ---------------------------------------------------------------- smv file


@dataclass
class MeshGrid:
    ijk: tuple[int, int, int]
    x: np.ndarray = None
    y: np.ndarray = None
    z: np.ndarray = None


@dataclass
class SMVInfo:
    meshes: list[MeshGrid] = field(default_factory=list)
    slices: list["Slice"] = field(default_factory=list)


def parse_smv(path: str, directory: str) -> SMVInfo:
    """Extract mesh grids and slice-file entries from a ``.smv`` file."""
    info = SMVInfo()
    with open(path, errors="replace") as f:
        lines = f.readlines()
    i = 0
    while i < len(lines):
        key = lines[i].split()[0] if lines[i].strip() else ""
        if key == "GRID":
            nx, ny, nz = map(int, lines[i + 1].split()[:3])
            info.meshes.append(MeshGrid((nx, ny, nz)))
            i += 2
        elif key in ("TRNX", "TRNY", "TRNZ") and info.meshes:
            mesh = info.meshes[-1]
            n_skip = int(lines[i + 1])
            n = mesh.ijk[("TRNX", "TRNY", "TRNZ").index(key)] + 1
            start = i + 2 + n_skip
            coords = np.array([float(lines[j].split()[1])
                               for j in range(start, start + n)])
            setattr(mesh, key[3].lower(), coords)
            i = start + n
        elif key.startswith("SLC"):  # SLCF / SLCC (cell-centered)
            mesh_no = int(lines[i].split("!")[0].split("&")[0].split("#")[0]
                          .split()[1])
            filename = lines[i + 1].strip()
            quantity = lines[i + 2].strip()
            units = lines[i + 4].strip()
            info.slices.append(Slice(os.path.join(directory, filename),
                                     quantity, units, mesh_no,
                                     info.meshes[mesh_no - 1],
                                     cell_centered=key == "SLCC"))
            i += 5
        else:
            i += 1
    return info


# --------------------------------------------------------------- sf reader


def _records(path: str):
    """Yield raw payloads of Fortran sequential unformatted records."""
    with open(path, "rb") as f:
        while True:
            head = f.read(4)
            if len(head) < 4:
                return
            (n,) = struct.unpack("<i", head)
            payload = f.read(n)
            f.read(4)  # trailing length marker
            yield payload


@dataclass
class Slice:
    """One animated 2D slice. Data loads lazily on first access."""

    path: str
    quantity: str
    units: str
    mesh: int
    grid: MeshGrid
    cell_centered: bool = False
    _times: np.ndarray = None
    _data: np.ndarray = None
    _bounds: tuple = None

    def _load(self):
        if self._data is not None:
            return
        recs = _records(self.path)
        next(recs), next(recs), next(recs)  # quantity / short label / units
        i1, i2, j1, j2, k1, k2 = struct.unpack("<6i", next(recs))
        shape = (i2 - i1 + 1, j2 - j1 + 1, k2 - k1 + 1)
        times, frames = [], []
        while True:
            try:
                t = struct.unpack("<f", next(recs))[0]
                frame = np.frombuffer(next(recs), dtype="<f4")
            except StopIteration:
                break
            times.append(t)
            frames.append(frame.reshape(shape, order="F"))
        self._bounds = (i1, i2, j1, j2, k1, k2)
        self._times = np.array(times)
        self._data = np.stack(frames) if frames else np.empty((0, *shape))

    @property
    def times(self) -> np.ndarray:
        self._load()
        return self._times

    @property
    def data(self) -> np.ndarray:
        """Array of shape (nt, ni, nj, nk); the slice axis has length 1."""
        self._load()
        return self._data

    def plane(self) -> tuple[str, float]:
        """The slice plane: axis name and coordinate."""
        self._load()
        i1, i2, j1, j2, k1, k2 = self._bounds
        for axis, (a, b), coords in (("x", (i1, i2), self.grid.x),
                                     ("y", (j1, j2), self.grid.y),
                                     ("z", (k1, k2), self.grid.z)):
            if a == b:
                return axis, float(coords[a])
        raise ValueError("slice is not planar")

    def frame(self, t: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """2D field nearest to time ``t``: returns (horiz, vert, values)."""
        self._load()
        idx = int(np.argmin(np.abs(self._times - t)))
        return self.frame_at(idx)

    def frame_at(self, idx: int):
        self._load()
        i1, i2, j1, j2, k1, k2 = self._bounds
        axis, _ = self.plane()
        data = self._data[idx]
        if axis == "x":
            return self.grid.y[j1:j2 + 1], self.grid.z[k1:k2 + 1], data[0].T
        if axis == "y":
            return self.grid.x[i1:i2 + 1], self.grid.z[k1:k2 + 1], data[:, 0].T
        return self.grid.x[i1:i2 + 1], self.grid.y[j1:j2 + 1], data[:, :, 0].T

    def __repr__(self):
        axis, val = (self.plane() if self._data is not None else ("?", float("nan")))
        return (f"Slice({self.quantity!r} [{self.units}], mesh {self.mesh}, "
                f"plane {axis}={val:g})")
