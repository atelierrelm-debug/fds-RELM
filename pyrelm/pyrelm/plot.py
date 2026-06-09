"""Matplotlib helpers for FDS results."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from .results import Results, Slice  # noqa: E402


def plot_devices(res: Results, columns: list[str] | None = None,
                 ax=None, title: str | None = None):
    """Plot device time histories (all devices by default)."""
    df = res.devices
    units = res.units
    columns = columns or list(df.columns)
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 5))
    for col in columns:
        ax.plot(df.index, df[col], label=col)
    unit_set = {units.get(c, "") for c in columns}
    ylabel = unit_set.pop() if len(unit_set) == 1 else "value"
    ax.set_xlabel("time (s)")
    ax.set_ylabel(ylabel)
    ax.set_title(title or f"{res.chid}: devices")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    return ax


def plot_hrr(res: Results, ax=None):
    """Plot the heat release rate over time."""
    df = res.hrr
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 4))
    ax.plot(df.index, df["HRR"], color="firebrick")
    ax.set_xlabel("time (s)")
    ax.set_ylabel("HRR (kW)")
    ax.set_title(f"{res.chid}: heat release rate")
    ax.grid(alpha=0.3)
    return ax


def plot_slice(slc: Slice, t: float, ax=None, cmap: str = "inferno",
               vmin: float | None = None, vmax: float | None = None):
    """Plot a slice snapshot at the output time nearest ``t``."""
    h, v, vals = slc.frame(t)
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 5))
    mesh = ax.pcolormesh(h, v, vals, cmap=cmap, vmin=vmin, vmax=vmax,
                         shading="gouraud")
    plt.colorbar(mesh, ax=ax, label=f"{slc.quantity} ({slc.units})")
    axis, coord = slc.plane()
    idx = int(np.argmin(np.abs(slc.times - t)))
    ax.set_title(f"{slc.quantity} @ {axis}={coord:g} m, t={slc.times[idx]:.1f} s")
    ax.set_aspect("equal")
    other = {"x": ("y", "z"), "y": ("x", "z"), "z": ("x", "y")}[axis]
    ax.set_xlabel(f"{other[0]} (m)")
    ax.set_ylabel(f"{other[1]} (m)")
    return ax


def animate_slice(slc: Slice, path: str, fps: int = 10,
                  cmap: str = "inferno", every: int = 1) -> str:
    """Render the whole slice history into a GIF at ``path``."""
    from matplotlib.animation import FuncAnimation, PillowWriter

    h, v, _ = slc.frame_at(0)
    vmin = float(slc.data.min())
    vmax = float(slc.data.max())
    fig, ax = plt.subplots(figsize=(8, 5))
    mesh = ax.pcolormesh(h, v, slc.frame_at(0)[2], cmap=cmap,
                         vmin=vmin, vmax=vmax, shading="gouraud")
    plt.colorbar(mesh, ax=ax, label=f"{slc.quantity} ({slc.units})")
    axis, coord = slc.plane()
    ax.set_aspect("equal")
    frames = range(0, len(slc.times), every)

    def update(i):
        mesh.set_array(slc.frame_at(i)[2].ravel())
        ax.set_title(f"{slc.quantity} @ {axis}={coord:g} m, "
                     f"t={slc.times[i]:.1f} s")
        return [mesh]

    anim = FuncAnimation(fig, update, frames=frames)
    anim.save(path, writer=PillowWriter(fps=fps))
    plt.close(fig)
    return path
