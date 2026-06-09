"""Run the FDS executable and report progress."""

from __future__ import annotations

import glob
import os
import re
import shutil
import subprocess
import sys
import time

from .results import Results


def find_fds() -> str:
    """Locate an FDS executable.

    Checks, in order: the ``FDS_EXE`` environment variable, ``fds`` on
    PATH, then binaries compiled in a checkout of the FDS repository next
    to (or containing) this package.
    """
    exe = os.environ.get("FDS_EXE")
    if exe and os.access(exe, os.X_OK):
        return exe
    exe = shutil.which("fds")
    if exe:
        return exe
    here = os.path.dirname(os.path.abspath(__file__))
    for root in (here, os.path.dirname(here), os.path.dirname(os.path.dirname(here)),
                 os.path.dirname(os.path.dirname(os.path.dirname(here)))):
        for hit in sorted(glob.glob(os.path.join(root, "Build", "*", "fds_*"))):
            if os.access(hit, os.X_OK):
                return hit
    raise FileNotFoundError(
        "no FDS executable found — set FDS_EXE, put 'fds' on PATH, "
        "or build one in <repo>/Build/<target>/"
    )


_TIME_RE = re.compile(r"(?:Simulation|Total) Time:\s*([0-9.Ee+-]+)\s*s")


def run(input_file: str, n_mpi: int = 1, n_threads: int = 1,
        fds_exe: str | None = None, progress: bool = True,
        timeout: float | None = None) -> Results:
    """Run FDS on ``input_file`` and return a :class:`Results` handle.

    ``n_mpi`` ranks (one per mesh at most is useful) and ``n_threads``
    OpenMP threads per rank. Progress is printed as the simulation
    advances. Raises ``RuntimeError`` if FDS reports an error.
    """
    input_file = os.path.abspath(input_file)
    workdir = os.path.dirname(input_file)
    chid = _read_chid(input_file)
    exe = fds_exe or find_fds()

    env = dict(os.environ, OMP_NUM_THREADS=str(n_threads))
    if n_mpi > 1:
        cmd = ["mpirun", "--oversubscribe", "-n", str(n_mpi), exe, input_file]
        if os.geteuid() == 0:
            cmd.insert(1, "--allow-run-as-root")
    else:
        cmd = [exe, input_file]

    # FDS appends to old output files; clear previous run products.
    for old in glob.glob(os.path.join(workdir, f"{chid}*")):
        if not old.endswith(".fds"):
            os.remove(old)

    t0 = time.time()
    proc = subprocess.Popen(cmd, cwd=workdir, env=env,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True)
    log_path = os.path.join(workdir, f"{chid}.log")
    t_end = _read_t_end(input_file)
    last_print = 0.0
    with open(log_path, "w") as log:
        for line in proc.stdout:
            log.write(line)
            m = _TIME_RE.search(line)
            if m and progress and time.time() - last_print > 2:
                sim_t = float(m.group(1))
                pct = f" ({100 * sim_t / t_end:.0f}%)" if t_end else ""
                print(f"\r  t = {sim_t:8.2f} s{pct}   "
                      f"[wall {time.time() - t0:6.1f} s]", end="",
                      file=sys.stderr, flush=True)
                last_print = time.time()
    proc.wait(timeout=timeout)
    if progress:
        print(file=sys.stderr)

    out_file = os.path.join(workdir, f"{chid}.out")
    ok = proc.returncode == 0 and _completed(out_file)
    if not ok:
        tail = _tail(log_path) + _tail(out_file)
        raise RuntimeError(f"FDS failed (exit {proc.returncode}):\n{tail}")
    if progress:
        print(f"  done in {time.time() - t0:.1f} s wall time", file=sys.stderr)
    return Results(chid, workdir)


def _read_chid(path: str) -> str:
    with open(path) as f:
        m = re.search(r"CHID\s*=\s*'([^']+)'", f.read())
    if not m:
        raise ValueError(f"no CHID found in {path}")
    return m.group(1)


def _read_t_end(path: str) -> float | None:
    with open(path) as f:
        m = re.search(r"T_END\s*=\s*([0-9.Ee+-]+)", f.read())
    return float(m.group(1)) if m else None


def _completed(out_file: str) -> bool:
    try:
        with open(out_file) as f:
            return "STOP: FDS completed successfully" in f.read()
    except OSError:
        return False


def _tail(path: str, n: int = 15) -> str:
    try:
        with open(path) as f:
            return "".join(f.readlines()[-n:])
    except OSError:
        return ""
