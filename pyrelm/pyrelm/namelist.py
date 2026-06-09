"""Serialization of FDS namelist groups.

FDS input files are made of Fortran namelist records:

    &GROUP KEY=VALUE, KEY=VALUE, ... /

This module turns Python values into correctly formatted namelist text.
"""

from __future__ import annotations

import numbers
from collections.abc import Sequence

MAX_LINE = 100


def fmt_scalar(v) -> str:
    if isinstance(v, bool):
        return ".TRUE." if v else ".FALSE."
    if isinstance(v, numbers.Integral):
        return str(int(v))
    if isinstance(v, numbers.Real):
        s = f"{float(v):.6g}"
        # FDS reads plain integers into REAL fields fine, but keep a decimal
        # point for clarity in the generated file.
        if "e" not in s and "." not in s and "inf" not in s and "nan" not in s:
            s += "."
        return s
    if isinstance(v, str):
        return "'" + v.replace("'", "''") + "'"
    raise TypeError(f"cannot serialize {v!r} into an FDS namelist")


def fmt_value(v) -> str:
    if isinstance(v, str) or not isinstance(v, Sequence):
        return fmt_scalar(v)
    return ",".join(fmt_scalar(x) for x in v)


class Namelist:
    """One FDS namelist record, e.g. ``&OBST XB=0,1,0,1,0,1 /``.

    Parameters are kept in insertion order. ``None`` values are skipped so
    callers can pass optional arguments straight through.
    """

    def __init__(self, group: str, **params):
        self.group = group.upper()
        self.params: dict[str, object] = {}
        for key, value in params.items():
            if value is not None:
                self.params[key.upper()] = value

    def __setitem__(self, key: str, value):
        if value is not None:
            self.params[key.upper()] = value

    def __getitem__(self, key: str):
        return self.params[key.upper()]

    def __contains__(self, key: str) -> bool:
        return key.upper() in self.params

    def get(self, key: str, default=None):
        return self.params.get(key.upper(), default)

    def render(self) -> str:
        parts = [f"{k}={fmt_value(v)}" for k, v in self.params.items()]
        lines = [f"&{self.group}"]
        for i, part in enumerate(parts):
            tail = "," if i < len(parts) - 1 else ""
            if len(lines[-1]) + len(part) + 1 > MAX_LINE:
                lines.append("      " + part + tail)
            else:
                lines[-1] += " " + part + tail
        return "\n".join(lines) + " /"

    def __repr__(self):
        return f"Namelist({self.render()!r})"
