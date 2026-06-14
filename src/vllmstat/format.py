from __future__ import annotations

from collections.abc import Sequence

_SPARK = "▁▂▃▄▅▆▇█"


def sparkline(values: Sequence[float]) -> str:
    vals = [v for v in values if v is not None]
    if not vals:
        return ""
    lo, hi = min(vals), max(vals)
    span = hi - lo
    if span <= 0:
        return _SPARK[0] * len(vals)
    out = []
    for v in vals:
        idx = int((v - lo) / span * (len(_SPARK) - 1))
        out.append(_SPARK[idx])
    return "".join(out)


def fmt_si(n: float | None) -> str:
    if n is None:
        return "—"
    n = float(n)
    for unit, div in (("T", 1e12), ("B", 1e9), ("M", 1e6), ("k", 1e3)):
        if abs(n) >= div:
            return f"{n / div:.1f}{unit}"
    return f"{n:.0f}"


def fmt_bytes(n: int | None) -> str:
    if n is None:
        return "—"
    g = n / 1e9
    return f"{g:.1f} GB"


def fmt_dur(seconds: float | None) -> str:
    if seconds is None:
        return "—"
    if seconds < 1.0:
        return f"{seconds * 1000:.0f}ms"
    return f"{seconds:.1f}s"


def fmt_pct(frac: float | None) -> str:
    if frac is None:
        return "—"
    return f"{frac * 100:.1f}%"


def fmt_dur_hms(seconds: float | None) -> str:
    """Compact h/m/s duration: ``None``→``—``; ``<60``→``42s``;
    ``<3600``→``12m03s``; else ``1h05m``. Never raises."""
    if seconds is None:
        return "—"
    total = int(seconds)
    if total < 60:
        return f"{total}s"
    if total < 3600:
        m, s = divmod(total, 60)
        return f"{m}m{s:02d}s"
    h, rem = divmod(total, 3600)
    m, _ = divmod(rem, 60)
    return f"{h}h{m:02d}m"
