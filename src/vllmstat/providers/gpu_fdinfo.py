"""DRM ``fdinfo`` aggregation for real Intel (``xe``/``i915``) util% and VRAM.

This is the method ``nvtop`` uses: every process that holds an open GPU file
descriptor exposes per-client accounting under ``/proc/<pid>/fdinfo/<fd>``. We
sum the per-client busy-cycle counters per engine and divide the delta by the
engine's elapsed-cycle delta between two samples to get utilisation, and sum the
per-client resident VRAM to get VRAM used.

Everything here is pure and injectable: ``proc_root`` and the previous-sample
state are passed in, so tests point it at a fake ``/proc`` tree under
``tmp_path``. Every read catches ``OSError`` and degrades; nothing here raises.

xe fdinfo schema (tab-separated ``key:\\tvalue``), captured live::

    drm-driver:           xe
    drm-pdev:             0000:06:00.0
    drm-client-id:        1559
    drm-cycles-ccs:       7850485381106     # THIS client's busy cycles, compute
    drm-total-cycles-ccs: 26411275121822    # engine elapsed-cycle counter
    drm-resident-vram0:   31645832 KiB      # THIS client's resident VRAM

``drm-cycles-<eng>`` is per-client busy; ``drm-total-cycles-<eng>`` is the
engine's elapsed-cycle counter (≈global, ~identical across clients). A client
may appear in several fds with the same ``drm-client-id`` — we dedup so each is
counted once.
"""

from __future__ import annotations

import glob
import os
from dataclasses import dataclass

_DRM_DRIVERS = frozenset({"xe", "i915"})


@dataclass(frozen=True)
class FdinfoStats:
    """Aggregated GPU stats for one PCI device from its DRM clients.

    ``util_pct`` is ``None`` until a second sample provides a cycle delta;
    ``vram_used_bytes`` is available from the first sample; ``clients`` is the
    number of unique DRM clients found on the device.
    """

    util_pct: float | None
    vram_used_bytes: int | None
    clients: int


def _parse_fdinfo(text: str) -> dict[str, str]:
    """Parse the tab/colon-separated ``key:\\tvalue`` fdinfo body into a dict."""
    out: dict[str, str] = {}
    for line in text.splitlines():
        key, sep, val = line.partition(":")
        if sep:
            out[key.strip()] = val.strip()
    return out


def _first_int(val: str) -> int | None:
    """Parse the leading integer of an fdinfo value (e.g. ``"31645832 KiB"``)."""
    parts = val.split()
    if not parts:
        return None
    try:
        return int(parts[0])
    except ValueError:
        return None


def read_fdinfo(
    pdev: str,
    *,
    proc_root: str = "/proc",
    prev_busy: dict[str, int] | None,
    prev_total: dict[str, int] | None,
    now: float,
) -> tuple[FdinfoStats, dict[str, int], dict[str, int]]:
    """Aggregate DRM client stats for ``pdev`` and compute utilisation.

    Scans ``<proc_root>/[0-9]*/fdinfo/*`` for clients whose ``drm-driver`` is
    ``xe``/``i915`` and whose ``drm-pdev`` matches ``pdev``, deduping by
    ``drm-client-id``. Per engine, ``busy`` sums per-client ``drm-cycles-<eng>``
    and ``total`` takes the max per-client ``drm-total-cycles-<eng>``; VRAM sums
    per-client ``drm-resident-vram0`` (KiB → bytes).

    Utilisation: with ``prev_busy``/``prev_total`` from the previous sample, for
    each engine ``util = (busy-prev_busy)/(total-prev_total)`` clamped to
    ``[0, 1]`` when the elapsed-cycle delta is positive; the overall
    ``util_pct`` is ``100 * max(util over engines)`` (compute/``ccs`` dominates
    for vLLM). When there is no previous sample, or no engine advanced, it is
    ``None``. VRAM and the client count are always reported.

    Returns ``(stats, new_busy, new_total)``; the caller carries ``new_busy``
    and ``new_total`` forward as the next call's ``prev_*``. Never raises.

    ``now`` is accepted for interface symmetry with the sysfs readers (and to
    let callers thread a monotonic clock through); utilisation is derived from
    the engines' own elapsed-cycle counters, not wall time.
    """
    del now  # cycle-counter ratio is self-normalising; wall time is unused.

    busy: dict[str, int] = {}
    total: dict[str, int] = {}
    vram_kib = 0
    seen_clients: set[str] = set()

    try:
        paths = glob.glob(os.path.join(proc_root, "[0-9]*", "fdinfo", "*"))
    except OSError:
        paths = []

    for path in paths:
        try:
            with open(path, encoding="utf-8", errors="replace") as fh:
                text = fh.read()
        except OSError:
            # PermissionError / FileNotFoundError / ProcessLookupError / etc.
            continue
        if "drm-" not in text:
            continue
        kv = _parse_fdinfo(text)
        if kv.get("drm-driver") not in _DRM_DRIVERS:
            continue
        if kv.get("drm-pdev") != pdev:
            continue
        client_id = kv.get("drm-client-id")
        if client_id is None or client_id in seen_clients:
            continue
        seen_clients.add(client_id)

        for key, raw in kv.items():
            if key.startswith("drm-total-cycles-"):
                eng = key[len("drm-total-cycles-") :]
                n = _first_int(raw)
                if n is not None and n > total.get(eng, 0):
                    total[eng] = n
            elif key.startswith("drm-cycles-"):
                eng = key[len("drm-cycles-") :]
                n = _first_int(raw)
                if n is not None:
                    busy[eng] = busy.get(eng, 0) + n
        resident = _first_int(kv.get("drm-resident-vram0", ""))
        if resident is not None:
            vram_kib += resident

    clients = len(seen_clients)
    vram_used_bytes = vram_kib * 1024 if clients else None

    util_pct: float | None = None
    if prev_busy is not None and prev_total is not None:
        best: float | None = None
        for eng, total_now in total.items():
            dtotal = total_now - prev_total.get(eng, 0)
            if dtotal <= 0:
                continue
            dbusy = busy.get(eng, 0) - prev_busy.get(eng, 0)
            frac = dbusy / dtotal
            frac = max(0.0, min(1.0, frac))
            if best is None or frac > best:
                best = frac
        if best is not None:
            util_pct = 100.0 * best

    stats = FdinfoStats(util_pct=util_pct, vram_used_bytes=vram_used_bytes, clients=clients)
    return stats, busy, total
