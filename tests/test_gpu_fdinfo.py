"""Tests for the DRM ``fdinfo`` aggregator (Intel xe/i915 util%/VRAM).

All trees are fakes under ``tmp_path``; nothing here touches the real ``/proc``.
The real-hardware path is root-owned with ``ptrace_scope=1``, so a non-root test
process could not read it anyway — these fixtures mirror the live xe schema.
"""

from __future__ import annotations

import os
import stat
from pathlib import Path

from vllmstat.providers.gpu_fdinfo import FdinfoStats, read_fdinfo

PDEV = "0000:06:00.0"

# A minimal but faithful xe fdinfo record. The aggregator only reads the keys it
# needs, but we keep the noise (multiple engines) to match the live capture.
_TEMPLATE = """drm-driver:\t{driver}
drm-pdev:\t{pdev}
drm-client-id:\t{client_id}
drm-cycles-rcs:\t0
drm-cycles-bcs:\t0
drm-cycles-ccs:\t{ccs}
drm-cycles-vcs:\t0
drm-cycles-vecs:\t0
drm-total-cycles-rcs:\t{total}
drm-total-cycles-bcs:\t{total}
drm-total-cycles-ccs:\t{total}
drm-total-cycles-vcs:\t{total}
drm-total-cycles-vecs:\t{total}
drm-resident-vram0:\t{vram} KiB
drm-active-vram0:\t{vram} KiB
drm-total-vram0:\t{vram} KiB
"""


def _client(
    root: Path,
    pid: int,
    fd: int,
    *,
    client_id: int,
    ccs: int,
    total: int,
    vram: int = 0,
    pdev: str = PDEV,
    driver: str = "xe",
) -> Path:
    """Write one fake ``<root>/<pid>/fdinfo/<fd>`` record. Returns its path."""
    fdinfo = root / str(pid) / "fdinfo"
    fdinfo.mkdir(parents=True, exist_ok=True)
    path = fdinfo / str(fd)
    path.write_text(
        _TEMPLATE.format(
            driver=driver,
            pdev=pdev,
            client_id=client_id,
            ccs=ccs,
            total=total,
            vram=vram,
        )
    )
    return path


def test_first_sample_util_none_vram_present(tmp_path: Path):
    # Two distinct clients, both on our pdev, with VRAM resident.
    _client(tmp_path, 100, 3, client_id=1, ccs=100, total=1000, vram=31_645_832)
    _client(tmp_path, 200, 4, client_id=2, ccs=50, total=1000, vram=1_000_000)

    stats, busy, total = read_fdinfo(
        PDEV, proc_root=str(tmp_path), prev_busy=None, prev_total=None, now=1.0
    )
    assert isinstance(stats, FdinfoStats)
    # No previous sample -> util is undefined on the first call.
    assert stats.util_pct is None
    # VRAM is the sum across unique clients (KiB -> bytes), available immediately.
    assert stats.vram_used_bytes == (31_645_832 + 1_000_000) * 1024
    assert stats.clients == 2
    # Aggregates carried forward for the next delta.
    assert busy["ccs"] == 150
    assert total["ccs"] == 1000


def test_second_sample_computes_util_from_ccs_delta(tmp_path: Path):
    _client(tmp_path, 100, 3, client_id=1, ccs=100, total=1000, vram=10_000)
    _client(tmp_path, 200, 4, client_id=2, ccs=50, total=1000, vram=20_000)
    _stats1, busy1, total1 = read_fdinfo(
        PDEV, proc_root=str(tmp_path), prev_busy=None, prev_total=None, now=1.0
    )

    # Advance: client 1 ccs 100->600 (+500), client 2 ccs unchanged; engine
    # elapsed cycles 1000->2000 (+1000).  dbusy/dtotal = 500/1000 = 50%.
    _client(tmp_path, 100, 3, client_id=1, ccs=600, total=2000, vram=10_000)
    _client(tmp_path, 200, 4, client_id=2, ccs=50, total=2000, vram=20_000)
    stats2, _busy2, _total2 = read_fdinfo(
        PDEV, proc_root=str(tmp_path), prev_busy=busy1, prev_total=total1, now=2.0
    )
    assert stats2.util_pct == 50.0


def test_util_clamped_to_100(tmp_path: Path):
    _client(tmp_path, 100, 3, client_id=1, ccs=0, total=1000)
    _stats1, busy1, total1 = read_fdinfo(
        PDEV, proc_root=str(tmp_path), prev_busy=None, prev_total=None, now=1.0
    )
    # busy advances more than the elapsed-cycle counter (pathological) -> clamp.
    _client(tmp_path, 100, 3, client_id=1, ccs=5000, total=2000)
    stats2, _b, _t = read_fdinfo(
        PDEV, proc_root=str(tmp_path), prev_busy=busy1, prev_total=total1, now=2.0
    )
    assert stats2.util_pct == 100.0


def test_max_engine_dominates(tmp_path: Path):
    # ccs busier than bcs: overall util follows the max engine (compute).
    _client(tmp_path, 100, 3, client_id=1, ccs=0, total=1000)
    _stats1, busy1, total1 = read_fdinfo(
        PDEV, proc_root=str(tmp_path), prev_busy=None, prev_total=None, now=1.0
    )
    # ccs +800 / total +1000 = 80%, while bcs stays at 0 (in the template).
    _client(tmp_path, 100, 3, client_id=1, ccs=800, total=2000)
    stats2, _b, _t = read_fdinfo(
        PDEV, proc_root=str(tmp_path), prev_busy=busy1, prev_total=total1, now=2.0
    )
    assert stats2.util_pct == 80.0


def test_duplicate_client_id_counted_once(tmp_path: Path):
    # Same client-id exposed via two fds (two PIDs sharing the GPU context):
    # it must be aggregated exactly once for both VRAM and cycles.
    _client(tmp_path, 100, 3, client_id=7, ccs=100, total=1000, vram=5_000)
    _client(tmp_path, 100, 9, client_id=7, ccs=100, total=1000, vram=5_000)
    _client(tmp_path, 200, 4, client_id=7, ccs=100, total=1000, vram=5_000)

    stats, busy, _total = read_fdinfo(
        PDEV, proc_root=str(tmp_path), prev_busy=None, prev_total=None, now=1.0
    )
    assert stats.clients == 1
    assert stats.vram_used_bytes == 5_000 * 1024  # counted once, not x3
    assert busy["ccs"] == 100  # counted once, not x3


def test_unreadable_file_is_skipped(tmp_path: Path):
    good = _client(tmp_path, 100, 3, client_id=1, ccs=10, total=1000, vram=4_000)
    bad = _client(tmp_path, 200, 4, client_id=2, ccs=99, total=1000, vram=9_000)
    # Make the second record unreadable -> PermissionError on read -> skipped.
    os.chmod(bad, 0)
    try:
        stats, busy, _total = read_fdinfo(
            PDEV, proc_root=str(tmp_path), prev_busy=None, prev_total=None, now=1.0
        )
    finally:
        os.chmod(bad, stat.S_IRUSR | stat.S_IWUSR)
    # Only the readable client contributes; no raise.
    assert stats.clients == 1
    assert stats.vram_used_bytes == 4_000 * 1024
    assert busy["ccs"] == 10
    assert good.exists()


def test_all_unreadable_yields_none_none_zero(tmp_path: Path):
    bad = _client(tmp_path, 200, 4, client_id=2, ccs=99, total=1000, vram=9_000)
    os.chmod(bad, 0)
    try:
        stats, busy, total = read_fdinfo(
            PDEV, proc_root=str(tmp_path), prev_busy=None, prev_total=None, now=1.0
        )
    finally:
        os.chmod(bad, stat.S_IRUSR | stat.S_IWUSR)
    assert stats.util_pct is None
    assert stats.vram_used_bytes is None
    assert stats.clients == 0
    assert busy == {} and total == {}


def test_empty_proc_root_yields_none_none_zero(tmp_path: Path):
    stats, busy, total = read_fdinfo(
        PDEV, proc_root=str(tmp_path), prev_busy=None, prev_total=None, now=1.0
    )
    assert stats.util_pct is None
    assert stats.vram_used_bytes is None
    assert stats.clients == 0
    assert busy == {} and total == {}


def test_different_pdev_is_ignored(tmp_path: Path):
    # A client bound to a *different* GPU must not leak into our aggregate.
    _client(tmp_path, 100, 3, client_id=1, ccs=10, total=1000, vram=4_000, pdev=PDEV)
    _client(tmp_path, 200, 4, client_id=2, ccs=99, total=1000, vram=9_000, pdev="0000:03:00.0")
    stats, _busy, _total = read_fdinfo(
        PDEV, proc_root=str(tmp_path), prev_busy=None, prev_total=None, now=1.0
    )
    assert stats.clients == 1
    assert stats.vram_used_bytes == 4_000 * 1024


def test_i915_driver_accepted(tmp_path: Path):
    # The aggregator accepts both xe and i915 records.
    _client(tmp_path, 100, 3, client_id=1, ccs=10, total=1000, vram=4_000, driver="i915")
    stats, _busy, _total = read_fdinfo(
        PDEV, proc_root=str(tmp_path), prev_busy=None, prev_total=None, now=1.0
    )
    assert stats.clients == 1
    assert stats.vram_used_bytes == 4_000 * 1024


def test_other_driver_rejected(tmp_path: Path):
    # An amdgpu/nouveau client on the same pdev string must be ignored.
    _client(tmp_path, 100, 3, client_id=1, ccs=10, total=1000, vram=4_000, driver="amdgpu")
    stats, _busy, _total = read_fdinfo(
        PDEV, proc_root=str(tmp_path), prev_busy=None, prev_total=None, now=1.0
    )
    assert stats.clients == 0
    assert stats.vram_used_bytes is None


def test_no_engine_delta_yields_util_none(tmp_path: Path):
    # Two samples with identical counters -> dtotal == 0 for every engine ->
    # util undefined (None), but VRAM/clients still reported.
    _client(tmp_path, 100, 3, client_id=1, ccs=100, total=1000, vram=4_000)
    _stats1, busy1, total1 = read_fdinfo(
        PDEV, proc_root=str(tmp_path), prev_busy=None, prev_total=None, now=1.0
    )
    _client(tmp_path, 100, 3, client_id=1, ccs=100, total=1000, vram=4_000)
    stats2, _b, _t = read_fdinfo(
        PDEV, proc_root=str(tmp_path), prev_busy=busy1, prev_total=total1, now=2.0
    )
    assert stats2.util_pct is None
    assert stats2.vram_used_bytes == 4_000 * 1024
    assert stats2.clients == 1
