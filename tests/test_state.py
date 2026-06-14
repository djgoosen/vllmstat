from vllmtop.core.state import GpuSnapshot, Quantiles, Snapshot


def test_quantiles_defaults_none():
    q = Quantiles()
    assert q.p50 is None and q.p90 is None and q.p99 is None and q.mean is None


def test_gpu_snapshot_unavailable():
    snap = GpuSnapshot(available=False, source="none", gpus=[], error="no nvml")
    assert snap.available is False
    assert snap.gpus == []


def test_snapshot_minimal_construction():
    s = Snapshot(ts=1.0, connected=True)
    assert s.connected is True
    assert s.running == 0.0
    assert s.spec_active is False
    assert s.gpu.available is False
