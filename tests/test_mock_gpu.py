from vllmstat.providers.mock import mock_gpu_snapshot


def test_mock_gpu_two_devices():
    snap = mock_gpu_snapshot(0)
    assert snap.available is True and snap.source == "mock"
    assert len(snap.gpus) == 2
    assert snap.gpus[0].mem_total == 24_000_000_000
    assert snap.gpus[1].index == 1
