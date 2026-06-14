from vllmtop import render
from vllmtop.core.history import History
from vllmtop.core.state import GpuSample, GpuSnapshot, Snapshot


def _snap(**kw) -> Snapshot:
    base = dict(
        ts=0.0,
        connected=True,
        model_names=["m"],
        engine_count=1,
        gen_tps=142.0,
        prompt_tps=318.0,
        running=2.0,
        waiting=0.0,
        kv_dtype="turboquant_k3v4_nc",
        kv_capacity_tokens=444608,
        kv_ratio=32 / 7,
        kv_ratio_kind="nominal",
        kv_usage=0.0009,
        prefix_hit_lifetime=0.315,
        prefix_hit_window=0.381,
        src_compute=0.69,
        src_cache_hit=0.31,
        src_external=0.0,
        spec_active=True,
        spec_acceptance=0.398,
        spec_accepted_per_draft=2.16,
    )
    base.update(kw)
    return Snapshot(**base)  # type: ignore[arg-type]


def test_cache_kv_panel_shows_dtype_and_ratio():
    text = render.cache_kv(_snap(), History())
    assert "turboquant_k3v4_nc" in text
    assert "4.6x" in text or "4.57x" in text
    assert "444" in text  # capacity shown (e.g. 445k or 444,608)


def test_throughput_panel_shows_tps():
    text = render.throughput(_snap(), History())
    assert "142" in text


def test_gpu_panel_unavailable_message():
    s = _snap(gpu=GpuSnapshot(available=False, source="none", error="no NVML"))
    text = render.gpu(s)
    assert "unavailable" in text.lower() or "no nvml" in text.lower()


def test_gpu_panel_renders_device():
    s = _snap(
        gpu=GpuSnapshot(
            available=True,
            source="nvml",
            gpus=[
                GpuSample(
                    index=0,
                    name="NVIDIA Test",
                    util_gpu=81,
                    mem_used=23_100_000_000,
                    mem_total=24_000_000_000,
                    temp_c=61,
                    power_w=142,
                    power_limit_w=200,
                    clock_sm_mhz=2520,
                    clock_mem_mhz=9501,
                    fan_pct=45,
                )
            ],
        )
    )
    text = render.gpu(s)
    assert "NVIDIA Test" in text and "81" in text and "61" in text


def test_specdecode_hidden_when_inactive():
    assert render.specdecode(_snap(spec_active=False)) == ""
