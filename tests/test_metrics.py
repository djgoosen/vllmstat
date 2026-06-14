from pathlib import Path

from vllmstat.core.metrics import MetricsEngine
from vllmstat.core.parse import parse_metrics

FIX = Path(__file__).parent / "fixtures" / "metrics_qwen3.txt"
DIMS = {"layers": 48, "kv_heads": 4, "head_dim": 128}


def test_derive_from_golden_two_samples_for_rates():
    text = FIX.read_text()
    fam = parse_metrics(text)
    eng = MetricsEngine(dims=DIMS, max_model_len=262144)
    # First derive establishes baselines (rates 0)
    s0 = eng.derive(fam, now=0.0)
    assert s0.connected is True
    assert s0.gen_tps == 0.0
    # Second derive: simulate +1420 generation tokens over 10s -> ~142 tok/s eventually
    fam2 = dict(fam)
    # bump generation_tokens_total by 1420
    base = fam["vllm:generation_tokens_total"][0][1]
    fam2["vllm:generation_tokens_total"] = [
        (fam["vllm:generation_tokens_total"][0][0], base + 1420)
    ]
    s1 = eng.derive(fam2, now=10.0)
    assert s1.gen_tps > 0.0


def test_kv_and_cache_fields_present():
    fam = parse_metrics(FIX.read_text())
    eng = MetricsEngine(dims=DIMS, max_model_len=262144)
    s = eng.derive(fam, now=0.0)
    assert s.kv_dtype == "turboquant_k3v4_nc"
    assert s.kv_capacity_tokens == 6947 * 64
    assert s.kv_ratio_kind == "nominal"  # memory_bytes is None on fixture
    assert s.prefix_hit_lifetime is not None and 0.0 <= s.prefix_hit_lifetime <= 1.0
    # token sources: local_cache_hit + local_compute fractions sum ~1
    assert s.src_cache_hit is not None
    assert s.spec_active is True  # fixture has spec decode
    assert s.spec_accepted_per_draft and s.spec_accepted_per_draft > 1.0


def test_efficiency_hidden_when_zero():
    fam = parse_metrics(FIX.read_text())
    eng = MetricsEngine(dims=DIMS, max_model_len=262144)
    s = eng.derive(fam, now=0.0)
    assert s.eff_active is False  # estimated_* are 0 on fixture


def test_latency_quantiles_computed():
    fam = parse_metrics(FIX.read_text())
    eng = MetricsEngine(dims=DIMS, max_model_len=262144)
    eng.derive(fam, now=0.0)
    s = eng.derive(fam, now=1.0)  # same fixture -> windowed delta 0; falls back to lifetime
    # With zero window delta, engine uses cumulative buckets so p50 is defined
    assert s.ttft.p50 is not None


# --- session averages (accumulated while serving) -----------------------------


def _fam(*, gen: float, prompt: float, req: float, running: float):
    """Minimal synthetic Families with just the session-relevant counters."""
    e = {"engine": "0", "model_name": "m"}
    return {
        "vllm:generation_tokens_total": [(e, gen)],
        "vllm:prompt_tokens_total": [(e, prompt)],
        "vllm:request_success_total": [(e, req)],
        "vllm:num_requests_running": [(e, running)],
    }


def test_session_accumulates_active_and_idle():
    eng = MetricsEngine()
    # t=0: baseline (first sample, no accumulation yet)
    s0 = eng.derive(_fam(gen=1000.0, prompt=5000.0, req=10.0, running=2.0), now=0.0)
    assert s0.session_active_s == 0.0
    assert s0.session_idle_s == 0.0
    assert s0.avg_decode_tps is None  # no active time yet

    # t=10: serving (running>0); +1000 gen, +2000 prompt over 10s
    s1 = eng.derive(_fam(gen=2000.0, prompt=7000.0, req=12.0, running=3.0), now=10.0)
    assert s1.session_active_s == 10.0
    assert s1.session_idle_s == 0.0
    assert s1.avg_decode_tps == 100.0  # 1000 gen / 10s active
    assert s1.avg_prefill_tps == 200.0  # 2000 prompt / 10s active

    # t=15: idle (running==0); gen/prompt do not advance, no decode added
    s2 = eng.derive(_fam(gen=2000.0, prompt=7000.0, req=12.0, running=0.0), now=15.0)
    assert s2.session_active_s == 10.0  # unchanged (idle window)
    assert s2.session_idle_s == 5.0
    assert s2.avg_decode_tps == 100.0  # still 1000/10 (idle added no tokens)

    # t=25: serving again; +500 gen, +1000 prompt over 10s active
    s3 = eng.derive(_fam(gen=2500.0, prompt=8000.0, req=14.0, running=1.0), now=25.0)
    assert s3.session_active_s == 20.0
    assert s3.session_idle_s == 5.0
    assert s3.avg_decode_tps == (1000.0 + 500.0) / 20.0  # 75.0
    assert s3.avg_prefill_tps == (2000.0 + 1000.0) / 20.0  # 150.0
    # active fraction = 20 / (20 + 5)
    assert s3.session_active_frac == 20.0 / 25.0
    # session totals/requests are baselined at the first sample
    assert s3.session_requests == 14 - 10  # 4
    assert s3.session_gen_tokens == 2500.0 - 1000.0  # 1500
    assert s3.session_prompt_tokens == 8000.0 - 5000.0  # 3000
    assert s3.avg_gen_tokens_per_req == 1500.0 / 4  # 375.0


def test_session_reset_zeroes_accumulators():
    eng = MetricsEngine()
    eng.derive(_fam(gen=1000.0, prompt=5000.0, req=10.0, running=2.0), now=0.0)
    s1 = eng.derive(_fam(gen=2000.0, prompt=7000.0, req=12.0, running=2.0), now=10.0)
    assert s1.session_active_s > 0.0 and s1.avg_decode_tps is not None

    eng.reset_session()
    # First derive after reset re-baselines: no accumulation, totals zero.
    s2 = eng.derive(_fam(gen=2000.0, prompt=7000.0, req=12.0, running=2.0), now=20.0)
    assert s2.session_active_s == 0.0
    assert s2.session_idle_s == 0.0
    assert s2.avg_decode_tps is None
    assert s2.avg_prefill_tps is None
    assert s2.session_requests == 0
    assert s2.session_gen_tokens == 0.0
    assert s2.avg_gen_tokens_per_req is None


def test_session_rebaselines_on_counter_reset():
    eng = MetricsEngine()
    eng.derive(_fam(gen=5000.0, prompt=9000.0, req=20.0, running=2.0), now=0.0)
    eng.derive(_fam(gen=6000.0, prompt=11000.0, req=22.0, running=2.0), now=10.0)
    # Server restarts: gen_total drops below the session baseline -> re-baseline.
    s = eng.derive(_fam(gen=100.0, prompt=200.0, req=1.0, running=2.0), now=20.0)
    assert s.session_active_s == 0.0
    assert s.session_idle_s == 0.0
    assert s.session_gen_tokens == 0.0
    assert s.session_requests == 0
    assert s.avg_decode_tps is None
