from vllmtop.core.metrics import MetricsEngine
from vllmtop.core.parse import parse_metrics
from vllmtop.providers.mock import MockProvider


def test_mock_emits_parseable_metrics_that_vary():
    p = MockProvider()
    t0 = p.metrics_text()
    t1 = p.metrics_text()
    assert "vllm:generation_tokens_total" in t0
    f0 = parse_metrics(t0)
    f1 = parse_metrics(t1)
    g0 = f0["vllm:generation_tokens_total"][0][1]
    g1 = f1["vllm:generation_tokens_total"][0][1]
    assert g1 > g0  # counter advances


def test_mock_drives_engine():
    p = MockProvider()
    eng = MetricsEngine(dims={"layers": 48, "kv_heads": 4, "head_dim": 128}, max_model_len=262144)
    eng.derive(parse_metrics(p.metrics_text()), now=0.0)
    s = eng.derive(parse_metrics(p.metrics_text()), now=1.0)
    assert s.gen_tps > 0.0
    assert s.kv_dtype is not None
