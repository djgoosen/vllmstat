import math

from vllmstat.core.kv import (
    compute_kv,
    fp16_bytes_per_token,
    nominal_ratio,
    parse_kv_bits,
)


def test_parse_kv_bits():
    assert parse_kv_bits("turboquant_k3v4_nc") == (3, 4)
    assert parse_kv_bits("auto") == (16, 16)
    assert parse_kv_bits("fp16") == (16, 16)
    assert parse_kv_bits("fp8_e4m3") == (8, 8)
    assert parse_kv_bits("something_weird") is None


def test_nominal_ratio_k3v4():
    r_k3v4 = nominal_ratio("turboquant_k3v4_nc")
    assert r_k3v4 is not None
    assert math.isclose(r_k3v4, 32 / 7, rel_tol=1e-9)
    assert nominal_ratio("fp16") == 1.0
    assert nominal_ratio("unknown") is None


def test_fp16_bytes_per_token_qwen3():
    # 2(K,V) * 48 layers * 4 kv heads * 128 head_dim * 2 bytes = 98304
    assert fp16_bytes_per_token(layers=48, kv_heads=4, head_dim=128) == 98304


def test_compute_kv_nominal_when_no_memory_bytes():
    info = compute_kv(
        cache_dtype="turboquant_k3v4_nc",
        num_gpu_blocks=6947,
        block_size=64,
        kv_usage=0.10,
        kv_cache_memory_bytes=None,
        dims={"layers": 48, "kv_heads": 4, "head_dim": 128},
        max_model_len=262144,
    )
    assert info.capacity_tokens == 6947 * 64  # 444608
    assert info.used_tokens == round(444608 * 0.10)
    assert info.ratio_kind == "nominal"
    assert info.ratio is not None
    assert math.isclose(info.ratio, 32 / 7, rel_tol=1e-9)
    assert info.fp16_equiv_tokens == round(444608 / (32 / 7))
    # fp16 KV for full ctx: 262144 * 98304 bytes = ~25.77 GB
    assert info.fp16_full_ctx_gb is not None
    assert math.isclose(info.fp16_full_ctx_gb, 262144 * 98304 / 1e9, rel_tol=1e-6)


def test_compute_kv_achieved_when_memory_bytes_present():
    # capacity 1000 tok, fp16=98304 B/tok -> fp16 bytes=98_304_000;
    # actual kv memory=10_000_000 -> achieved ratio=9.8304
    info = compute_kv(
        cache_dtype="turboquant_k3v4_nc",
        num_gpu_blocks=1000,
        block_size=1,
        kv_usage=0.0,
        kv_cache_memory_bytes=10_000_000,
        dims={"layers": 48, "kv_heads": 4, "head_dim": 128},
        max_model_len=1000,
    )
    assert info.ratio_kind == "achieved"
    assert info.ratio is not None
    assert math.isclose(info.ratio, 98_304_000 / 10_000_000, rel_tol=1e-9)
