from __future__ import annotations

import math


class MockProvider:
    """Deterministic synthetic vLLM metrics for --mock and tests."""

    def __init__(self) -> None:
        self._tick = 0
        self._gen = 1_000_000.0
        self._prompt = 3_000_000.0
        self._q = 5_000_000.0
        self._h = 1_500_000.0

    def metrics_text(self) -> str:
        self._tick += 1
        t = self._tick
        running = 2 + int(2 * (math.sin(t / 3) + 1))
        waiting = max(0, int(3 * math.sin(t / 5)))
        self._gen += 120 + 40 * math.sin(t / 2)
        self._prompt += 300 + 80 * math.cos(t / 2)
        self._q += 1000
        self._h += 400 + 50 * math.sin(t / 4)
        kv = 0.10 + 0.05 * (math.sin(t / 6) + 1)
        ttft_buckets = self._hist("vllm:time_to_first_token_seconds", base=0.05, n=t)
        tpot_buckets = self._hist("vllm:request_time_per_output_token_seconds", base=0.01, n=t)
        return f"""\
# TYPE vllm:num_requests_running gauge
vllm:num_requests_running{{engine="0",model_name="mock-7b"}} {running}.0
# TYPE vllm:num_requests_waiting gauge
vllm:num_requests_waiting{{engine="0",model_name="mock-7b"}} {waiting}.0
# TYPE vllm:num_preemptions_total counter
vllm:num_preemptions_total{{engine="0",model_name="mock-7b"}} 0.0
# TYPE vllm:generation_tokens_total counter
vllm:generation_tokens_total{{engine="0",model_name="mock-7b"}} {self._gen:.1f}
# TYPE vllm:prompt_tokens_total counter
vllm:prompt_tokens_total{{engine="0",model_name="mock-7b"}} {self._prompt:.1f}
# TYPE vllm:request_success_total counter
vllm:request_success_total{{engine="0",model_name="mock-7b"}} {t * 4}.0
# TYPE vllm:prefix_cache_queries_total counter
vllm:prefix_cache_queries_total{{engine="0",model_name="mock-7b"}} {self._q:.1f}
# TYPE vllm:prefix_cache_hits_total counter
vllm:prefix_cache_hits_total{{engine="0",model_name="mock-7b"}} {self._h:.1f}
# TYPE vllm:prompt_tokens_cached_total counter
vllm:prompt_tokens_cached_total{{engine="0",model_name="mock-7b"}} {self._h:.1f}
# TYPE vllm:prompt_tokens_recomputed_total counter
vllm:prompt_tokens_recomputed_total{{engine="0",model_name="mock-7b"}} 12.0
# TYPE vllm:prompt_tokens_by_source_total counter
vllm:prompt_tokens_by_source_total{{engine="0",model_name="mock-7b",source="local_compute"}} {self._prompt * 0.7:.1f}
vllm:prompt_tokens_by_source_total{{engine="0",model_name="mock-7b",source="local_cache_hit"}} {self._prompt * 0.3:.1f}
vllm:prompt_tokens_by_source_total{{engine="0",model_name="mock-7b",source="external_kv_transfer"}} 0.0
# TYPE vllm:kv_cache_usage_perc gauge
vllm:kv_cache_usage_perc{{engine="0",model_name="mock-7b"}} {kv:.4f}
# TYPE vllm:cache_config_info gauge
vllm:cache_config_info{{block_size="16",cache_dtype="fp8_e4m3",num_gpu_blocks="20000",enable_prefix_caching="True",engine="0"}} 1.0
# TYPE vllm:iteration_tokens_total histogram
vllm:iteration_tokens_total_sum{{engine="0",model_name="mock-7b"}} {t * 1024}.0
vllm:iteration_tokens_total_count{{engine="0",model_name="mock-7b"}} {t}.0
# TYPE vllm:spec_decode_num_drafts_total counter
vllm:spec_decode_num_drafts_total{{engine="0",model_name="mock-7b"}} {t * 100}.0
# TYPE vllm:spec_decode_num_draft_tokens_total counter
vllm:spec_decode_num_draft_tokens_total{{engine="0",model_name="mock-7b"}} {t * 500}.0
# TYPE vllm:spec_decode_num_accepted_tokens_total counter
vllm:spec_decode_num_accepted_tokens_total{{engine="0",model_name="mock-7b"}} {t * 210}.0
# TYPE vllm:estimated_flops_per_gpu_total counter
vllm:estimated_flops_per_gpu_total{{engine="0",model_name="mock-7b"}} 0.0
{ttft_buckets}
{tpot_buckets}
"""

    def _hist(self, name: str, *, base_le: float = 0.05, n: int = 1, **kw) -> str:
        base_le = kw.get("base", base_le)
        les = [base_le * m for m in (1, 2, 4, 8, 16, 32)]
        lines = [f"# TYPE {name} histogram"]
        cum = 0.0
        for le in les:
            cum += n  # cumulative grows each tick
            lines.append(f'{name}_bucket{{le="{le}"}} {cum:.1f}')
        lines.append(f'{name}_bucket{{le="+Inf"}} {cum + n:.1f}')
        lines.append(f"{name}_count {cum + n:.1f}")
        lines.append(f"{name}_sum {cum * base_le:.3f}")
        return "\n".join(lines)
