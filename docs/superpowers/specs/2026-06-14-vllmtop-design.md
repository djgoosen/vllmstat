# vllmtop — Design Spec

- **Status:** Design approved; pre-implementation
- **Date:** 2026-06-14
- **Author:** Bryan Vine (bryan@bryanvine.com)
- **Planned repo:** `github.com/bryanvine/vllmtop` (public)
- **License:** Apache-2.0

## 1. Summary

`vllmtop` is a single-command, zero-infra terminal dashboard for a live vLLM
server — `nvtop`, but for vLLM serving performance. It is a full-screen,
interactive [Textual](https://textual.textualize.io/) TUI that unifies **vLLM
serving telemetry** (scraped from the Prometheus `/metrics` endpoint) with
**GPU hardware stats** (via NVML), so an operator no longer needs `nvtop` plus a
Prometheus/Grafana stack to understand how a server is performing. It is
read-only, like `htop`/`nvtop`.

The headline metrics are concurrency, throughput (tokens/s), cache behavior
(prefix-cache hit rate, KV-cache usage), latency percentiles (TTFT/TPOT/e2e),
speculative-decoding acceptance, and per-GPU utilization/memory/temp/power/clocks.

## 2. Goals / Non-goals

**Goals**
- One command, no infrastructure: `pip install vllmtop && vllmtop`.
- Make a separate `nvtop` unnecessary for the common case (headline per-GPU stats).
- Surface the vLLM-specific signals that `nvtop` and generic GPU tools cannot:
  concurrency/queueing, throughput, cache effectiveness, latency percentiles,
  speculative-decoding acceptance.
- Multi-GPU aware (test target has one GPU; design must not assume one).
- Robust against version drift, missing metrics, GPU-less hosts, and
  disconnects — degrade gracefully, never crash.

**Non-goals (v1)**
- Multi-instance "fleet" view (architecture leaves room; not built in v1).
- Per-process GPU-memory table (explicitly out — per-GPU summary only).
- Recording/replay to disk, Prometheus re-export, alerting.
- Any server-control actions (pause/cancel/etc.) — strictly read-only.

## 3. Background & prior art

- The mainstream approach to vLLM observability is **Prometheus + Grafana**
  (heavy, web-based) or vLLM's official Grafana "Performance Dashboard"
  (benchmarking-oriented). Both require infrastructure.
- The closest existing terminal tool, PyPI **`vllm-top`** (yeok-c, MIT, last
  release Aug 2025), is a lightweight `watch`-style metrics *printer*: it prints
  selected Prometheus numbers on an interval. It is not an interactive
  full-screen TUI, has no GPU-hardware monitoring, no sparklines, no latency
  percentiles, and no speculative-decode analytics.
- **Gap:** no interactive, zero-infra, `nvtop`-class TUI exists for vLLM. That
  gap is what `vllmtop` fills. (The name `vllm-top` is taken; we use `vllmtop`,
  matching the `htop`/`btop`/`nvtop` convention.)

## 4. Decisions (locked)

| Decision | Choice |
|---|---|
| Language / framework | Python + Textual |
| Scope | Single endpoint; multi-GPU / multi-model / multi-engine aware; fleet-ready architecture |
| GPU depth | Per-GPU summary (util/mem/temp/power/fan/clocks). No per-process table. |
| Name | `vllmtop` (repo + PyPI package + command); import package `vllmtop` |
| License | Apache-2.0 |
| Interaction model | Read-only |
| Default target | `http://localhost:8000`, metrics path `/metrics`, refresh 1.0 s |

## 5. Data sources

### 5.1 vLLM `/metrics` (Prometheus text; all names below confirmed live on the test server)

Scraped over HTTP. Names are the `vllm:` family. Labels seen:
`engine="0"`, `model_name="qwen3-30b-tq"`.

- **Concurrency / scheduling:** `num_requests_running`, `num_requests_waiting`
  (gauges); `num_preemptions_total` (counter).
- **Throughput:** `generation_tokens_total`, `prompt_tokens_total` (counters);
  `iteration_tokens_total` (histogram, tokens/engine-step);
  `request_success_total` (counter).
- **Cache (reuse / overlapping KV):** `prefix_cache_queries_total`,
  `prefix_cache_hits_total`, `prompt_tokens_cached_total`,
  `prompt_tokens_recomputed_total`, and
  `prompt_tokens_by_source_total{source=…}` with sources `local_compute`,
  `local_cache_hit`, `external_kv_transfer` — the clearest token-level view of
  the overlapping-KV (prefix-sharing) benefit. Cross-instance:
  `external_prefix_cache_{queries,hits}_total` (0 on test server); multimodal:
  `mm_cache_{queries,hits}_total`.
- **KV memory (capacity / compression):** `kv_cache_usage_perc` (gauge 0–1) plus
  `cache_config_info` labels: `cache_dtype` (e.g. `turboquant_k3v4_nc`),
  `num_gpu_blocks` (6947 on test server), `block_size` (64),
  `kv_cache_memory_bytes` (`None` here), `enable_prefix_caching`,
  `prefix_caching_hash_algo`, `kv_offloading_backend`, `sliding_window`.
- **Latency (histograms):** `time_to_first_token_seconds` (TTFT),
  `inter_token_latency_seconds` (ITL), `request_time_per_output_token_seconds`
  (TPOT), `e2e_request_latency_seconds`, `request_queue_time_seconds`,
  `request_prefill_time_seconds`, `request_decode_time_seconds`,
  `request_inference_time_seconds`.
- **Speculative decoding** (present & active on test server):
  `spec_decode_num_drafts_total`, `spec_decode_num_draft_tokens_total`,
  `spec_decode_num_accepted_tokens_total`,
  `spec_decode_num_accepted_tokens_per_pos_total` (vector by draft position).
- **Efficiency (conditional):** `estimated_flops_per_gpu_total`,
  `estimated_read_bytes_per_gpu_total`, `estimated_write_bytes_per_gpu_total`.
  **These read 0.0 on the test server**, so the efficiency panel must hide
  itself when inputs are zero.
- **Config / info:** `cache_config_info` (labels carry block size, cache dtype,
  prefix-caching flag, gpu mem util, etc.); `engine_sleep_state`.
- Also one-time `GET /v1/models` for served model id(s), `max_model_len`, and the
  model `root` path; when that path is locally readable, its `config.json`
  supplies KV dims (layers / kv-heads / head-dim) for the memory math.

### 5.2 GPU hardware (NVML)

Primary: `nvidia-ml-py` (pynvml). Per device:
`nvmlDeviceGetCount`, `…GetName`, `…GetUtilizationRates` (gpu, memory),
`…GetMemoryInfo` (used/total), `…GetTemperature`, `…GetPowerUsage` +
`…GetEnforcedPowerLimit`, `…GetFanSpeed`, `…GetClockInfo` (SM, MEM),
optionally `…GetMemoryBusWidth` + memory clock for theoretical bandwidth.

Fallback: parse `nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,
memory.total,temperature.gpu,power.draw,power.limit,clocks.sm,clocks.mem,
fan.speed --format=csv,noheader,nounits`.

If neither is available (no NVML library, no `nvidia-smi`, no `/dev/nvidia*` —
e.g. monitoring a remote server or from inside a container without GPU
passthrough), the GPU provider returns "unavailable" and the GPU panel renders a
clear notice while every vLLM panel continues to work.

## 6. Architecture

Strict one-directional data flow: **Providers → Core/State → UI**. All I/O is in
providers; all derivation is pure and unit-tested; the UI only renders state.

### 6.1 Layers

1. **Providers** (async, I/O):
   - `VllmProvider` — async `GET /metrics` (httpx), returns parsed raw metrics;
     one-time `/v1/models`.
   - `GpuProvider` — NVML → `nvidia-smi` → `None` fallback chain; multi-GPU.
   - `MockProvider` — deterministic synthetic data for `--mock` (demos,
     screenshots, CI).
2. **Core / State** (pure, no I/O, no UI):
   - `parse.py` — Prometheus text → metric families (label-aware).
   - `metrics.py` — raw families → derived `Snapshot` (rates, percentiles,
     cache/spec/efficiency derivations); aggregates across `engine`/`model_name`.
   - `histogram.py` — quantiles from cumulative buckets (windowed).
   - `rates.py` — counter→rate with EWMA smoothing and reset detection.
   - `history.py` — fixed-size ring buffers per series (sparklines).
   - `kv.py` — KV-memory capacity, dtype parsing, compression ratio
     (achieved/nominal), and token-source split; reads the served model's
     `config.json` when locally available.
   - `gpu_specs.py` — optional peak-FLOPs lookup for MFU (best-effort).
3. **UI** (Textual): `VllmTopApp` runs a timer (default 1 s), awaits both
   providers concurrently, folds results into `AppState`, and reactive widgets
   re-render. One widget per panel.

### 6.2 Repo layout

```
vllmtop/
  pyproject.toml          # hatchling; deps; console_scripts: vllmtop = vllmtop.cli:main
  README.md  LICENSE  CONTRIBUTING.md  .gitignore
  .github/workflows/ci.yml   # ruff + pyright + pytest (matrix py3.10–3.13)
  src/vllmtop/
    __init__.py  __main__.py
    cli.py  config.py  app.py  state.py
    providers/ base.py vllm.py gpu.py mock.py
    core/ parse.py metrics.py histogram.py rates.py history.py kv.py gpu_specs.py
    widgets/ header.py concurrency.py throughput.py cache.py latency.py
             specdecode.py gpu.py footer.py
  tests/
    conftest.py
    fixtures/metrics_qwen3.txt  fixtures/models.json
    test_parse.py test_histogram.py test_rates.py test_metrics.py
    test_history.py test_providers_vllm.py test_providers_gpu.py
    test_mock.py test_app.py
```

### 6.3 Data flow

`timer tick → asyncio.gather(VllmProvider.sample(), GpuProvider.sample())
→ metrics.derive(raw, prev_raw, dt) → Snapshot → history.push(snapshot)
→ AppState updated → widgets read AppState and render`.

## 7. Derived metrics & formulas

- **Counter rate:** `rate = (c_t − c_{t−1}) / dt`, then EWMA smoothing
  (α ≈ 0.3). If `c_t < c_{t−1}` (server restart), rebaseline and skip one tick.
  Applies to: gen tok/s, prompt tok/s, req/s, preempt/s, cache queries/s.
- **Prefix-cache hit rate (reuse):** lifetime `= hits_total / queries_total`;
  windowed `= Δhits / Δqueries` over the sample window. Cached fraction `=
  prompt_tokens_cached_total / prompt_tokens_total`. Token-source split from
  `prompt_tokens_by_source_total` (compute vs local-cache-hit vs
  external-KV-transfer) — this is the headline "overlapping-KV" view.
- **KV memory & compression:** effective capacity `= num_gpu_blocks ×
  block_size` tokens; current tokens `= capacity × kv_cache_usage_perc`. fp16
  bytes/token `= 2 × num_layers × num_kv_heads × head_dim × 2` (dims from the
  served model's `config.json` when locally readable). Compression ratio:
  **achieved** `= (capacity × fp16_bytes_per_token) / kv_cache_memory_bytes` when
  that field is populated; otherwise **nominal** parsed from the dtype name
  (`…kNvM…` → `32 / (N + M)`; `fp8`/`int8` → 4×; `fp16`/`bf16`/`auto` → 1×),
  shown with a `~` and an "approx" tag. fp16-equivalent capacity `= capacity /
  ratio`. Compression and caching are orthogonal axes (shrink vs reuse) and are
  shown as distinct readouts in the same panel.
- **Latency percentiles (p50/p90/p99):** from cumulative histogram buckets,
  windowed via per-bucket deltas (`bucket_t − bucket_{t−1}`); locate the bucket
  where the cumulative fraction crosses the target quantile and linearly
  interpolate within bucket bounds (Prometheus `histogram_quantile` semantics).
  Mean = `Δsum / Δcount`.
- **Speculative decode:** token acceptance `= Δaccepted / Δdraft_tokens`;
  accepted per draft `= Δaccepted / Δdrafts`; mean draft length `=
  Δdraft_tokens / Δdrafts`; per-position acceptance from
  `…accepted_tokens_per_pos_total` normalized by drafts. Panel hidden when
  spec-decode metrics are absent.
- **Efficiency (conditional):** achieved FLOP/s `= rate(estimated_flops_…)`;
  achieved GB/s `= rate(estimated_read_bytes_… + estimated_write_bytes_…)`;
  MFU `= achieved_FLOPs / peak_FLOPs(device, dtype)` (peak from `gpu_specs`
  table, best-effort); bandwidth util `= achieved_BW / peak_BW` (peak from NVML
  bus width × mem clock × 2). Whole panel hidden when inputs are zero or peak is
  unknown.
- **tokens/iteration:** mean `= iteration_tokens_total_sum /
  iteration_tokens_total_count` (windowed).
- **Multi-engine / multi-model:** gauges and counters are summed across
  `engine` and `model_name` labels for headline numbers; the header shows the
  engine count and model id(s). Per-engine/per-model breakdown is future work.

## 8. UI

### 8.1 Layout (illustrative, with test-server data)

```
┌ vllmtop ─ qwen3-30b-tq @ localhost:8000 ─ engine 0 ─ ● connected ─ up 3h12m ─ 1.0s ─ 14:22:07 ┐
                                                                                                  
 CONCURRENCY                 THROUGHPUT                      LATENCY (recent)   p50    p90    p99  
  running  1  ▕▂▃▅▇▆▃▂▏       gen    142 tok/s ▕▃▅▇▆▇█▆▏       TTFT             73ms  180ms  520ms 
  waiting  0  ▕▁▁▂▁▁▁▁▏       prompt 318 tok/s ▕▂▇▃▅▂▆▃▏       TPOT            9.1ms   14ms   28ms 
  preempt 0/s  max-seqs 1     tok/iter 1024 · 4.1 req/s        e2e             1.8s   6.2s    22s  
                                                               queue            0ms    2ms   40ms 
 CACHE & KV MEMORY                                                                                 
  reuse   prefix hit 38.1% ▕▅▆▆▇▆▇▏ life 31.5%   sources compute 69% · cache-hit 31% · ext 0%     
  memory  KV usage 0.09% ▕▁▁▁▁▁▁▏ (0.4k/445k tok)   turboquant_k3v4_nc  ~4.6x vs fp16 (=43.7GB)   
                                                                                                  
 SPEC DECODE (suffix)   acceptance 39.8% ▕▅▆▇▆▇▆▏   accepted/draft 2.16   pos1 78% pos2 41% …      
                                                                                                  
 GPU 0  NVIDIA <name>                              81%   23.1/24.0 GB (96%)   61°C   142/200 W     
        sm  ▕███████████████░░░░▏ 81%   mem ▕██████████████████░▏ 96%   clk 2520/9501 MHz  fan 45% 
                                                                                                  
 [q]uit [p]ause [+/-]interval [g]pu [c]umulative [?]help                        ● vLLM ✓   GPU ✓   
└──────────────────────────────────────────────────────────────────────────────────────────────┘
```

### 8.2 Panels

- **Header:** target URL · model id(s) · engine count · connection state
  (connected/stale/down) · server uptime · refresh interval · clock.
- **Concurrency:** running / waiting (value + sparkline) · preemptions/s.
- **Throughput:** generation tok/s + prompt tok/s (value + sparkline) ·
  tokens/iteration · successful req/s.
- **Cache & KV memory:** *Reuse* — prefix-hit % (windowed + lifetime, sparkline)
  and the token-source split (compute / local-cache-hit / external-KV). *Memory*
  — KV dtype + effective capacity (tokens) + current usage (tokens & %) +
  compression ratio (achieved when `kv_cache_memory_bytes` is known, else
  ~nominal from the dtype). The external-KV line appears only when cross-instance
  transfer is active. (Reuse vs compression are orthogonal: avoid-recompute vs
  shrink-per-token.)
- **Latency:** TTFT, TPOT, e2e, queue — p50/p90/p99 (windowed).
- **Spec decode** (conditional): acceptance, accepted/draft, per-position bars.
- **GPU(s):** per GPU — name · util % (sparkline) · mem used/total % (sparkline)
  · temp · power/limit · fan · clocks. Multiple GPUs stack vertically.
- **Efficiency** (conditional): achieved GFLOP/s, GB/s, MFU %, BW % — shown only
  when `estimated_*` are non-zero.
- **Footer:** keybindings + transient errors/toasts + provider health badges.

### 8.3 Interaction (read-only)

`q` / Ctrl-C quit · `p` pause refresh · `+` / `-` adjust interval · `g` toggle
GPU panel · `c` toggle cumulative ↔ windowed rates · `?` help overlay. Mouse
optional (Textual default).

## 9. CLI & configuration

```
vllmtop [OPTIONS]
  -u, --url URL            base URL (default http://localhost:8000)
      --metrics-path PATH  default /metrics
  -i, --interval SECONDS   refresh interval (default 1.0)
      --api-key KEY        Bearer token (or env VLLM_API_KEY)
      --no-gpu             disable GPU panel
      --mock               synthetic data; no server needed
      --once --json        print one snapshot as JSON and exit (scriptable)
      --version
```

Precedence: CLI flags > env vars > `~/.config/vllmtop/config.toml` > defaults.

## 10. Resilience & graceful degradation

- **No GPU access:** GPU panel shows "unavailable (no NVML/nvidia-smi)"; all
  vLLM panels keep working. (This is the state of the dev sandbox and of any
  remote-monitoring use.)
- **`/metrics` unreachable:** header → ● down; exponential-backoff reconnect;
  last snapshot retained and marked *stale*.
- **Missing metrics (version drift):** panels self-hide or show n/a; defensive
  parsing; never crash.
- **Counter reset (restart):** detected via decreasing counter → rebaseline.

## 11. Testing strategy (TDD)

- Capture a **real fixture** from the live test server
  (`tests/fixtures/metrics_qwen3.txt`, plus `models.json`) as golden input.
- **Unit (pure core):** `parse`, `histogram` (quantiles), `rates` (EWMA +
  reset), `metrics` (all derivations), `history` (ring buffer) — fast, no I/O.
- **Providers:** `VllmProvider` against the fixture via mocked httpx;
  `GpuProvider` against a fake NVML object and a fake `nvidia-smi` output;
  `MockProvider` determinism.
- **TUI:** Textual `App.run_test()` / Pilot for layout snapshots and key
  handling.
- **E2E smoke:** run against the live server on `:8000` for the vLLM half. The
  GPU half is validated with the fake NVML in CI plus a manual host smoke test by
  the maintainer (the dev sandbox has no GPU access).
- CI: GitHub Actions running ruff (lint/format), pyright (types), pytest
  (matrix Python 3.10–3.13).

## 12. Packaging & distribution

- `src/`-layout package, `pyproject.toml` via hatchling, console entry
  `vllmtop = vllmtop.cli:main`, and `python -m vllmtop`.
- Runtime deps: `textual`, `httpx`, `prometheus-client` (text parser),
  `nvidia-ml-py` (import-guarded; absence is handled).
- Python ≥ 3.10.
- README with `--mock`-generated screenshots; Apache-2.0 `LICENSE`;
  `CONTRIBUTING.md`. Public repo `github.com/bryanvine/vllmtop`.

## 13. Milestones (high-level; detailed by the implementation plan)

1. Scaffold: package layout, `pyproject.toml`, CI skeleton, LICENSE, `.gitignore`.
2. Core (TDD against fixture): `parse` → `histogram` → `rates` → `metrics` →
   `history`.
3. Providers: `vllm` + `mock`; then `gpu` (NVML/`nvidia-smi`/none) with fake.
4. TUI: app loop + `AppState`; widgets panel-by-panel; `--mock` for visual dev.
5. CLI/config: flags, env, config file, `--once --json`.
6. Resilience: reconnect/backoff, stale state, counter-reset, panel self-hide.
7. Docs + screenshots; create public GitHub repo and push.

## 14. Out of scope (v1)

Fleet/multi-instance view · per-process GPU table · recording/replay · Prometheus
re-export · alerting · any server-control actions.

## 15. Open questions / risks

- **MFU accuracy:** peak-FLOPs are dtype- and device-dependent; the efficiency
  panel is best-effort and hidden when inputs are zero (as on the test server).
- **GPU testing:** the dev sandbox has no GPU access; GPU code relies on a fake
  NVML in CI plus a manual host smoke test.
- **Multi-engine/model aggregation:** v1 sums across labels for headline numbers
  and reports counts in the header; per-engine/model drill-down is future work.
- **Histogram bucket coverage:** windowed quantiles assume buckets don't change
  between scrapes; bucket-set changes (rare, config-dependent) trigger a
  rebaseline.
- **Compression ratio:** the nominal ratio parsed from the dtype name ignores
  quantization overhead (scales/groups), so it is labeled approximate; the
  achieved ratio is shown instead when `kv_cache_memory_bytes` is populated.
  Absolute GB figures need the served model's `config.json` to be locally
  readable — remote monitoring falls back to nominal ratio + token capacity.
