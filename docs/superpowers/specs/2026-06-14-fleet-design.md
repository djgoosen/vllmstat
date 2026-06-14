# vllmstat v0.3 — Fleet / multi-instance monitoring — Design

**Status:** approved forks (2026-06-14) → spec for review
**Goal:** Monitor many vLLM instances from one `vllmstat` process — multiple local Docker
containers (each pinned to different GPUs) *and* remote servers across a network — with an
nvtop-style **overview + drill-in** to the full per-instance dashboard that already exists.

---

## 1. Design invariants (backward compatibility)

- **Single instance stays zero-config and unchanged.** `vllmstat` (default `localhost:8000`) or a
  single `--url` goes **straight to the existing full dashboard** — no overview, no breadcrumb, all
  current bindings identical. Nothing about today's UX regresses.
- **Fleet mode activates only when >1 instance resolves** (from config, repeated `--url`, and/or
  Docker discovery).
- No change to the pure render functions' contracts for the existing panels; drill-in reuses them
  verbatim.

## 2. Instance model

A fleet is an ordered list of `Instance` definitions (identity/config, not live data):

| field | type | meaning |
|---|---|---|
| `name` | `str` | display name; unique. Defaults to container name, or `host:port`. |
| `url` | `str` | base URL. |
| `metrics_path` | `str` | default `/metrics`. |
| `api_key` | `str \| None` | per-instance bearer token; falls back to global. |
| `gpus` | `list[int]` | physical GPU indices this instance uses (local only); `[]` = unknown. |
| `locality` | `"local" \| "remote"` | **auto-classified**: local if URL host ∈ {localhost, 127.0.0.1, ::1, this host's hostname/IPs} or discovered from local Docker; else remote. Config may override with `local = true/false`. |

**Local** instances can show GPU hardware stats (sliced from the host `GpuProvider` by `gpus`).
**Remote** instances show serving metrics only — HTTP `/metrics` carries no GPU hardware data, and
reaching another host's GPUs would need an agent there (explicitly out of scope; see §10).

## 3. Instance sources & merge

Resolved fleet = union of, deduped by normalized URL (later source augments earlier by name):

1. **Config file** — first existing of: `--config PATH`, `$VLLMSTAT_CONFIG`, `./vllmstat.toml`,
   `~/.config/vllmstat/config.toml`. An `[[instance]]` array (§4).
2. **Docker discovery** — when `--discover-docker` is passed (§5). All discovered = local.
3. **`--url` flags** — repeatable, ad-hoc instances.

If none yields an instance → fall back to the single default `http://localhost:8000` (today's
behavior). One resolved instance → single-instance mode. Global flags (`--interval`, `--api-key`,
`--metrics-path`, `--no-gpu`) are defaults applied to instances that don't set their own.

## 4. Config file (TOML)

Parsed with stdlib `tomllib` (3.11+) or `tomli` (3.10; added as a conditional dependency).

```toml
# optional global overrides
interval = 1.0
gpu = true

[[instance]]
name = "qwen-30b"
url  = "http://localhost:8000"
gpus = [0]                         # local → show GPU 0 hardware stats

[[instance]]
name = "llama-70b"
url  = "http://localhost:8001"
gpus = [1]

[[instance]]
name    = "remote-a"
url     = "http://gpu-box-2:8000"  # remote → serving metrics only
api_key = "sk-..."
```

- Each `[[instance]]` requires `url`; `name` optional (derived). Unknown keys ignored
  (forward-compatible). A malformed file fails fast with a clear, single-line error.

## 5. Docker discovery (`--discover-docker`)

No Docker SDK dependency — shell out to the `docker` CLI (consistent with the existing "shell out to
vendor tools" pattern). All parsing is in **pure functions over captured JSON** so it is unit-tested
without a Docker daemon; only a thin subprocess wrapper is impure.

- `docker ps --no-trunc --format '{{json .}}'` → candidate containers. A container is a vLLM
  candidate when its **image** or **command** mentions `vllm` / `api_server` / `openai.api_server`
  (generous; documented).
- `docker inspect <id>` per candidate →
  - **port**: from `.NetworkSettings.Ports` — host port mapped to the serving port (default 8000) →
    `http://localhost:<hostport>`.
  - **gpus**: from `.HostConfig.DeviceRequests[].DeviceIDs` (modern `--gpus`), else
    `NVIDIA_VISIBLE_DEVICES` env, else `--device=/dev/dri/*` (Intel/AMD, best-effort). `"all"` →
    all host GPU indices.
  - **name**: container name.
- Merged with config/flags. If `docker` is absent or errors → log a one-line warning and continue
  (never crash). Best-effort by design; documented as such.

## 6. Concurrent polling

- Per instance: one `InstanceRuntime` bundling `VllmProvider` + `MetricsEngine` + `History` +
  `Instance` + last `Snapshot`. Lazy one-time `model_info`/KV-dims load per instance (as today's
  `_ensure_dims`).
- Each tick: `await asyncio.gather(*(rt.poll(now) for rt in runtimes))` — concurrent fetch, **per
  instance failure isolated** (`VllmProvider` already returns errors as data, so one down instance
  shows `✗` and never breaks the others).
- **GPU**: one host `GpuProvider.sample()` per tick, shared. Each local instance's `Snapshot.gpu` is
  set to the **subset** of host GPUs in its `gpus` list (so drill-in reuses `render.gpu` unchanged);
  the overview shows a compact GPU cell.

## 7. Data shapes

- `Instance` (§2) — new dataclass in `core/state.py`.
- `InstanceRuntime` — new, in a new `core/fleet.py` (owns provider/engine/history; `async poll(now)`).
- `FleetSnapshot` — new dataclass: `ts: float`, `items: list[tuple[Instance, Snapshot]]`,
  `gpu: GpuSnapshot` (host-wide). Single-instance mode still uses a bare `Snapshot`.
- No new fields needed on `Snapshot` — the overview reads existing fields (`running`, `waiting`,
  `gen_tps`, `kv_usage`, `ttft.p50`, `connected`, `gpu`).

## 8. UI — overview + drill-in (Textual screens)

- **OverviewScreen** (default in fleet mode): a single text Panel rendered by a **pure**
  `render.fleet_overview(fleet, selected, width) -> str`, with a `▸` cursor. Columns: name, status
  (`●`/`✗`), running/waiting, gen tok/s, KV%, p50 TTFT, GPU cell (`G0 intel 100%` /
  `G1,2 nvidia 87%` / `(remote)` / `—`). A header line (`fleet · N instances · up …`) and a footer
  hint line. Long fleets scroll in a Textual container (per-row windowing deferred).
  - Bindings: `↑/↓`/`j/k` select · `enter` drill-in · `p` pause · `g` GPU column on/off · `r` reset
    selected instance's session · `+`/`-` interval · `q` quit.
- **DetailScreen**: the existing dashboard panels for the selected instance, with a breadcrumb
  `‹ fleet / <name> @ <url>     esc back`. `esc` pops back to the overview. All current detail
  bindings (`p/g/r/+/-/q`) work; `r` resets that instance.
- **Single-instance mode**: mount the detail dashboard directly (no overview screen, no breadcrumb,
  `esc` is a no-op) — byte-for-byte today's UX.
- Renderers stay pure and string-returning (testable); the app owns selection state and screen
  push/pop.

## 9. CLI / config surface

`Config` grows: `instances: list[Instance]`, `discover_docker: bool`, `config_path: str | None`;
existing scalar fields remain as the single-instance defaults/fallback.

| flag | change |
|---|---|
| `-u/--url` | now **repeatable** (`action="append"`); 0→default, 1→single, >1→fleet |
| `--config PATH` | **new** — explicit config file |
| `--discover-docker` | **new** — merge local Docker vLLM containers |
| `--interval/--api-key/--metrics-path/--no-gpu/--mock/--once/--json/--version` | unchanged (global defaults) |

- `--once --json`: single → unchanged (one object); fleet → **JSON array** of per-instance snapshots
  (each tagged with `name`/`url`).
- `--mock`: fleet mode synthesizes a few mock instances for demos/screenshots.

## 10. Out of scope (later)

- Remote GPU hardware stats (needs a vllmstat agent per host).
- Fleet-wide aggregate panel (a small total in the overview header is OK; full aggregation later).
- Overview sorting/filtering, alert thresholds, k8s/pod discovery (Docker only for v0.3).

## 11. Testing strategy

- **Config**: `tmp_path` TOML files — valid / partial / malformed / precedence — extend
  `tests/test_config.py`.
- **Docker discovery**: captured `docker ps` + `docker inspect` JSON fixtures → pure parsers → assert
  instances + ports + gpus (incl. `NVIDIA_VISIBLE_DEVICES`, `--gpus all`, no-docker path). New
  `tests/test_discover_docker.py`.
- **Fleet polling**: stub providers → `asyncio.gather` poll → assert per-instance isolation (one
  raises/down, others fine). New `tests/test_fleet.py`.
- **Overview render**: `FleetSnapshot` fixtures → assert rows, cursor, remote/down handling, GPU
  cell, None-safety, width. Extend `tests/test_render.py`.
- **App/nav**: fleet vs single mounting; `enter`/`esc` push/pop; bindings. Extend `tests/test_app.py`.
- **Live**: real `localhost:8000` (single, backward compat) + a synthesized 2nd `--url` to exercise
  the fleet overview/drill-in on real data; `--mock` fleet for the screenshot.

## 12. Deliverable

v0.3.0: bump `pyproject.toml` + `__init__.py`; README "Fleet / multi-instance" section (config
example, `--url`, `--discover-docker`, GPU mapping, local-vs-remote, overview/drill-in keys); new
fleet-overview screenshot; full `ruff && pyright && pytest` green.
