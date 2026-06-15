# vllmstat — Tee panel (request feed + content tee) — Design

**Status:** approach approved (2026-06-15) — log-tail request feed **and** proxy content tee, "Both".
**Goal:** A live **TEE** panel under the dashboard showing traffic to a vLLM instance, from two
sources sharing one panel + event model:
- **Phase 1 (v0.4.0) — log request feed:** tail the server's logs (Docker container or file) → a live
  feed of HTTP requests (method, path, status, client). Zero reconfiguration. **No prompt/response
  text** — modern vLLM (V1) does not log content (verified live: a probed request logged only
  `"POST /v1/chat/completions HTTP/1.1" 200 OK`).
- **Phase 2 (v0.5.0) — proxy content tee:** vllmstat runs a small reverse proxy; clients point at it;
  it forwards to vLLM and tees **full prompts + (streamed) completions**. The only way to see content.

Each phase ships independently. The TEE event model + renderer are designed once to serve both.

---

## 1. Shared model

```python
@dataclass
class TeeEvent:
    ts: float                       # wall-clock (time.time()) for HH:MM:SS display
    kind: str                       # "http" | "exchange" | "note"
    # kind == "http" (log feed)
    method: str | None = None
    path: str | None = None
    status: int | None = None
    client: str | None = None
    # kind == "exchange" (proxy)
    endpoint: str | None = None     # "chat" | "completions" | raw path
    prompt: str | None = None
    response: str | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    streaming: bool = False
    done: bool = True               # streaming: False until [DONE]
    # kind == "note" (status/error from the source itself)
    text: str | None = None
```

- `TeeBuffer` — a bounded ring (`collections.deque(maxlen=500)`); `push(event)`; `recent(n)`.
- One `TeeBuffer` **per instance**, owned by its `InstanceRuntime`.

## 2. Renderer — `render.tee(events, *, width, height, source_desc="") -> str`

- Title: `TEE · {source_desc}` (e.g. `docker:vllm-xpu`, `proxy :9000`, or `—`).
- Render the last `height-1` events, **newest at the bottom** (tail -f feel). Each event one or two
  lines, truncated to `width` with `…`, whitespace/newlines collapsed to a single line:
  - `http`: `{hh:mm:ss}  {method:<5} {path}  {status}` (4xx/5xx prefixed with `!`).
  - `exchange`: `▶ {prompt}` then `◀ {response}` (`◀ … ▌` while `not done`).
  - `note`: `· {text}` (e.g. `· docker logs unavailable — is the container running?`).
- Pure function, returns a plain string (consistent with the other renderers). Empty buffer →
  `TEE · {source}\n (waiting for requests…)`.

## 3. Panel placement & toggle

- A `Panel(id="tee")` in the **detail** view, below the GPU panel. Fixed CSS height (~10 rows).
- Shown only when the **current** instance has a tee source configured **and** the panel is toggled
  on. Key **`t`** toggles it (default on when a source exists). In fleet mode the panel reflects the
  drilled-in instance; the overview is unchanged.

## 4. Phase 1 — log request feed

### Source spec (a string on the instance)
- `docker:NAME` → `docker logs -f --since 0s NAME` (new lines only), read stdout+stderr.
- a filesystem path → follow like `tail -f` (seek to end; reopen on rotation/missing).
- Set via `--logs SOURCE` (applies to the single/default instance), per-instance `logs = "..."` in the
  config file, **or auto-set by `--discover-docker`** (`logs = "docker:<container>"`).

### `providers/logsource.py`
- `parse_access_line(line) -> TeeEvent | None` — pure. Matches the uvicorn access format seen live:
  `(APIServer pid=1) INFO:     172.18.0.1:47402 - "POST /v1/chat/completions HTTP/1.1" 200 OK`
  Extract method, path, status, client (`ip:port` → ip). Return `None` for non-access lines
  (throughput stats, startup, etc.). **Filter** noise paths by default: `/health`, `/metrics`,
  `/ping`, `/v1/models` → `None`.
- `class LogTailer` — async, best-effort, never raises:
  - `__init__(source, *, on_event)`; `async run()` loops yielding parsed events to `on_event`.
  - docker: `asyncio.create_subprocess_exec("docker","logs","-f","--since","0s",name,...)`; on spawn
    failure push a `note` event and stop.
  - file: async readline-with-EOF-poll; reopen on `FileNotFoundError`/inode change.
  - `async stop()` — cancel task, terminate the subprocess.

### Wiring
- `Instance.logs: str | None = None`; `Config.logs: str | None = None`; `--logs` flag;
  `instance_from_dict` reads `logs`; `discover_docker` sets `logs=docker:<name>`.
- `InstanceRuntime` gains `tee: TeeBuffer` and (if `instance.logs`) a `LogTailer` started by the app
  on mount, `on_event=self.tee.push`; stopped on app exit.
- Detail view renders the TEE panel for the shown instance; `t` toggles.

### Out of scope (Phase 1)
Prompt/response content (needs Phase 2); per-request latency from logs (not logged); fleet-wide
aggregated tee.

## 5. Phase 2 — proxy content tee

### CLI / behaviour
- `--proxy [HOST:]PORT` (e.g. `--proxy 9000`). Enables a reverse proxy → the single instance's
  upstream URL. (Fleet = one proxy → one upstream for v0.5; multi-upstream later.)
- Insertion point for this box: set **open-webui**'s OpenAI base URL to `http://<host>:9000`.

### Dependency
- The proxy needs an async HTTP **server**; the base install stays light. Add an **optional extra**:
  `pip install vllmstat[proxy]` → `aiohttp`. `--proxy` without it exits with a clear
  `pip install 'vllmstat[proxy]'` hint. Forwarding reuses the existing `httpx` client (streaming).

### `providers/proxy.py`
- `class TeeProxy(upstream_url, host, port, *, on_event)`; `async start()/stop()`.
- Per request: capture method/path/headers/body → forward to upstream (streaming) → relay the
  response to the client **unchanged** (the client's traffic is sacred; any capture error must not
  corrupt it).
- For `/v1/chat/completions` & `/v1/completions`: extract the prompt (chat `messages` → last/all user
  content; completions `prompt`); push an `exchange` event; **accumulate** the response — SSE
  `data: {...delta.content...}` until `data: [DONE]`, updating the event in place (`done=False` →
  `True`); non-streaming JSON → parse `choices[].message.content` / `.text`. Capture token counts from
  `usage` when present.
- Other paths: transparent proxy; optionally emit an `http` event so the feed still shows them.

### Out of scope (Phase 2)
Multi-upstream/fleet proxy; request replay/editing; redaction (note: prompts/responses render in your
local terminal only — documented as a privacy consideration).

## 6. Testing

- **Phase 1:** `parse_access_line` over captured real lines (incl. the live `POST /v1/chat/completions
  … 200 OK`, health-filter, 4xx, IPv6 client, non-access → None); `LogTailer` over a temp file (write
  lines → assert events; rotation) and a stubbed `run` for docker; `render.tee` (http/exchange/note,
  truncation, newest-last, empty); app: an instance with `logs` mounts a tailer and `t` toggles the
  panel; `discover_docker` sets `logs`.
- **Phase 2:** `parse` of request bodies (chat/completions) + SSE delta accumulation as pure
  functions over captured payloads; a proxy round-trip against a stub upstream (aiohttp test server)
  asserting passthrough + captured exchange + streaming aggregation; missing-aiohttp → friendly exit.
- Live: Phase 1 against `docker:vllm-xpu` (real access feed); Phase 2 by routing a curl/open-webui
  call through `--proxy` and seeing the real prompt+response.

## 7. Deliverables
- **v0.4.0** = Phase 1 (log request feed) + README "Tee / request feed" section + `t` binding + screenshot.
- **v0.5.0** = Phase 2 (proxy content tee) + `vllmstat[proxy]` extra + README update.
