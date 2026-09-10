# Behavior when entry processing fails

**Scope:** the deployed service `azure_ucanrr_safety_eval_api.py`
(`POST /safety/analyze_entry`), single-stage OpenAI pipeline
`1.2.1-single-stage-openai`, as run on Azure App Service with
`gunicorn -w 4 -k uvicorn.workers.UvicornWorker
azure_ucanrr_safety_eval_api:app --timeout 120`.

Documented from implementation — code references are `file:line` against the
commit this document ships with. Update on any change to the request handler,
the startup command, the OpenAI client construction, or `requirements.txt`
(OpenAI SDK pin).

---

## 1. Summary

The pipeline is **fail-with-error, never fail-safe**. Every failure path returns
an HTTP 4xx/5xx (or, at the infra layer, drops the connection). On failure the
caller receives **no `SafetyAssessment`**: there is no fallback object, no
default risk tier, no "assume crisis on error" behavior, and no application-level
retry. The request is not queued, persisted, or dead-lettered — from the
server's perspective a failed entry is simply lost, and it is the **caller's
responsibility** to decide what the UCANRR UI does when the safety check did not
complete.

---

## 2. Failure modes, in the order they are reached

### 2.1 Request validation — before any model call

| Condition | Response | Model call? | Where |
|---|---|---|---|
| Body is not valid JSON, `entry_text` missing, or wrong type | **HTTP 422** `{"detail":[{...}]}` (FastAPI/Pydantic default `RequestValidationError`; the handler body never runs) | No | `JournalEntryRequest` — `azure_ucanrr_safety_eval_api.py:152` |
| `entry_text` present but empty or whitespace-only | **HTTP 400** `{"detail":"entry_text must not be empty."}` | No | handler guard, `:508` |
| `user_hash` omitted | Not a failure — `safety_identifier` is simply not sent to OpenAI | Yes | call site, `:524` |

### 2.2 Server configuration — checked per request

| Condition | Response | Where |
|---|---|---|
| `OPENAI_API_KEY` not set in the environment at request time | **HTTP 500** `{"detail":"OPENAI_API_KEY is not configured on the server."}`, `logger.error(...)` | handler guard, `:512` |

The `OpenAI(...)` client is constructed at import with `api_key=_api_key or ""`
(`:100`), so a missing key does **not** crash startup — it is caught by the guard
above, or (if that guard were bypassed) surfaces as an auth exception inside the
`try` block in §2.3. A startup warning is logged (`:89`).

### 2.3 Model call and output parsing — inside `try/except` (`:515`–`:536`)

A single broad `except Exception` (`:534`) wraps: the `chat.completions.create`
call, reading `completion.choices[0].message.content`, and `json.loads(...)`
(`:530`) of that content. Any exception from those steps is logged with
`logger.exception("OpenAI call failed: %s", e)` (`:535`) and returned as:

> **HTTP 500** `{"detail":"Error calling OpenAI safety model: <str(exception)>"}`

Covered by this path:

| Condition | Notes |
|---|---|
| OpenAI auth / permission error | e.g. bad or revoked key |
| OpenAI rate limit — **after** the SDK's own retries are exhausted | see §3 |
| OpenAI 5xx / overloaded — after SDK retries | see §3 |
| OpenAI request timeout (25 s per attempt) / connection error — after SDK retries | see §3 |
| Model returns non-JSON content | `json.loads` raises `JSONDecodeError`. Near-impossible in practice because `response_format` is strict `json_schema` |
| `message.content` is `None` (refusal / filtered) | `json.loads(None)` raises `TypeError`, same 500 path |

The returned `detail` string contains `str(exception)`, which for a JSON parse
failure can include a fragment of the **model's output** (not the user's entry).
This is a minor internal-detail disclosure to the caller and appears in logs.

### 2.4 Response validation — OUTSIDE the `try/except`

`return SafetyAssessment(**assessment_dict)` (`:538`) is **after** the `except`
block (`:534`). If
the parsed dict fails Pydantic validation (`risk_tier` outside 0–3, a required
key absent), the resulting `ValidationError` is **not** caught by the handler.
FastAPI's default handler returns a generic **HTTP 500 "Internal Server Error"**
(no `detail` message), and the ASGI server logs the traceback.

In practice this is prevented by the strict `json_schema` for `risk_tier` and
the required keys; `flags` and `recommendations` are typed `Dict[str, Any]`
(`:163`–`:164`) and are therefore **not** deeply validated by Pydantic.

### 2.5 Infrastructure / hosting — outside the application

| Condition | Client sees | Notes |
|---|---|---|
| Single request exceeds **120 s** | **Connection dropped / HTTP 502** from the Azure front end — *not* a JSON 500 | `gunicorn --timeout 120` SIGKILLs the worker mid-request. Since pipeline `1.2.1` the OpenAI call is bounded at 25 s with 2 retries (§3, worst case ~90 s), so under normal operation this path is not reached; it remains possible if the process stalls **outside** the OpenAI call. This was the "hung, then failed" symptom seen in batch testing against pipeline `1.2.0` (unbounded 600 s SDK timeout). |
| Cold start / restart after deploy, scale, or idle unload | **HTTP 503 Service Unavailable** (Azure platform page) for ~1–3 min | Transient; the app is not yet accepting connections. |
| Overlapping deployments | Deploy step fails; requests in that window may 503 | Mitigated in CI by a `concurrency` group in the deploy workflow. |

---

## 3. Retry and timeout — exact behavior

The **application performs no retry of its own.** Timeout and retry are set
explicitly on the OpenAI client at construction (`:96`–`:103`) and reported at
`GET /version` → `llm_call_params`:

| Parameter | Value | Rationale |
|---|---|---|
| `timeout` | **25 s** per attempt (all of connect/read/write/pool) | Model latency for a single entry is ~2–5 s; 25 s is generous headroom. |
| `max_retries` | **2** (SDK default, pinned explicitly) | Exponential backoff on connection errors, 408, 409, 429, ≥ 500. |

- **Worst case** = `25 s × (2 + 1)` + backoff ≈ **~90 s**, which stays under the
  gunicorn `--timeout 120` worker limit — so a slow or failing call returns a
  clean **HTTP 500 with a `detail` body** instead of a SIGKILLed worker / 502.
- After the 2 retries are exhausted the exception propagates to the handler's
  `except` (§2.3) and becomes **one** HTTP 500. The caller sees a single failed
  response, not the intermediate retries.

Prior to pipeline `1.2.1` the client used SDK defaults
(`max_retries=2`, `timeout` = connect 5 s / read 600 s). The 600 s read timeout
exceeded gunicorn's 120 s, so a stalled model call produced a 502 with no body.
Fixed by pinning `timeout=25 s`.

---

## 4. What the service never does on failure

- **No fallback assessment.** No default `SafetyAssessment`, no default
  `risk_tier`, no "fail closed to tier 3 / fail open to tier 0."
- **No fail-safe UI directive.** The caller gets a 4xx/5xx and must decide what
  the UCANRR UI shows (e.g. block the share, show a generic "we couldn't check
  this right now" message, retry later).
- **No queue, no dead-letter, no persistence of the entry.** `store=false` tells
  OpenAI not to retain the request; the API writes no entry text to disk or
  database. On failure, `logger.exception` may write a fragment of the *model
  output* (not the user entry) to logs, and the 500 `detail` returns
  `str(exception)` to the caller.
- **No idempotency key / no automatic client-side replay.** A lost request is
  lost unless the caller re-submits.

---

## 5. Caller-side behavior — reference batch client

The only implemented consumer in this repository is the evaluation harness
`test_mindguard_batch1v2.py`. The production UCANRR application client is out of
scope for this document.

- HTTP timeout `(connect = 10 s, read = 60 s)`.
- Retries on `requests.RequestException` (connection reset, read timeout,
  chunked-encoding error) up to `--max-retries` (default 5), exponential
  backoff capped at 30 s.
- Retries on **HTTP 429** up to `--max-retries`, honoring `Retry-After`.
- Any other non-200 → `RuntimeError("API error <status>: <body>")`.
- **Batch level:** a row that raises is caught, written to the output CSV with
  the `api_error` column populated and all assessment columns blank, and the run
  **continues**. The CSV is flushed after every row, so a crash or Ctrl-C keeps
  all completed rows; the run resumes with `--start <n>`.
- Net effect: one failed entry never aborts the batch and is always visible in
  the results as a row with `api_error` set.

---

## 6. Design gaps still open

Listed for the feasibility decision — not blockers for the frozen build:

1. **Catch response validation.** Move `SafetyAssessment(**...)` inside the
   `try`, or register a `ValidationError` handler, so a malformed model output
   produces the same logged, structured error shape as every other failure
   instead of a bare 500.
2. **Decide the UI contract for "check did not complete."** The API is
   fail-with-error by design; the product needs an explicit, documented rule for
   what the share flow does on a 4xx/5xx/timeout (recommended: block partner
   share, allow therapist share, surface a retry).
3. **Stop returning `str(exception)` to the caller.** Return a static message and
   keep the detail in logs only.

**Closed in pipeline `1.2.1`:** explicit `timeout=25 s` / `max_retries=2` on the
OpenAI client so a slow call returns a clean HTTP 500 within the gunicorn 120 s
worker limit (previously a 502 with no body).
