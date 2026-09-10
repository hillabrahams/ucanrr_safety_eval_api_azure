# Behavior when entry processing fails

**Scope:** the deployed service `azure_ucanrr_safety_eval_api.py`
(`POST /safety/analyze_entry`), single-stage OpenAI pipeline, as run on Azure
App Service with `gunicorn -w 4 -k uvicorn.workers.UvicornWorker
azure_ucanrr_safety_eval_api:app --timeout 120`.

Documented from implementation — code references are `file:line` against the
commit this document ships with. Update on any change to the request handler,
the startup command, or `requirements.txt` (OpenAI SDK pin).

---

## 1. Summary

The pipeline is **fail-with-error, never fail-safe**. Every failure path returns
an HTTP 4xx/5xx (or, at the infrastructure layer, drops the connection). On
failure the caller receives **no `SafetyAssessment`**: there is no fallback
object, no default risk tier, no "assume crisis on error" behavior, and no
retry beyond what the OpenAI SDK does internally. The request is not queued,
persisted, or dead-lettered — from the server's perspective a failed entry is
simply lost, and it is the **caller's responsibility** to decide what the UCANRR
UI does when the safety check did not complete.

---

## 2. Failure modes, in the order they are reached

### 2.1 Request validation — before any model call

| Condition | Response | Model call? | Where |
|---|---|---|---|
| Body is not valid JSON, `entry_text` missing, or wrong type | **HTTP 422** `{"detail":[{...}]}` (FastAPI/Pydantic default `RequestValidationError`; the handler body never runs) | No | `JournalEntryRequest` — `azure_ucanrr_safety_eval_api.py:140` |
| `entry_text` present but empty or whitespace-only | **HTTP 400** `{"detail":"entry_text must not be empty."}` | No | `azure_ucanrr_safety_eval_api.py:496` |
| `user_hash` omitted | Not a failure — `safety_identifier` is simply not sent to OpenAI | Yes | `azure_ucanrr_safety_eval_api.py:512` |

### 2.2 Server configuration — checked per request

| Condition | Response | Where |
|---|---|---|
| `OPENAI_API_KEY` not set in the environment at request time | **HTTP 500** `{"detail":"OPENAI_API_KEY is not configured on the server."}`, `logger.error(...)` | `azure_ucanrr_safety_eval_api.py:499` |

The `OpenAI(...)` client is constructed at import with `api_key=_api_key or ""`
(`:91`), so a missing key does **not** crash startup — it is caught by the guard
above, or (if that guard were bypassed) surfaces as an auth exception inside the
`try` block in §2.3. A startup warning is logged (`:89`).

### 2.3 Model call and output parsing — inside `try/except` (`:503`–`:524`)

A single broad `except Exception` (`:522`) wraps: the `chat.completions.create`
call, reading `completion.choices[0].message.content`, and `json.loads(...)` of
that content. Any exception from those steps is logged with
`logger.exception("OpenAI call failed: %s", e)` and returned as:

> **HTTP 500** `{"detail":"Error calling OpenAI safety model: <str(exception)>"}`

Covered by this path:

| Condition | Notes |
|---|---|
| OpenAI auth / permission error | e.g. bad or revoked key |
| OpenAI rate limit — **after** the SDK's own retries are exhausted | see §3 |
| OpenAI 5xx / overloaded — after SDK retries | see §3 |
| OpenAI request timeout / connection error — after SDK retries | see §3 |
| Model returns non-JSON content | `json.loads` raises `JSONDecodeError`. Near-impossible in practice because `response_format` is strict `json_schema` (`:511`) |
| `message.content` is `None` (refusal / filtered) | `json.loads(None)` raises `TypeError`, same 500 path |

The returned `detail` string contains `str(exception)`, which for a JSON parse
failure can include a fragment of the **model's output** (not the user's entry).
This is a minor internal-detail disclosure to the caller and appears in logs.

### 2.4 Response validation — OUTSIDE the `try/except`

`return SafetyAssessment(**assessment_dict)` is at `:526`, **after** the
`except` block. If the parsed dict fails Pydantic validation (`risk_tier`
outside 0–3, a required key absent), the resulting `ValidationError` is **not**
caught by the handler. FastAPI's default handler returns a generic
**HTTP 500 "Internal Server Error"** (no `detail` message), and the ASGI server
logs the traceback.

In practice this is prevented by the strict `json_schema` for `risk_tier` and
the required keys; `flags` and `recommendations` are typed `Dict[str, Any]`
(`:151`–`:152`) and are therefore **not** deeply validated by Pydantic.

> **Implementation note / discrepancy:** the `FREEZE.md` "Behavior when entry
> processing fails" row states that a `SafetyAssessment(**...)` validation error
> is caught by the broad `except`. It is not — see above. Either move the
> `return` inside the `try`, or correct `FREEZE.md`.

### 2.5 Infrastructure / hosting — outside the application

| Condition | Client sees | Notes |
|---|---|---|
| Single request exceeds **120 s** (SDK retries + model latency) | **Connection dropped / HTTP 502** from the Azure front end — *not* a JSON 500 | `gunicorn --timeout 120` SIGKILLs the worker mid-request. The SDK's own read timeout is 600 s (§3), so **gunicorn's 120 s is the effective ceiling**. This is the "hung, then failed" symptom observed in batch testing when the service was slow. |
| Cold start / restart after deploy, scale, or idle unload | **HTTP 503 Service Unavailable** (Azure platform page) for ~1–3 min | Transient; the app is not yet accepting connections. |
| Overlapping deployments | Deploy step fails; requests in that window may 503 | Mitigated in CI by a `concurrency` group in the deploy workflow. |

---

## 3. Retry and timeout — exact behavior

The **application adds no retry logic and does not override the client timeout.**
The behavior below comes entirely from the pinned SDK, `openai==2.9.0`
(`requirements.txt`):

- **Automatic retries:** up to **2**, with exponential backoff, on connection
  errors, 408, 409, 429, and ≥ 500. (`openai.DEFAULT_MAX_RETRIES = 2`.)
- **Timeout:** `connect = 5 s`, `read = 600 s`, `write = 600 s`, `pool = 600 s`.
  (`openai.DEFAULT_TIMEOUT`.)
- After the 2 retries are exhausted, the exception propagates to the handler's
  `except` (§2.3) and becomes **one** HTTP 500. The caller sees a single failed
  response, not the intermediate retries.
- Because gunicorn kills the worker at 120 s (§2.5), the 600 s SDK read timeout
  is never actually reached in the deployed configuration.

---

## 4. What the service never does on failure

- **No fallback assessment.** No default `SafetyAssessment`, no default
  `risk_tier`, no "fail closed to tier 3 / fail open to tier 0."
- **No fail-safe UI directive.** The caller gets a 4xx/5xx and must decide what
  the UCANRR UI shows (e.g. block the share, show a generic "we couldn't check
  this right now" message, retry later).
- **No queue, no dead-letter, no persistence of the entry.** `store=false`
  (`:513`) tells OpenAI not to retain the request; the API writes no entry text
  to disk or database. On failure, `logger.exception` may write a fragment of
  the *model output* (not the user entry) to logs, and the 500 `detail` returns
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

## 6. Recommendations (not yet implemented)

These are design gaps surfaced by the audit, listed for the feasibility
decision — none are in the frozen build:

1. **Align the timeouts.** Set an explicit OpenAI client timeout below gunicorn's
   120 s (e.g. `OpenAI(timeout=90)`) so a slow call returns a clean HTTP 500
   with a `detail` body instead of a 502 with a dropped connection.
2. **Catch response validation.** Move `SafetyAssessment(**...)` inside the
   `try`, or add a `RequestValidationError` / `ValidationError` handler, so a
   malformed model output produces a consistent, logged error shape.
3. **Decide the UI contract for "check did not complete."** The API is
   fail-with-error by design; the product needs an explicit, documented rule for
   what the share flow does on a 4xx/5xx/timeout (recommended: block partner
   share, allow therapist share, surface a retry).
4. **Stop returning `str(exception)` to the caller.** Return a static message and
   keep the detail in logs only.
