# Pilot-study freeze baseline

This is the fixed configuration of the UCANRR Safety Evaluation API for the pilot
study. Nothing in this table changes for the duration of the study. The running
service reports the same identifiers at `GET /version`.

**Deployed module:** [`azure_ucanrr_safety_eval_api.py`](azure_ucanrr_safety_eval_api.py)
(OpenAI-only, single LLM call). The SASI + OpenAI pipeline described in
[`docs/SAFETY_PIPELINE_AUDIT.md`](docs/SAFETY_PIPELINE_AUDIT.md) and
[`docs/VERIFICATION_RUN.md`](docs/VERIFICATION_RUN.md) (`ucanrr_sasi_safety_eval_api.py`)
is **not** deployed and is not part of this baseline.

**Live host:** `https://safetyapi-c6cqctbghub5f5d8.canadacentral-01.azurewebsites.net`
· Azure Web App `safetyapi` · region `canadacentral` · Python 3.11 ·
`gunicorn -w 4 -k uvicorn.workers.UvicornWorker azure_ucanrr_safety_eval_api:app`

| Component | Identifier at freeze | Status |
|---|---|---|
| **Application build** | Git tag `pilot-freeze-2026-09-08` → commit `__________` (fill after merge). Deploy: GitHub Actions run `__________` (fill from Actions URL). Runtime build identity is written to `BUILD_INFO.txt` by CI and echoed at `GET /version` → `build.commit` / `build.run_id` / `build.built_at`. | Set on merge + deploy |
| **Model** | `gpt-4o-2024-08-06` — dated snapshot, pinned. Code default in [`azure_ucanrr_safety_eval_api.py`](azure_ucanrr_safety_eval_api.py); Azure App Setting `OPENAI_MODEL` **must be set to the same value**. Call parameters (frozen): `temperature=0`, `response_format` = strict `json_schema`, `store=false`, `safety_identifier` = `user_hash` when supplied, client `timeout=25s`, `max_retries=2`. SDK: `openai==2.9.0` (pinned in `requirements.txt`; full resolved set captured in CI as `BUILD_PIP_FREEZE.txt`). | **Confirm snapshot** — see note 1 |
| **Prompt version** | `2026-07-01` — `SYSTEM_PROMPT` constant in [`azure_ucanrr_safety_eval_api.py`](azure_ucanrr_safety_eval_api.py). SHA-256 `c2cc265c8854a0e64f17225e87dd96cddba7145ac5bc22bdabfdbb71fc55acc2`. Last substantive change: commit `cbada3e` ("edge case prompt update"). Exposed at `/version` → `prompt_version` / `prompt_sha256`. | Frozen |
| **Pipeline version** | `1.2.1-single-stage-openai`. Flow: input validation → one `chat.completions.create` call (client `timeout=25s`) with strict JSON schema → `json.loads` of the model output → return as `SafetyAssessment`. No pre-moderation, no SASI, no cache, no persistence of entry text. No application-level retry; the OpenAI SDK performs up to 2 automatic retries on connection errors / 429 / 5xx. Routes: `POST /safety/analyze_entry`, `GET /health`, `GET /version`. | Frozen (bumped from `1.2.0` — see note 5) |
| **Safety-configuration version** | `2026-07-01`. Defined entirely by: (a) `SYSTEM_PROMPT` tiering rules (hash above); (b) `SAFETY_JSON_SCHEMA` strict enum constraints — SHA-256 `a5c993c4ca528d0c05bcc876ab5546979194e7d35e24fa57d73cf8a24fc6a33d`; (c) call params `temperature=0` / `store=false` / `safety_identifier` passthrough; (d) OpenAI platform default moderation. No separate safety-config file. `config/sasi.yaml` is an unused template. Exposed at `/version` → `safety_config_version` / `safety_schema_sha256`. | Frozen |
| **Behavior when entry processing fails** | Full write-up: [`docs/ENTRY_FAILURE_BEHAVIOR.md`](docs/ENTRY_FAILURE_BEHAVIOR.md). Summary: malformed body → **HTTP 422**; empty / whitespace `entry_text` → **HTTP 400** `{"detail":"entry_text must not be empty."}` (no model call); `OPENAI_API_KEY` missing at request time → **HTTP 500** `{"detail":"OPENAI_API_KEY is not configured on the server."}`. Any exception in the OpenAI call or in `json.loads()` of the model output → caught by a broad `except Exception`, logged via `logger.exception`, returns **HTTP 500** `{"detail":"Error calling OpenAI safety model: <exception>"}`. A `SafetyAssessment(**...)` response-model **validation** error is **not** inside that `try` → uncaught → generic **HTTP 500 "Internal Server Error"** (in practice prevented by the strict `json_schema`). The OpenAI call is bounded by an explicit **25 s** client timeout with **2** SDK retries, kept under the gunicorn `--timeout 120` worker limit so a slow call returns a 500 body rather than a 502; a request that still exceeds 120 s is SIGKILLed by gunicorn → **HTTP 502**, no body. **No application-level retry, no fallback assessment, no default/fail-closed risk tier** — a failed entry produces no assessment and the caller receives a 4xx/5xx. Logging: stdlib logging (INFO) to stdout/stderr; Azure Application Insights only if `APPLICATIONINSIGHTS_CONNECTION_STRING` is set. `store=false` → OpenAI does not retain the request; the API persists no entry text. | Documented from implementation |

## Notes / actions to close out the freeze

1. **Confirm the model snapshot.** The batch validation runs (2026-09-06,
   `results/mindguard_result_20260906_113950.*`) were made against the floating
   `gpt-4o` alias. Confirm which dated snapshot that alias resolved to on that
   date (OpenAI dashboard → usage/logs, or a one-off API call inspecting the
   response `model` field) and, if it was not `gpt-4o-2024-08-06`, change the
   pin in `requirements.txt`-adjacent code default and the Azure App Setting to
   match, then re-run the batch before tagging.
2. **Set the Azure App Setting** `OPENAI_MODEL = gpt-4o-2024-08-06` (or the
   confirmed snapshot) explicitly — do not rely on the code default.
3. **After merge + successful deploy:** curl `/version`, paste `build.commit`
   and the Actions run URL into the table above, then
   `git tag pilot-freeze-2026-09-08 <merge-commit> && git push origin pilot-freeze-2026-09-08`.
4. **Re-run batch validation** against the frozen deploy and confirm parity with
   the 2026-09-06 results (the `openai` SDK pin may differ from what Azure built
   with previously). Archive `BUILD_PIP_FREEZE.txt` from the deploy run.
5. **Pipeline `1.2.0` → `1.2.1`:** added an explicit OpenAI client
   `timeout=25s` / `max_retries=2` so a slow model call returns a clean HTTP 500
   instead of a gunicorn-timeout 502. No change to the prompt, JSON schema, or
   output-affecting call parameters (`temperature`, `response_format`, `store`),
   so `SAFETY_CONFIG_VERSION` and `PROMPT_VERSION` are unchanged and prior batch
   results remain comparable. Re-confirm on the frozen deploy per note 4.
