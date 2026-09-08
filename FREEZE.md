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
| **Model** | `gpt-4o-2024-08-06` — dated snapshot, pinned. Code default in [`azure_ucanrr_safety_eval_api.py`](azure_ucanrr_safety_eval_api.py); Azure App Setting `OPENAI_MODEL` **must be set to the same value**. Call parameters (frozen): `temperature=0`, `response_format` = strict `json_schema`, `store=false`, `safety_identifier` = `user_hash` when supplied. SDK: `openai==2.9.0` (pinned in `requirements.txt`; full resolved set captured in CI as `BUILD_PIP_FREEZE.txt`). | **Confirm snapshot** — see note 1 |
| **Prompt version** | `2026-07-01` — `SYSTEM_PROMPT` constant in [`azure_ucanrr_safety_eval_api.py`](azure_ucanrr_safety_eval_api.py). SHA-256 `c2cc265c8854a0e64f17225e87dd96cddba7145ac5bc22bdabfdbb71fc55acc2`. Last substantive change: commit `cbada3e` ("edge case prompt update"). Exposed at `/version` → `prompt_version` / `prompt_sha256`. | Frozen |
| **Pipeline version** | `1.2.0-single-stage-openai`. Flow: input validation → one `chat.completions.create` call with strict JSON schema → `json.loads` of the model output → return as `SafetyAssessment`. No pre-moderation, no SASI, no retry, no cache, no persistence of entry text. Routes: `POST /safety/analyze_entry`, `GET /health`, `GET /version`. | Frozen |
| **Safety-configuration version** | `2026-07-01`. Defined entirely by: (a) `SYSTEM_PROMPT` tiering rules (hash above); (b) `SAFETY_JSON_SCHEMA` strict enum constraints — SHA-256 `a5c993c4ca528d0c05bcc876ab5546979194e7d35e24fa57d73cf8a24fc6a33d`; (c) call params `temperature=0` / `store=false` / `safety_identifier` passthrough; (d) OpenAI platform default moderation. No separate safety-config file. `config/sasi.yaml` is an unused template. Exposed at `/version` → `safety_config_version` / `safety_schema_sha256`. | Frozen |
| **Behavior when entry processing fails** | Empty / whitespace `entry_text` → **HTTP 400** `{"detail":"entry_text must not be empty."}`, no model call. `OPENAI_API_KEY` missing at request time → **HTTP 500** `{"detail":"OPENAI_API_KEY is not configured on the server."}`. Any exception in the OpenAI call, in `json.loads()` of the model output, or in `SafetyAssessment(**...)` validation → caught by a broad `except Exception`, logged via `logger.exception`, returns **HTTP 500** `{"detail":"Error calling OpenAI safety model: <exception>"}`. **No retry, no client-side timeout override, no fallback assessment, no default/fail-closed risk tier** — a failed entry produces no assessment and the caller receives a 5xx. Logging: stdlib logging (INFO) to stdout/stderr; Azure Application Insights only if `APPLICATIONINSIGHTS_CONNECTION_STRING` is set. `store=false` → OpenAI does not retain the request; the API persists no entry text. | Documented from implementation |

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
