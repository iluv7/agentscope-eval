# Tool-loading benchmark

This benchmark evaluates the reliability of AgentScope's proposed dynamic tool-loading path. It scores captured observations using strict JSON parsing, JSON Schema validation, and deterministic assertions. It never calls a model, repairs arguments, searches a registry on behalf of an agent, or executes a tool.

The first experiment is **nested argument generation**: compare native calls with an `execute_tool` envelope while giving the model the required schema directly. The second experiment adds discovery and real execution evidence to evaluate all four checkpoints.

## Run the included fixture

Run these commands from the repository root. All suite-specific code, tests, examples, and documentation live in this directory. The shared service discovers this suite through `api.py`; `__main__.py` provides its standalone command.

```bash
uv sync --locked
uv run python -m evaluations.tool_loading evaluations/tool_loading/examples/fixture.json \
  --output /tmp/tool-loading-report.json \
  --markdown /tmp/tool-loading-report.md
```

The fixture contains 18 synthetic trials covering escaping, Unicode, code strings, arrays, nested objects, long strings, repairs, retries, missing calls, search misses, and tool errors. **Its numbers test the evaluator; they are not model benchmark results.** Each configuration is labeled `source: fixture`. Regenerate it with:

```bash
python evaluations/tool_loading/examples/generate_fixture.py
```

For recorded experiments, replace the fixture observations and configuration metadata with actual captures and use `source: recorded`. `--fail-on-failure` returns exit code 1 when any trial fails or errors. Invalid input or an output-file error returns 2; otherwise report generation returns 0.

The same evaluator is available through the API:

```bash
curl -s http://127.0.0.1:8787/v1/benchmarks/tool-loading/evaluate \
  -H 'Content-Type: application/json' \
  --data-binary @evaluations/tool_loading/examples/fixture.json
```

Malformed **model argument strings** produce failed checkpoint results in an HTTP 200 report. Invalid benchmark configuration or inconsistent capture records return 422. Use this endpoint for raw reliability experiments; the existing `/v1/agentscope/evaluate` endpoint still expects normalized, parseable tool arguments for its answer-quality metrics.

## Experimental controls

`configuration.variant` defines the calling convention:

| Variant | Generated entry point | Schema availability |
|---|---|---|
| `native` | The real tool name, with its argument object | Tool declared natively |
| `dispatcher` | `execute_tool` | Real schema provided without requiring search |
| `discovery` | `search_tools`, then `execute_tool` | Real schema returned by search |

The dispatcher uses this exact envelope, with an object-valued `arguments` field:

```json
{"tool_name": "save", "arguments": {"text": "hello"}}
```

The native equivalent calls `save` with `{"text":"hello"}`. In both variants, the captured `RawCall.arguments` is the provider's original JSON **string**. Storing that string inside the benchmark file necessarily escapes it; this storage representation is not evidence of double serialization by the model. A generated envelope whose inner `arguments` value is itself a string fails envelope validation.

Set `configuration.evaluation_scope` to `generation` for the initial argument-only experiment. Execution is then `not_applicable`, and supplying execution evidence is rejected to avoid silently discarding it. Set it to `trajectory` for the full pipeline; supply `expected_output` explicitly, including `null` when that is the expected result.

Keep provider, model/version, AgentScope version, schema placement, structured-output settings, temperature, seed, catalog version, and prompt version in `configuration`. Every distinct configuration is summarized separately, also broken down by `scenario`. Use the same cases and repeated trials across variants. `repetition` identifies repeated observations; `run_id` must remain unique.

Changes in schema placement and native output constraints are potential confounders. A difference between native and dispatcher results alone does not prove that nesting caused it. The evaluator reports observations, not statistical significance or confidence intervals.

## Input contract

`tools` is the authoritative registry: each entry contains `name` and `input_schema`. Schemas use JSON Schema draft 2020-12. Local references and `$defs` are supported; external references, unresolved references, other dialects, and nested `$id` scopes are rejected. Validation checks schema constraints without coercion; `format` remains an annotation.

Each trial contains:

- `run_id`, `case_id`, the original task `input`, `scenario`, `configuration`, and optional `repetition`.
- `target_tool` and `expected_arguments`, validated against the authoritative registry before evaluation.
- `expected_output` for trajectory trials.
- `attempts`: chronological generation opportunities, including unsuccessful attempts. An empty list explicitly means no call was produced.
- `outcome`: `completed`, `provider_error`, `timeout`, or `cancelled`.
- Optional `telemetry`: measured duration, input/output/cache-read tokens, model call count, and top-level tool fingerprints. Missing values stay null.

An attempt may include:

```json
{
  "search_call": {
    "call_id": "s1",
    "name": "search_tools",
    "arguments": "{\"query\":\"save text\"}"
  },
  "search_result": {
    "call_id": "s1",
    "status": "success",
    "raw_output": "{\"tools\":[{\"name\":\"save\",\"input_schema\":{\"type\":\"object\",\"properties\":{\"text\":{\"type\":\"string\"}},\"required\":[\"text\"]}}]}"
  },
  "call": {
    "call_id": "e1",
    "name": "execute_tool",
    "arguments": "{\"tool_name\":\"save\",\"arguments\":{\"text\":\"hello\"}}"
  },
  "execution": {
    "call_id": "e1",
    "tool_name": "save",
    "arguments": {"text": "hello"},
    "status": "success",
    "output": {"saved": true}
  }
}
```

Search results use `{"tools": [{"name": ..., "input_schema": ...}]}`. If the actual search implementation has another response format, normalize its recorded response explicitly before submission. This version accepts the keyword-search call `search_tools({"query": "..."})`; direct-name loading, group activation/deactivation, and multi-target tasks require additional benchmark profiles.

`execution` must describe the **real backend dispatch**, including the real tool name, actual received arguments, execution status, and observed output. A successful outer `execute_tool` response alone is insufficient evidence. The evaluator compares those values with the labels; it cannot inspect external state. For state-changing tools, the capture harness must supply an independently verified result or use deterministic test tools.

Supply `repaired_arguments` only when a repair was actually observed. It contains the complete corrected argument string, including the dispatcher envelope. Raw generation and repaired generation are scored separately. Retries are additional attempts with distinct call IDs. This first version models one target invocation per attempt and at most one associated search; log repeated searches as separate opportunities, or deliberately reuse a schema in the `dispatcher` profile.

## Four checkpoints

1. **Search generation:** call presence, complete generation, JSON syntax, entry point, and the search input schema.
2. **Search response:** successful search, response structure, unique candidates, target Recall@K, and exact schema agreement with the registry. Candidate schema comparisons ignore object key order but preserve array order. Search wording itself is not compared with a fixed phrase.
3. **Tool generation:** strict JSON parsing, dispatcher envelope, real tool selection, real argument schema, and decoded argument values. Duplicate JSON keys, NaN/Infinity, trailing commas, and fenced JSON fail strict parsing. Strings compare after decoding, so valid escape spellings are equivalent; booleans remain distinct from numbers.
4. **Execution:** real dispatch name, actual arguments, execution status, and expected output. A correct generated call does not imply a successful execution.

Each checkpoint is `passed`, `failed`, `not_reached`, or `not_applicable`. If upstream failure prevents a downstream action, an absent downstream observation is `not_reached`. If the action was observed despite an upstream failure, its own evidence is still scored, while the complete pipeline remains unsuccessful. Missing execution after a valid call fails a trajectory trial. Incomplete streamed calls retain `complete: false` and fail generation even when their current JSON happens to parse.

## Metrics and denominators

The report returns every attempt's raw evidence and separate repaired result, plus trial labels and configuration. `summary` covers all configurations; compare `groups` to avoid pooling different providers, scopes, or settings.

| Summary metric | Unit and meaning |
|---|---|
| `first_json_valid` | Per trial: first tool-generation opportunity has valid raw JSON |
| `first_envelope_valid` | Per dispatcher/discovery trial: first JSON matches the wrapper schema; native trials are not applicable |
| `first_tool_name_correct` | Per trial: first decoded target name matches the labeled tool |
| `first_tool_schema_valid` | Per trial: first generated real arguments satisfy the selected tool schema |
| `first_arguments_correct` | Per trial: first generated arguments match the labeled values |
| `first_generation_success` | Per trial: all first-generation checks pass |
| `first_attempt_success` | Per trial: every applicable checkpoint passes on the first attempt without repair |
| `eventual_success` | Per trial: at least one complete applicable pipeline succeeds; capture outcome must be completed |
| `repair_used` | Per trial: a repair was recorded |
| `repair_recovery` | Among trials with a repair: at least one failed raw generation becomes a passing repaired generation |
| `retry_used` | Per trial: more than one attempt was recorded |
| `retry_recovery` | Among retried trials: the first effective attempt fails and a later attempt succeeds |
| `search_recall_at_k` | Per discovery attempt: a successful search returns the target in the first K candidates |

Checkpoint counts use attempts as their unit; first-attempt metrics use trials. With a single labeled target, Recall@K is binary for each search. Empty `attempts` contribute one failed generation opportunity, with `captured_attempts: 0` preserved in the report. Provider errors are separately counted as trial errors and remain in overall success denominators.

Every count includes `passed`, `failed`, `not_reached`, `not_applicable`, and `eligible`. `eligible = passed + failed + not_reached`. `rate = passed / eligible`; `reached_rate = passed / (passed + failed)`. A zero denominator produces null, never a fabricated 0% or 100%. Stage-specific checks that cannot run because of malformed generated arguments count as unsuccessful in per-trial first-generation metrics.

Repair and retry usage rates describe events, not quality scores. A repair recovery does not itself establish end-to-end success. Telemetry summaries include the number of observations contributing to each total or mean. Tool fingerprints diagnose catalog changes; actual cache hits must come from provider usage, not fingerprint equality.

## Capture with AgentScope

```python
from evaluations.tool_loading.agentscope import ToolLoadingRecorder

recorder = ToolLoadingRecorder()
async for event in agent.reply_stream(user_msg):
    recorder.observe(event)

calls = recorder.calls
attempt = recorder.attempt(calls[-1].call_id if calls else None)
```

The recorder accepts native event objects or their `model_dump(mode="json")` dictionaries. It selects one `reply_id`, accumulates raw argument fragments, and matches search results by call ID. Pass `search_call_id` to `attempt()` for discovery trials. Keep the same recorder through permission/external-execution resumption. Capture the event stream before context compaction, which may remove or alter historical evidence.

Use `record_repair(call_id, corrected_arguments)` at the parser/validation boundary only when a real repair occurred. In the current AgentScope implementation, `on_check_permission` receives parsed arguments after validation; comparing these with a strictly parsed raw call can detect repairs that reached this hook. Calls rejected earlier still appear in the event stream. A hook at the parser boundary is needed to capture repairs that subsequently fail schema validation.

Use `record_execution(Execution(...))` at the real tool dispatch boundary to record actual arguments and output. The recorder deliberately does not infer real execution from model output or an outer dispatcher result. `on_acting` covers permitted, validated tool execution only, so it cannot replace the event stream for recording validation failures.

Call `record_model_tools(tools)` from the model-call hook if catalog fingerprints are useful, and `telemetry(duration_ms=...)` to export explicitly measured latency and usage. The current model-call count is the number of observed `ModelCallEndEvent` records; failed requests without an end event are not inferred.

This package provides the recorder and evaluator. The tested AgentScope application supplies the model calls, search implementation, dispatcher, and test-tool environment.
