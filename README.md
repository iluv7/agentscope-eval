# agentscope-eval

An evaluation layer for **AgentScope**, powered by [DeepEval](https://github.com/confident-ai/deepeval) 4.1.8. Submit recorded AgentScope replies to evaluate execution status, tool calls, and answer quality through a local API or Python adapter. Results include scores, explanations, and batch summaries.

## Quick start

Requires Python 3.11+ and [uv](https://docs.astral.sh/uv/getting-started/installation/).

```bash
git clone https://github.com/iluv7/agentscope-eval.git
cd agentscope-eval
uv sync --locked
uv run agentscope-eval
```

Open <http://127.0.0.1:8787/docs> to submit requests through the interactive API documentation. To use a different port, run `uv run agentscope-eval --port 8788`.

Dependencies are installed from PyPI and pinned in `uv.lock`; a separate DeepEval checkout is unnecessary. The evaluation service reads JSON exported by AgentScope. AgentScope runs in the environment of the project under test and does not need to be installed in the evaluation service.

## Evaluate AgentScope records

The dedicated endpoint accepts the AgentScope **2.x** message block format, verified locally with native `Msg` objects from version 2.0.7.post1. The 1.x `tool_use` format is unsupported. The example was serialized from native AgentScope message objects and includes a tool call and a final answer:

```bash
curl -s http://127.0.0.1:8787/v1/agentscope/evaluate \
  -H 'Content-Type: application/json' \
  --data-binary @examples/agentscope.json
```

The example evaluates `execution_success`, `tool_correctness`, and `exact_match`. All three work without a model API key. The expected summary is:

```json
{"total": 1, "passed": 1, "failed": 0, "errors": 0, "pass_rate": 1.0}
```

Each metric entry in `results` contains `score`, `threshold`, `status`, `reason`, and `duration_ms`.

The adapter preserves tool call order, matches results by tool call ID, and extracts the text after the last tool call or result as the final answer. It excludes `thinking`, `hint`, and binary blocks from the answer. Unknown block types, duplicate call IDs, invalid arguments, and inconsistent results return HTTP 422.

### Capture an existing AgentScope event stream

Use the native `Msg.append_event()` method in the event consumer of your AgentScope project to accumulate a complete reply record:

```python
from agentscope.event import ReplyStartEvent
from agentscope.message import Msg

record = None
async for event in agent.reply_stream(user_msg):
    if isinstance(event, ReplyStartEvent) and record is None:
        record = Msg(
            id=event.reply_id, name=event.name, role="assistant", content=[]
        )
    elif record is not None and getattr(event, "reply_id", None) == record.id:
        record.append_event(event)

payload = {
    "cases": [
        {
            "case_id": "case-001",
            "input": user_msg.get_text_content(),
            "reply": record.model_dump(mode="json"),
        }
    ],
    "metrics": [{"name": "execution_success", "threshold": 1}],
}
```

POST `payload` to `/v1/agentscope/evaluate`. For replies that require human confirmation or external tool execution, complete the existing resumption flow and continue appending events to the same record until `ReplyEndEvent` arrives. Incomplete records without a finish reason return HTTP 422.

Submit the complete accumulated record. The final message returned by `agent.reply()` alone may omit tool calls and results needed to evaluate tool behavior. Evaluation currently operates on one reply at a time. For multiple agents, collect records separately by `reply_id`; compressed conversation context cannot replace the original records.

The `examples/tools.json` and `examples/answers.json` files demonstrate the normalized `/v1/evaluate` endpoint for AgentScope test code that already prepares evaluation fields.

## Configure an LLM judge

```bash
cp .env.example .env
```

Set `EVAL_JUDGE_MODEL` and `EVAL_JUDGE_API_KEY` in `.env`, optionally set `EVAL_JUDGE_BASE_URL`, and restart the service. The judge uses Chat Completions with `response_format: json_object`, so choose a model that supports this API and JSON mode. If a local model server does not validate API keys, use a placeholder such as `local`.

```bash
curl -s http://127.0.0.1:8787/v1/evaluate \
  -H 'Content-Type: application/json' \
  --data-binary @examples/answers.json
```

The judge receives the content to evaluate and the scoring instructions. The service does not execute the agent under test or feed `expected_output` to it. Judge requests go to the external or local model endpoint you configure. API keys are configured only on the server.

## Metrics

| Metric | Additional required fields | Uses an LLM | Scoring behavior |
|---|---|---|---|
| `execution_success` | `finished_reason`, `tool_result_states` (extracted automatically by the AgentScope endpoint) | No | Scores 1 when the reply is `completed` and every tool result is `success`; otherwise 0 |
| `exact_match` | `expected_output` | No | Scores 1 for exact string equality, including whitespace; otherwise 0 |
| `tool_correctness` | `tools_called`, `expected_tools` | No | DeepEval compares tool count, order, names, and input parameters exactly; tool outputs are not compared |
| `answer_correctness` | `expected_output` | Yes | G-Eval assesses semantic correctness against a reference answer |
| `answer_relevancy` | None | Yes | DeepEval assesses whether the answer addresses the input |
| `faithfulness` | Non-empty `retrieval_context` | Yes | DeepEval assesses whether the answer is supported by the retrieved context |

Every normalized case requires `case_id`, `input`, and `actual_output`. The AgentScope endpoint extracts `actual_output` from the supplied `reply`. All selected metrics apply to every case in a batch. The default `threshold` is 0.7; a score at or above the threshold passes. For tool correctness, explicitly provide empty arrays `[]` when no tools were called or expected. Omitting these fields means the data was not collected and causes validation to fail.

Tool correctness measures call matching, so use it alongside `execution_success` to check execution outcomes. Successful execution does not establish that the business task was completed: after a ticket creation tool succeeds, for example, the resulting ticket should still be verified. This version does not expose `TaskCompletionMetric`, which requires a complete trace.

## Endpoints and errors

- `GET /health`: Service status and whether a judge is configured. Does not probe model connectivity.
- `GET /v1/metrics`: Available metrics, input requirements, and whether each needs a judge.
- `POST /v1/agentscope/evaluate`: Score accumulated AgentScope messages, with up to 100 cases and 6 distinct metrics per batch.
- `POST /v1/evaluate`: Score normalized evaluation inputs.
- `GET /docs`: Interactive OpenAPI documentation.

Missing fields, duplicate IDs, and unknown metrics return HTTP 422. Requesting LLM metrics without a configured judge returns HTTP 503. Timeouts and model errors during scoring are recorded on the affected metric as `status=error` and `score=null`; other scores continue. HTTP 200 means a report was generated. Check `summary` and the individual results to determine whether the evaluation passed.

A case has status `error` if any metric errors, `failed` if no metric errors but at least one falls below its threshold, and `passed` if all metrics pass. The pass rate is `passed / total`, with errored cases included in the denominator. Metric results remain separate; the service does not average scores across different metrics.

## Python API

```python
import asyncio
import json
from pathlib import Path

from agentscope_eval.agentscope import AgentScopeRequest
from agentscope_eval.config import Settings
from agentscope_eval.engine import Evaluator
from agentscope_eval.judge import JsonJudge


async def main():
    payload = json.loads(Path("examples/agentscope.json").read_text())
    request = AgentScopeRequest.model_validate(payload).to_evaluate_request()
    evaluator = Evaluator(Settings(), JsonJudge("", None))
    result = await evaluator.evaluate(request)
    print(result.model_dump_json(indent=2))


asyncio.run(main())
```

For LLM metrics, create an `AsyncOpenAI` client, pass it to `JsonJudge`, and close it when finished. See `src/agentscope_eval/api.py` for an example. Reuse one `Evaluator` within an event loop to share its concurrency limit.

In a test process with AgentScope installed, import `from_agentscope` from `agentscope_eval.agentscope` and pass `case_id`, `input`, and an accumulated native `Msg` as `reply`. The function returns a normalized `EvalCase`.

## Implementation and scope

```text
src/agentscope_eval/
├── api.py        HTTP routes and client lifecycle
├── agentscope.py AgentScope 2.x message adapter
├── schemas.py    Input validation and response contracts
├── engine.py     Metric instances, concurrency, and error isolation
├── judge.py      OpenAI-compatible JSON judge adapter
├── config.py     Server configuration
└── cli.py        Local service entry point
```

The engine calls `a_measure()` on fresh metric instances, avoiding the global run state and temporary files used by `deepeval.evaluate()`. DeepEval telemetry is disabled by default, and its file system is configured as read-only. The service does not configure the Confident AI platform. Active metrics share a per-process concurrency limit, which defaults to 4. A metric may issue multiple model requests internally, so this limit is not an exact request rate limit. Each metric has a default execution timeout of 120 seconds, excluding queue time.

This is a stateless local API that waits for scoring to finish before returning a report. It does not include agent execution, a database, a task queue, report history, or a multi-tenant platform. Callers are responsible for saving the returned JSON. Judge cost and token usage reporting are not yet available. LLM scores may vary; keep the judge model and scoring configuration fixed when comparing versions, and manually review a sample of results. The service binds to `127.0.0.1` by default and does not implement authentication or traffic controls for public deployment.

## Development

```bash
uv run pytest
uv run ruff check .
uv run ruff format --check .
```

Tests exercise the real DeepEval tool matching and LLM metric implementations. LLM HTTP responses are mocked, so the tests do not incur model charges. Coverage also includes input validation, error isolation, timeouts, concurrency, AgentScope message conversion, and the JSON judge adapter.

GitHub Actions runs tests and Ruff checks on Python 3.11 and 3.13 without model API keys.

## License

Copyright 2026 iluv7. Licensed under the [Apache License 2.0](LICENSE).

DeepEval is installed as a separate third-party dependency and is covered by its own [Apache-2.0 license](https://github.com/confident-ai/deepeval/blob/main/LICENSE.md).
