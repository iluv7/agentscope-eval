# agentscope-eval

An evaluation service for [AgentScope](https://github.com/agentscope-ai/agentscope), built on [DeepEval](https://github.com/confident-ai/deepeval).

Evaluate agent responses, tool calls, and execution results through a local API.

## Tool-loading reliability benchmark

The [tool-loading benchmark](docs/tool-loading-benchmark.md) compares native tool calls, an `execute_tool` dispatcher, and the complete search-to-execution path. It preserves malformed model arguments as failed observations and reports first-pass, repair, retry, and end-to-end results with explicit denominators. Scoring is deterministic and does not require a judge API key.

```bash
uv run agentscope-eval-benchmark examples/tool_loading/fixture.json \
  --output /tmp/tool-loading-report.json \
  --markdown /tmp/tool-loading-report.md
```

The included observations are synthetic fixtures, not measurements of real models. Use the AgentScope event recorder and your application's execution instrumentation to supply real observations. The raw benchmark endpoint is `POST /v1/benchmarks/tool-loading/evaluate`.

## Quick start

Requires Python 3.11+ and [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/iluv7/agentscope-eval.git
cd agentscope-eval
uv sync --locked
uv run agentscope-eval
```

Open [API documentation](http://127.0.0.1:8787/docs) to get started.

## Acknowledgments

Thanks to [DeepEval](https://github.com/confident-ai/deepeval) for its evaluation tools and [AgentScope](https://github.com/agentscope-ai/agentscope) for its agent framework.
