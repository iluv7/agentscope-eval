# agentscope-eval

An evaluation service for [AgentScope](https://github.com/agentscope-ai/agentscope), built on [DeepEval](https://github.com/confident-ai/deepeval).

Evaluate agent responses, tool calls, and execution results through a local API.

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
