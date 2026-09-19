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

## Full AgentScope evaluation flow

Each benchmark case provides a task to the agent and keeps the reference
tool, arguments, and output private for deterministic scoring. The complete
trajectory is captured as one `Trial` and then evaluated checkpoint by
checkpoint:

```mermaid
flowchart TD
    A["Load Case<br/>input + reference labels"]
    B["Send only input to<br/>AgentScope Agent"]
    C["First reasoning phase<br/>model calls search_tools"]
    D["AgentScope executes search_tools<br/>and returns tools + schemas"]
    E["Second reasoning phase<br/>model calls execute_tool"]
    F["Dispatcher validates arguments<br/>and invokes the real tool"]
    G["Real tool returns status + output"]
    H["Recorder correlates evidence<br/>by call_id into one Attempt"]
    I["Case labels + Attempt<br/>become one Trial"]
    J["Evaluator scores four checkpoints"]
    K["Aggregate Trials<br/>into JSON + Markdown reports"]

    A --> B --> C --> D --> E --> F --> G --> H --> I --> J --> K

    C -. "ToolCall events<br/>raw search arguments" .-> H
    D -. "ToolResult events<br/>actual search response" .-> H
    E -. "ToolCall events<br/>raw execute arguments" .-> H
    F -. "Tool middleware<br/>actual tool + arguments" .-> H
    G -. "Tool middleware<br/>status + output" .-> H
```

The evaluator checks search-call generation, search-result quality, nested
`execute_tool` arguments, and real execution output. Scoring uses the recorded
evidence, JSON Schema validation, and exact comparisons; it does not use an
LLM judge.

## Acknowledgments

Thanks to [DeepEval](https://github.com/confident-ai/deepeval) for its evaluation tools and [AgentScope](https://github.com/agentscope-ai/agentscope) for its agent framework.
