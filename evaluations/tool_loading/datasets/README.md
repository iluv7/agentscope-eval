# Argument generation dataset v1

A small, hand-designed benchmark for comparing native tool calls with the
`execute_tool` envelope. The checked-in JSON is the reproducible source of
truth: 16 fixed tasks, four tool schemas, and expected arguments. It contains
no model responses, execution records, or performance results.

This is an initial generation-only suite. It does not measure search quality,
real execution, cache hits, or general agent capability. The tools are schema
definitions for capture tasks; no backend implementations are needed for
this first experiment.

## Cases

| Tool | Cases | Coverage |
|---|---:|---|
| `record_text` | 8 | Plain text, quotes/backslashes, real versus literal whitespace escapes, Unicode, code, JSON as text, empty text, long text |
| `record_items` | 3 | Escaped array elements, duplicates/order, empty array |
| `record_options` | 2 | Booleans, zero/negative integers, null versus empty string |
| `record_tree` | 3 | Depth 2, depth 8, branching objects and arrays |

Tree depth counts node levels, with a leaf at depth 1. Text includes composed
and combining Unicode characters; preserve their code points. Arrays preserve
order and duplicates. JSON object key order and equivalent JSON escape
spellings do not affect scoring.

Each prompt supplies the target name and payload explicitly. Copying the
payload is intentional: the experiment tests faithful argument generation,
not task reasoning or tool discovery. `expected_arguments` is the scoring
label. Give the model the `input` field and relevant tool schema, not the
entire dataset record. All payloads are synthetic and authored for this suite.

## Validate locally

From the repository root:

```bash
uv run python -m evaluations.tool_loading.dataset
```

This validates the dataset and every expected argument against its tool
schema. It does not call models or produce benchmark scores. The task JSON
cannot be submitted directly to the scoring API, which requires observations.

## Experiment protocol

1. Use each task unchanged in both variants. For `native`, expose the target
   tool with its original input schema. For `dispatcher`, expose only
   `execute_tool` using `DISPATCH_SCHEMA` from `evaluate.py`, and supply the
   target's name and unchanged input schema in one fixed history or system
   message. The inner `arguments` field must be an object, not a JSON string.
2. Keep the model/version, sampling settings, and output-token budget fixed
   within a comparison. Give long-text cases enough output budget. Record
   schema placement and structured-output settings; provider schema
   restrictions must be reported, not silently rewritten. Schema placement
   and output constraints can affect results independently of nesting.
3. Start with one run per case and variant (32 generations per model) to
   check integration. Use repeated runs for measurements; for example,
   five repetitions give 160 generations per model. This is a small baseline,
   not evidence of broad reliability or statistical significance.
4. Capture raw argument strings before parsing/repair. Include missing calls,
   truncated calls, timeouts, and provider errors. A run with no call has
   `attempts=[]`; a provider failure also sets the appropriate `outcome`.
   Record actual retries/repairs separately if the tested application uses
   them. Do not fill observations from `expected_arguments`.
5. Score with the existing evaluator. Compare configuration groups and
   scenarios, especially `first_generation_success`, JSON/schema validity,
   and argument correctness. Keep native and dispatcher results separate.

## Connect captured observations

`load_dataset()` loads the bundled tasks. `make_trial()` joins one task's
labels to explicit observations; it does not run AgentScope or synthesize
successful calls. An experiment runner still needs to select schemas, invoke
the model through AgentScope, and collect each run's events.

The following code belongs **after** a real run has been captured. Supply
`case_id`, `attempts`, `outcome`, `telemetry`, unique `run_id`, `repetition`,
and the actual provider/model/AgentScope metadata from that runner:

```python
from evaluations.tool_loading.dataset import load_dataset
from evaluations.tool_loading.evaluate import evaluate_tool_loading
from evaluations.tool_loading.schemas import BenchmarkRequest, Configuration

dataset = load_dataset()
configuration = Configuration(
    provider=provider,
    model=model,
    variant=variant,  # "native" or "dispatcher"
    evaluation_scope="generation",
    schema_placement=schema_placement,
    agentscope_version=agentscope_version,
    catalog_version=dataset.version_tag,
    prompt_version=dataset.version_tag,
    source="recorded",
)
trial = dataset.make_trial(
    case_id,
    configuration=configuration,
    run_id=run_id,
    repetition=repetition,
    attempts=attempts,  # list[Attempt], from the real capture
    outcome=outcome,
    telemetry=telemetry,
)
request = BenchmarkRequest(tools=dataset.tools, trials=[trial])
report = evaluate_tool_loading(request)
```

For a batch, collect trials and submit them in one `BenchmarkRequest` (up to
1,000 trials per request). The same request JSON works with the existing
CLI or `/v1/benchmarks/tool-loading/evaluate` endpoint. Keep generated reports
outside this dataset directory. Bump the dataset version if prompts, schemas,
or labels change; version any runner-specific prompts separately as well.
