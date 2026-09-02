# Evaluation suites

Keep each evaluation in one folder. The shared `src/agentscope_eval/` package provides the service and reusable evaluation infrastructure; suite-specific behavior belongs here.

```text
evaluations/
├── __init__.py              Discover suite API routers
├── README.md               Directory conventions
└── tool_loading/
    ├── __init__.py
    ├── __main__.py          Standalone entry point
    ├── cli.py               Input files and report export
    ├── api.py               Suite-owned FastAPI router
    ├── evaluate.py          Scoring logic
    ├── schemas.py           Input and report contracts
    ├── agentscope.py        Raw event capture
    ├── json_utils.py        Suite-specific JSON validation
    ├── examples/
    │   ├── fixture.json
    │   └── generate_fixture.py
    ├── tests/
    │   └── test_tool_loading.py
    └── README.md
```

## Add an evaluation

1. Create `evaluations/<name>/` with an `__init__.py`. Use a Python identifier such as `memory_recall` for the folder name.
2. Put its evaluator, schemas, capture helpers, examples, and documentation in that folder. Use relative imports between the suite's modules. Only extract helpers into the shared package when multiple evaluations need them.
3. Add `__main__.py` to run its CLI. From the repository root, invoke `uv run python -m evaluations.<name> ...`. No central command registration is required. The suite owns its CLI arguments and input format.
4. Optionally add `api.py` exposing a FastAPI `APIRouter` named `router`. Give routes a unique prefix, such as `/v1/benchmarks/memory-recall`. The shared service registers all installed suite routers when it starts; restart after adding a suite. Keep imports free of model calls and execution side effects. CLI-only suites can omit `api.py`.
5. Add tests under the suite's `tests/` folder. `uv run pytest` and CI collect both shared tests and all suite tests. Use `uv run pytest evaluations/<name>/tests` to run one suite. Test modules are imported by path, allowing different suites to reuse test filenames.
6. Declare any new dependencies in the root `pyproject.toml` and update `uv.lock`. The root wheel configuration already includes the entire `evaluations` package, including its examples and documentation.

For example, a suite API module can contain:

```python
from fastapi import APIRouter

from .evaluate import evaluate
from .schemas import EvaluationRequest, EvaluationReport

router = APIRouter(prefix="/v1/benchmarks/memory-recall")


@router.post("/evaluate", response_model=EvaluationReport)
def score(payload: EvaluationRequest):
    return evaluate(payload)
```

Use the [tool-loading suite](tool_loading/README.md) as a working example. Keep generated run reports and credentials outside the committed fixture data. Label synthetic fixtures explicitly; they are not measurements of real models.
