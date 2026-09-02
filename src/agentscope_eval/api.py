"""Local HTTP interface for evaluation of existing outputs."""

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from openai import AsyncOpenAI

from agentscope_eval import __version__
from agentscope_eval.agentscope import AgentScopeRequest
from agentscope_eval.benchmarks.schemas import (
    BenchmarkReport,
    BenchmarkRequest,
)
from agentscope_eval.benchmarks.tool_loading import evaluate_tool_loading
from agentscope_eval.config import Settings
from agentscope_eval.engine import Evaluator
from agentscope_eval.judge import JsonJudge
from agentscope_eval.schemas import (
    LLM_METRICS,
    EvaluateRequest,
    EvaluateResponse,
)

METRICS = [
    {
        "name": "execution_success",
        "requires_judge": False,
        "required_fields": ["finished_reason", "tool_result_states"],
        "description": "Completed AgentScope reply with all tools successful.",
    },
    {
        "name": "exact_match",
        "requires_judge": False,
        "required_fields": ["expected_output"],
        "description": "Exact string equality, including whitespace.",
    },
    {
        "name": "tool_correctness",
        "requires_judge": False,
        "required_fields": ["tools_called", "expected_tools"],
        "description": "Exact tool sequence, names and input parameters.",
    },
    {
        "name": "answer_correctness",
        "requires_judge": True,
        "required_fields": ["expected_output"],
        "description": "G-Eval semantic correctness against a reference.",
    },
    {
        "name": "answer_relevancy",
        "requires_judge": True,
        "required_fields": [],
        "description": "Relevance of the answer to the user's input.",
    },
    {
        "name": "faithfulness",
        "requires_judge": True,
        "required_fields": ["retrieval_context"],
        "description": "Whether claims are supported by retrieved context.",
    },
]


def create_app(settings: Settings | None = None) -> FastAPI:
    """Build an app with one shared client and evaluator per process."""
    settings = settings if settings is not None else Settings()

    @asynccontextmanager
    async def lifespan(app):
        client = None
        if settings.judge_configured:
            client = AsyncOpenAI(
                api_key=settings.judge_api_key.get_secret_value(),
                base_url=settings.judge_base_url or None,
                timeout=settings.metric_timeout_seconds,
                max_retries=1,
            )
        app.state.evaluator = Evaluator(
            settings, JsonJudge(settings.judge_model, client)
        )
        try:
            yield
        finally:
            if client is not None:
                await client.close()

    app = FastAPI(
        title="AgentScope Eval",
        version=__version__,
        lifespan=lifespan,
        description="Evaluate recorded AgentScope replies and tool outcomes.",
    )

    @app.get("/health")
    async def health():
        return {"status": "ok", "judge_configured": settings.judge_configured}

    @app.post(
        "/v1/benchmarks/tool-loading/evaluate", response_model=BenchmarkReport
    )
    def benchmark_tool_loading(payload: BenchmarkRequest):
        return evaluate_tool_loading(payload)

    @app.get("/v1/metrics")
    async def list_metrics():
        return {"metrics": METRICS}

    @app.post("/v1/evaluate", response_model=EvaluateResponse)
    async def evaluate(payload: EvaluateRequest, request: Request):
        if any(m.name in LLM_METRICS for m in payload.metrics):
            if not settings.judge_configured:
                raise HTTPException(
                    status_code=503,
                    detail="Configure EVAL_JUDGE_MODEL and EVAL_JUDGE_API_KEY "
                    "for LLM metrics; rule metrics work without a judge.",
                )
        return await request.app.state.evaluator.evaluate(payload)

    @app.post("/v1/agentscope/evaluate", response_model=EvaluateResponse)
    async def evaluate_agentscope(
        payload: AgentScopeRequest, request: Request
    ):
        try:
            normalized = payload.to_evaluate_request()
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return await evaluate(normalized, request)

    return app
