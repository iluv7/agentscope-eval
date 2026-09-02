"""HTTP endpoint for captured tool-loading observations."""

from fastapi import APIRouter

from .evaluate import evaluate_tool_loading
from .schemas import BenchmarkReport, BenchmarkRequest

router = APIRouter(prefix="/v1/benchmarks/tool-loading", tags=["tool-loading"])


@router.post("/evaluate", response_model=BenchmarkReport)
def evaluate(payload: BenchmarkRequest):
    """Score observations without executing tools or calling a model."""
    return evaluate_tool_loading(payload)
