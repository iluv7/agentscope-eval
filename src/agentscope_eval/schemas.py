"""Normalized contracts used by the AgentScope evaluation layer."""

from typing import Any, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

MetricName = Literal[
    "exact_match",
    "tool_correctness",
    "answer_correctness",
    "answer_relevancy",
    "faithfulness",
    "execution_success",
]
FinishReason = Literal["completed", "interrupted", "exceed_max_iters", "error"]
ToolState = Literal["success", "error", "interrupted", "denied", "running"]
Status = Literal["passed", "failed", "error"]
LLM_METRICS = {"answer_correctness", "answer_relevancy", "faithfulness"}


class Contract(BaseModel):
    """Reject misspelled fields and non-finite numbers at the API boundary."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class ToolCall(Contract):
    """A completed or expected tool invocation, supplied by the caller."""

    name: str = Field(min_length=1, max_length=256)
    input_parameters: dict[str, Any] = Field(default_factory=dict)
    output: Any = None


class EvalCase(Contract):
    """One recorded agent response with optional reference evidence."""

    case_id: str = Field(min_length=1, max_length=256)
    input: str = Field(min_length=1, max_length=100_000)
    actual_output: str = Field(max_length=100_000)
    expected_output: str | None = Field(default=None, max_length=100_000)
    tools_called: list[ToolCall] | None = Field(default=None, max_length=500)
    expected_tools: list[ToolCall] | None = Field(default=None, max_length=500)
    retrieval_context: list[str] | None = Field(default=None, max_length=100)
    finished_reason: FinishReason | None = None
    tool_result_states: list[ToolState] | None = None


class MetricSpec(Contract):
    """Select a metric and its minimum passing score."""

    name: MetricName
    threshold: float = Field(default=0.7, ge=0, le=1)


class EvaluateRequest(Contract):
    """A bounded batch; every selected metric applies to every case."""

    cases: list[EvalCase] = Field(min_length=1, max_length=100)
    metrics: list[MetricSpec] = Field(min_length=1, max_length=6)

    @model_validator(mode="after")
    def validate_requirements(self) -> Self:
        """Reject ambiguous identities and missing metric inputs."""
        ids = [case.case_id for case in self.cases]
        names = [metric.name for metric in self.metrics]
        if len(ids) != len(set(ids)):
            raise ValueError("case_id must be unique within a batch")
        if len(names) != len(set(names)):
            raise ValueError("metric names must be unique within a batch")
        for case in self.cases:
            if "execution_success" in names:
                if (
                    case.finished_reason is None
                    or case.tool_result_states is None
                ):
                    raise ValueError(
                        f"{case.case_id}: finished_reason and "
                        "tool_result_states are required"
                    )
            if set(names) & {"exact_match", "answer_correctness"}:
                if case.expected_output is None:
                    raise ValueError(
                        f"{case.case_id}: expected_output is required"
                    )
            if "tool_correctness" in names:
                if case.tools_called is None or case.expected_tools is None:
                    raise ValueError(
                        f"{case.case_id}: tools_called and expected_tools "
                        "are required; use [] for no tools"
                    )
            if "faithfulness" in names:
                if not case.retrieval_context or not all(
                    context.strip() for context in case.retrieval_context
                ):
                    raise ValueError(
                        f"{case.case_id}: non-empty retrieval_context required"
                    )
        return self


class MetricResult(Contract):
    """A score or an execution error; errors never receive a numeric score."""

    name: MetricName
    threshold: float
    status: Status
    score: float | None = None
    reason: str | None = None
    error: str | None = None
    duration_ms: float


class CaseResult(Contract):
    """Independent metric results for one input case."""

    case_id: str
    status: Status
    metrics: list[MetricResult]


class Summary(Contract):
    """Counts over all submitted cases, including evaluation errors."""

    total: int
    passed: int
    failed: int
    errors: int
    pass_rate: float


class EvaluateResponse(Contract):
    """An ephemeral report returned directly to the caller."""

    evaluation_id: str
    evaluator_version: str
    deepeval_version: str
    judge_model: str | None
    results: list[CaseResult]
    summary: Summary
