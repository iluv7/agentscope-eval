"""Evidence and reports for native, dispatcher, and discovery trials."""

import json
from typing import Any, Literal, Self

from pydantic import Field, field_validator, model_validator

from agentscope_eval.schemas import Contract

from .json_utils import check_schema, schema_matches

Variant = Literal["native", "dispatcher", "discovery"]
CheckStatus = Literal["passed", "failed", "not_reached", "not_applicable"]


class ToolSpec(Contract):
    """An authoritative tool definition, never a model-generated schema."""

    name: str = Field(min_length=1, max_length=256)
    input_schema: dict[str, Any]

    @field_validator("input_schema")
    @classmethod
    def validate_schema(cls, value):
        """Reject invalid schemas and external schema resolution."""
        try:
            return check_schema(value)
        except Exception as exc:
            raise ValueError(
                "Invalid or unsupported tool JSON schema"
            ) from exc


class Configuration(Contract):
    """Controls that must remain separate when comparing trial results."""

    provider: str = Field(min_length=1)
    model: str = Field(min_length=1)
    variant: Variant
    evaluation_scope: Literal["generation", "trajectory"] = "trajectory"
    schema_placement: Literal["tools", "history", "system"]
    agentscope_version: str = Field(min_length=1)
    model_version: str | None = None
    strict_output: bool | None = None
    temperature: float | None = None
    seed: int | None = None
    catalog_version: str = "unspecified"
    prompt_version: str = "unspecified"
    source: Literal["recorded", "fixture"] = "recorded"


class RawCall(Contract):
    """The unmodified argument string produced by a model."""

    call_id: str = Field(min_length=1)
    name: str
    arguments: str = Field(max_length=200_000)
    complete: bool = True


class SearchResult(Contract):
    """Raw search response: a JSON object containing a tools array."""

    call_id: str
    status: Literal[
        "success", "error", "timeout", "denied", "interrupted", "running"
    ]
    raw_output: str = Field(max_length=500_000)


class Execution(Contract):
    """Observed real-tool execution, recorded at the backend boundary."""

    call_id: str
    tool_name: str
    arguments: dict[str, Any]
    status: Literal["success", "error", "timeout", "denied", "interrupted"]
    output: Any


class Attempt(Contract):
    """One generation opportunity; missing calls are meaningful evidence."""

    search_call: RawCall | None = None
    search_result: SearchResult | None = None
    call: RawCall | None = None
    repaired_arguments: str | None = Field(default=None, max_length=200_000)
    execution: Execution | None = None

    @model_validator(mode="after")
    def validate_evidence(self) -> Self:
        """Reject inconsistent capture records, not malformed model JSON."""
        if self.repaired_arguments is not None and self.call is None:
            raise ValueError("A repair requires its original call")
        if self.search_result is not None:
            if self.search_call is None or (
                self.search_result.call_id != self.search_call.call_id
            ):
                raise ValueError("Search result must match its call ID")
        if self.execution is not None:
            if (
                self.call is None
                or self.execution.call_id != self.call.call_id
            ):
                raise ValueError("Execution must match its outer call ID")
        return self


class Telemetry(Contract):
    """Optional observations; unavailable measurements remain null."""

    duration_ms: float | None = Field(default=None, ge=0)
    input_tokens: int | None = Field(default=None, ge=0)
    output_tokens: int | None = Field(default=None, ge=0)
    cache_read_tokens: int | None = Field(default=None, ge=0)
    model_calls: int | None = Field(default=None, ge=0)
    top_level_tools_hashes: list[str] = Field(default_factory=list)


class Trial(Contract):
    """One task under one configuration and repetition."""

    run_id: str = Field(min_length=1)
    case_id: str = Field(min_length=1)
    input: str = Field(min_length=1, max_length=100_000)
    scenario: str = Field(min_length=1)
    configuration: Configuration
    repetition: int = Field(default=0, ge=0)
    target_tool: str
    expected_arguments: dict[str, Any]
    expected_output: Any = None
    attempts: list[Attempt] = Field(default_factory=list, max_length=50)
    outcome: Literal["completed", "provider_error", "timeout", "cancelled"] = (
        "completed"
    )
    telemetry: Telemetry = Field(default_factory=Telemetry)

    @model_validator(mode="after")
    def validate_variant(self) -> Self:
        """Ensure native/dispatcher trials do not silently hide searches."""
        if self.configuration.evaluation_scope == "trajectory":
            if "expected_output" not in self.model_fields_set:
                raise ValueError("Trajectory trials require expected_output")
        elif any(attempt.execution is not None for attempt in self.attempts):
            raise ValueError("Generation-only trials cannot include execution")
        if self.configuration.variant != "discovery" and any(
            attempt.search_call is not None
            or attempt.search_result is not None
            for attempt in self.attempts
        ):
            raise ValueError("Search observations require variant=discovery")
        ids = [
            call.call_id
            for attempt in self.attempts
            for call in [attempt.search_call, attempt.call]
            if call is not None
        ]
        if len(ids) != len(set(ids)):
            raise ValueError("Call IDs must be unique within a trial")
        return self


class BenchmarkRequest(Contract):
    """Score recorded trials against a versioned, authoritative registry."""

    tools: list[ToolSpec] = Field(min_length=1, max_length=1000)
    trials: list[Trial] = Field(min_length=1, max_length=1000)
    search_k: int = Field(default=5, ge=1, le=100)

    @model_validator(mode="after")
    def validate_ground_truth(self) -> Self:
        """Validate labels before any generated evidence is scored."""
        try:
            json.dumps(self.model_dump(), allow_nan=False)
        except (ValueError, TypeError) as exc:
            raise ValueError(
                "Evidence must contain finite JSON values"
            ) from exc
        registry = {tool.name: tool for tool in self.tools}
        if len(registry) != len(self.tools):
            raise ValueError("Tool names must be unique")
        if len({trial.run_id for trial in self.trials}) != len(self.trials):
            raise ValueError("run_id must be unique")
        for trial in self.trials:
            if trial.target_tool not in registry:
                raise ValueError(f"Unknown target tool: {trial.target_tool}")
            try:
                valid = schema_matches(
                    trial.expected_arguments,
                    registry[trial.target_tool].input_schema,
                )
            except Exception as exc:
                raise ValueError("Unresolvable ground-truth schema") from exc
            if not valid:
                raise ValueError(
                    f"{trial.run_id}: expected_arguments violate tool schema"
                )
        return self


class Checkpoint(Contract):
    """A stage outcome, with optional individual checks and failure codes."""

    status: CheckStatus
    failures: list[str] = Field(default_factory=list)
    checks: dict[str, bool | None] = Field(default_factory=dict)
    recall_at_k: float | None = None


class AttemptResult(Contract):
    """Raw and repaired generations remain separate in the returned report."""

    index: int
    evidence: Attempt
    search_generation: Checkpoint
    search_response: Checkpoint
    generation: Checkpoint
    repaired_generation: Checkpoint | None = None
    execution: Checkpoint
    first_pass: bool
    effective_pass: bool


class TrialResult(Contract):
    """A trial is successful only if one complete attempt succeeds."""

    run_id: str
    case_id: str
    input: str
    scenario: str
    configuration: Configuration
    repetition: int
    target_tool: str
    expected_arguments: dict[str, Any]
    expected_output: Any
    captured_attempts: int
    outcome: str
    status: Literal["passed", "failed", "error"]
    attempts: list[AttemptResult]
    telemetry: Telemetry


class Counts(Contract):
    """Rates expose both unconditional and reached-stage denominators."""

    passed: int
    failed: int
    not_reached: int
    not_applicable: int
    eligible: int
    rate: float | None
    reached_rate: float | None


class GroupSummary(Contract):
    """Results for an exact configuration, optionally one scenario."""

    configuration: Configuration | None = None
    scenario: str | None = None
    runs: int
    errors: int
    metrics: dict[str, Counts]
    checkpoints: dict[str, Counts]
    telemetry: dict[str, float | int | None]


class BenchmarkReport(Contract):
    """A reproducible report, including labels and unmodified observations."""

    benchmark: Literal["tool_loading"] = "tool_loading"
    evaluator_version: str
    search_k: int
    tools: list[ToolSpec]
    trials: list[TrialResult]
    summary: GroupSummary
    groups: list[GroupSummary]
