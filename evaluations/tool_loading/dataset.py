"""Load generation tasks and attach captured attempts for existing scoring."""

import argparse
import json
from pathlib import Path
from typing import Any, Literal, Self

from pydantic import Field, model_validator

from agentscope_eval.schemas import Contract

from .json_utils import parse_json, schema_matches
from .schemas import Attempt, Configuration, Telemetry, ToolSpec, Trial

DEFAULT_DATASET = (
    Path(__file__).parent / "datasets" / "argument_generation_v1.json"
)


class GenerationCase(Contract):
    """A task and its labels, without any model observations."""

    case_id: str = Field(min_length=1)
    scenario: str = Field(min_length=1)
    input: str = Field(min_length=1, max_length=100_000)
    target_tool: str = Field(min_length=1)
    expected_arguments: dict[str, Any]


class GenerationDataset(Contract):
    """Versioned tasks for native versus dispatcher argument generation."""

    dataset_id: str = Field(min_length=1)
    version: str = Field(min_length=1)
    evaluation_scope: Literal["generation"]
    tools: list[ToolSpec] = Field(min_length=1, max_length=1000)
    cases: list[GenerationCase] = Field(min_length=1, max_length=1000)

    @model_validator(mode="after")
    def validate_labels(self) -> Self:
        """Reject duplicate identities and labels outside the tool schemas."""
        json.dumps(self.model_dump(), allow_nan=False)
        registry = {tool.name: tool for tool in self.tools}
        if len(registry) != len(self.tools):
            raise ValueError("Tool names must be unique")
        if len({case.case_id for case in self.cases}) != len(self.cases):
            raise ValueError("case_id must be unique")
        for case in self.cases:
            if case.target_tool not in registry:
                raise ValueError(f"Unknown target tool: {case.target_tool}")
            if not schema_matches(
                case.expected_arguments,
                registry[case.target_tool].input_schema,
            ):
                raise ValueError(
                    f"{case.case_id}: expected_arguments violate tool schema"
                )
        return self

    @property
    def version_tag(self) -> str:
        """Return an identifier for the unchanged catalog and prompts."""
        return f"{self.dataset_id}:{self.version}"

    def make_trial(
        self,
        case_id: str,
        *,
        configuration: Configuration,
        run_id: str,
        attempts: list[Attempt],
        repetition: int = 0,
        outcome: Literal[
            "completed", "provider_error", "timeout", "cancelled"
        ] = "completed",
        telemetry: Telemetry | None = None,
    ) -> Trial:
        """Attach caller-supplied captures without generating or repairing.

        Pass an empty attempts list when no call was produced. Provider
        failures must be recorded through outcome, rather than omitted.
        """
        if configuration.evaluation_scope != "generation":
            raise ValueError("This dataset supports generation scope only")
        if configuration.variant not in ("native", "dispatcher"):
            raise ValueError("This dataset supports native and dispatcher")
        case = next(
            (case for case in self.cases if case.case_id == case_id), None
        )
        if case is None:
            raise ValueError(f"Unknown case_id: {case_id}")
        return Trial(
            **case.model_dump(),
            configuration=configuration,
            run_id=run_id,
            attempts=attempts,
            repetition=repetition,
            outcome=outcome,
            telemetry=telemetry if telemetry is not None else Telemetry(),
        )


def load_dataset(path: str | Path = DEFAULT_DATASET) -> GenerationDataset:
    """Load strictly parsed JSON and validate all task labels."""
    return GenerationDataset.model_validate(
        parse_json(Path(path).read_text(encoding="utf-8"))
    )


def main():
    """Validate a task dataset; this command does not run an experiment."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", nargs="?", type=Path, default=DEFAULT_DATASET)
    args = parser.parse_args()
    try:
        dataset = load_dataset(args.input)
    except (ValueError, OSError) as exc:
        parser.exit(2, f"Invalid dataset: {exc}\n")
    print(
        f"Validated {dataset.version_tag}: {len(dataset.cases)} cases, "
        f"{len(dataset.tools)} tools (generation only; no model calls)."
    )


if __name__ == "__main__":
    main()
