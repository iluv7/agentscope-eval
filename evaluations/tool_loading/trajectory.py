"""Full search, generation, and execution benchmark primitives."""

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any, Literal, Self

from pydantic import Field, model_validator

from agentscope_eval.schemas import Contract

from .json_utils import json_equal, parse_json, schema_matches
from .schemas import (
    Attempt,
    BenchmarkRequest,
    Configuration,
    Execution,
    Telemetry,
    ToolSpec,
    Trial,
)

DEFAULT_TRAJECTORY_DATASET = (
    Path(__file__).parent / "datasets" / "full_trajectory_v1.json"
)

ExecutorName = Literal[
    "store_text",
    "sum_integers",
    "format_contact",
    "flatten_tree",
    "render_template",
    "build_manifest",
    "filter_inventory",
    "merge_settings",
]


class TrajectoryTool(Contract):
    """A searchable tool definition bound to a deterministic executor."""

    name: str = Field(min_length=1, max_length=256)
    description: str = Field(min_length=1)
    keywords: list[str] = Field(min_length=1)
    input_schema: dict[str, Any]
    executor: ExecutorName

    def tool_spec(self) -> ToolSpec:
        """Return the authoritative schema used by the evaluator."""
        return ToolSpec(name=self.name, input_schema=self.input_schema)


class TrajectoryCase(Contract):
    """One end-to-end task with search, call, and execution labels."""

    case_id: str = Field(min_length=1)
    scenario: str = Field(min_length=1)
    input: str = Field(min_length=1, max_length=100_000)
    reference_query: str = Field(min_length=1)
    relevant_tools: list[str] = Field(min_length=1)
    target_tool: str = Field(min_length=1)
    expected_arguments: dict[str, Any]
    expected_output: Any


class TrajectoryDataset(Contract):
    """A versioned catalog and labels for all four checkpoints."""

    dataset_id: str = Field(min_length=1)
    version: str = Field(min_length=1)
    tools: list[TrajectoryTool] = Field(min_length=2, max_length=1000)
    cases: list[TrajectoryCase] = Field(min_length=1, max_length=1000)
    search_k: int = Field(default=5, ge=1, le=100)

    @model_validator(mode="after")
    def validate_labels(self) -> Self:
        """Validate search labels, argument schemas, and expected outputs."""
        json.dumps(self.model_dump(), allow_nan=False)
        registry = {tool.name: tool for tool in self.tools}
        if len(registry) != len(self.tools):
            raise ValueError("Tool names must be unique")
        if len({case.case_id for case in self.cases}) != len(self.cases):
            raise ValueError("case_id must be unique")
        for tool in self.tools:
            tool.tool_spec()
            if len(tool.keywords) != len(set(tool.keywords)):
                raise ValueError(f"{tool.name}: keywords must be unique")
        for case in self.cases:
            if case.target_tool not in registry:
                raise ValueError(f"Unknown target tool: {case.target_tool}")
            if case.target_tool not in case.relevant_tools:
                raise ValueError(
                    f"{case.case_id}: relevant_tools must include target_tool"
                )
            if len(case.relevant_tools) != len(set(case.relevant_tools)):
                raise ValueError(
                    f"{case.case_id}: relevant_tools must be unique"
                )
            unknown = set(case.relevant_tools) - registry.keys()
            if unknown:
                raise ValueError(
                    f"{case.case_id}: unknown relevant tools: "
                    + ", ".join(sorted(unknown))
                )
            schema = registry[case.target_tool].input_schema
            if not schema_matches(case.expected_arguments, schema):
                raise ValueError(
                    f"{case.case_id}: expected_arguments violate tool schema"
                )
            observed = execute_registered_tool(
                registry[case.target_tool], case.expected_arguments
            )
            if not json_equal(observed, case.expected_output):
                raise ValueError(
                    f"{case.case_id}: expected_output does not match executor"
                )
            reference_results = search_catalog(
                self, case.reference_query, self.search_k
            )
            reference_names = [tool.name for tool in reference_results]
            if case.target_tool not in reference_names:
                raise ValueError(
                    f"{case.case_id}: reference_query misses target at "
                    f"K={self.search_k}"
                )
        return self

    @property
    def version_tag(self) -> str:
        """Return a stable catalog and prompt identifier."""
        return f"{self.dataset_id}:{self.version}"

    @property
    def tool_specs(self) -> list[ToolSpec]:
        """Return schemas without search-only metadata."""
        return [tool.tool_spec() for tool in self.tools]

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
        """Join real observations to one case's immutable labels."""
        if configuration.variant != "discovery":
            raise ValueError("Full trajectories require variant=discovery")
        if configuration.evaluation_scope != "trajectory":
            raise ValueError("Full trajectories require trajectory scope")
        case = next(
            (case for case in self.cases if case.case_id == case_id), None
        )
        if case is None:
            raise ValueError(f"Unknown case_id: {case_id}")
        values = case.model_dump(exclude={"reference_query"})
        return Trial(
            **values,
            configuration=configuration,
            run_id=run_id,
            attempts=attempts,
            repetition=repetition,
            outcome=outcome,
            telemetry=telemetry if telemetry is not None else Telemetry(),
        )

    def request(self, trials: list[Trial]) -> BenchmarkRequest:
        """Build an evaluator request with this exact registry and cutoff."""
        return BenchmarkRequest(
            tools=self.tool_specs,
            trials=trials,
            search_k=self.search_k,
        )


def load_trajectory_dataset(
    path: str | Path = DEFAULT_TRAJECTORY_DATASET,
) -> TrajectoryDataset:
    """Strictly load and validate a full-trajectory dataset."""
    return TrajectoryDataset.model_validate(
        parse_json(Path(path).read_text(encoding="utf-8"))
    )


def _tokens(value: str) -> set[str]:
    normalized = value.lower().replace("_", "-")
    return set(re.findall(r"[a-z0-9]+|[\u4e00-\u9fff]+", normalized))


def search_catalog(
    dataset: TrajectoryDataset, query: str, k: int | None = None
) -> list[TrajectoryTool]:
    """Rank tools with a deterministic lexical search implementation."""
    query_tokens = _tokens(query)
    if not query_tokens:
        return []
    ranked = []
    for tool in dataset.tools:
        name_tokens = _tokens(tool.name)
        keyword_tokens = set().union(
            *(_tokens(item) for item in tool.keywords)
        )
        description_tokens = _tokens(tool.description)
        score = (
            4 * len(query_tokens & keyword_tokens)
            + 2 * len(query_tokens & name_tokens)
            + len(query_tokens & description_tokens)
        )
        if score:
            ranked.append((score, tool.name, tool))
    ranked.sort(key=lambda item: (-item[0], item[1]))
    limit = dataset.search_k if k is None else k
    return [item[2] for item in ranked[:limit]]


def search_response(tools: list[TrajectoryTool]) -> str:
    """Serialize candidates in the benchmark's canonical response format."""
    return json.dumps(
        {
            "tools": [
                {"name": tool.name, "input_schema": tool.input_schema}
                for tool in tools
            ]
        },
        ensure_ascii=False,
        separators=(",", ":"),
    )


def execute_registered_tool(
    tool: TrajectoryTool, arguments: dict[str, Any]
) -> Any:
    """Validate and execute one deterministic benchmark tool."""
    if not schema_matches(arguments, tool.input_schema):
        raise ValueError(f"Arguments violate schema for {tool.name}")
    functions = {
        "store_text": _store_text,
        "sum_integers": _sum_integers,
        "format_contact": _format_contact,
        "flatten_tree": _flatten_tree,
        "render_template": _render_template,
        "build_manifest": _build_manifest,
        "filter_inventory": _filter_inventory,
        "merge_settings": _merge_settings,
    }
    return functions[tool.executor](arguments)


def execute_by_name(
    dataset: TrajectoryDataset, name: str, arguments: dict[str, Any]
) -> Execution:
    """Dispatch a generated name and retain actual backend evidence."""
    tool = next((tool for tool in dataset.tools if tool.name == name), None)
    if tool is None:
        return Execution(
            call_id="unassigned",
            tool_name=name,
            arguments=arguments,
            status="error",
            output={"error": "tool_not_found"},
        )
    try:
        output = execute_registered_tool(tool, arguments)
        status = "success"
    except (KeyError, TypeError, ValueError) as exc:
        status = "error"
        output = {"error": type(exc).__name__}
    return Execution(
        call_id="unassigned",
        tool_name=name,
        arguments=arguments,
        status=status,
        output=output,
    )


def _store_text(args):
    text = args["text"]
    return {
        "characters": len(text),
        "lines": text.count("\n") + 1,
        "labels": args["labels"],
        "sha256": hashlib.sha256(text.encode()).hexdigest(),
    }


def _sum_integers(args):
    values = args["values"]
    result = {"count": len(values), "sum": sum(values)}
    if args["include_extrema"]:
        result.update(min=min(values), max=max(values))
    return result


def _format_contact(args):
    contact = args["contact"]
    channels = [contact["email"]]
    if contact["phone"] is not None:
        channels.append(contact["phone"])
    return {
        "display": f"{args['name']} <{contact['email']}>",
        "channels": channels,
        "tag_count": len(args["tags"]),
    }


def _flatten_tree(args):
    labels = []
    max_depth = 0

    def visit(node, depth):
        nonlocal max_depth
        labels.append(node["label"])
        max_depth = max(max_depth, depth)
        for child in node["children"]:
            visit(child, depth + 1)

    visit(args["tree"], 1)
    return {
        "labels": labels,
        "node_count": len(labels),
        "max_depth": max_depth,
    }


def _render_template(args):
    rendered = args["template"]
    for key, value in args["values"].items():
        rendered = rendered.replace("{" + key + "}", value)
    if re.search(r"\{[A-Za-z0-9_]+\}", rendered):
        raise ValueError("unresolved_template_variable")
    return {"rendered": rendered, "characters": len(rendered)}


def _build_manifest(args):
    files = []
    total = 0
    for item in args["files"]:
        encoded = item["content"].encode()
        total += len(encoded)
        files.append(
            {
                "path": item["path"],
                "bytes": len(encoded),
                "sha256": hashlib.sha256(encoded).hexdigest(),
            }
        )
    return {"files": files, "total_bytes": total}


def _filter_inventory(args):
    maximum = args["max_price_cents"]
    selected = [
        item
        for item in args["items"]
        if item["stock"] >= args["min_stock"]
        and (maximum is None or item["price_cents"] <= maximum)
    ]
    return {
        "selected_skus": [item["sku"] for item in selected],
        "total_stock": sum(item["stock"] for item in selected),
    }


def _merge_settings(args):
    return {**args["base"], **args["overrides"]}


def main():
    """Validate the full dataset, backend labels, and reference searches."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "input", nargs="?", type=Path, default=DEFAULT_TRAJECTORY_DATASET
    )
    args = parser.parse_args()
    try:
        dataset = load_trajectory_dataset(args.input)
    except (ValueError, OSError) as exc:
        parser.exit(2, f"Invalid trajectory dataset: {exc}\n")
    reciprocal_ranks = []
    for case in dataset.cases:
        names = [
            tool.name
            for tool in search_catalog(
                dataset, case.reference_query, dataset.search_k
            )
        ]
        reciprocal_ranks.append(1 / (names.index(case.target_tool) + 1))
    mean_rank = sum(reciprocal_ranks) / len(reciprocal_ranks)
    print(
        f"Validated {dataset.version_tag}: {len(dataset.cases)} cases, "
        f"{len(dataset.tools)} tools, reference MRR={mean_rank:.3f}."
    )


if __name__ == "__main__":
    main()
