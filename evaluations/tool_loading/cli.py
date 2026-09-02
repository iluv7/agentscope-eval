"""Score saved tool-loading observations and export a report."""

import argparse
import sys
from pathlib import Path

from .evaluate import evaluate_tool_loading
from .json_utils import parse_json
from .schemas import BenchmarkRequest


def render_markdown(report) -> str:
    """Render comparisons with explicit numerators and denominators."""

    def cell(value):
        return str(value).replace("|", "\\|").replace("\n", " ")

    def rate(counts):
        if counts.rate is None:
            return "N/A"
        return f"{counts.passed}/{counts.eligible} ({counts.rate:.1%})"

    lines = [
        "# AgentScope tool-loading benchmark",
        "",
        f"Evaluator version: {report.evaluator_version}. "
        f"Runs: {report.summary.runs}. Errors: {report.summary.errors}.",
        "",
        "Each row keeps provider, model, variant and settings separate. "
        "Rows marked `fixture` are synthetic checks, not model measurements.",
        "",
        "| Group | Provider / model | Variant | Source | Scenario | Runs | "
        "First JSON | First schema | First arguments | "
        "First attempt | Eventual |",
        "|---|---|---|---|---|---:|---|---|---|---|---|",
    ]
    for index, group in enumerate(report.groups, 1):
        config = group.configuration
        values = [
            str(index),
            f"{config.provider} / {config.model}",
            config.variant,
            config.source,
            group.scenario or "all",
            str(group.runs),
            *[
                rate(group.metrics[name])
                for name in (
                    "first_json_valid",
                    "first_tool_schema_valid",
                    "first_arguments_correct",
                    "first_attempt_success",
                    "eventual_success",
                )
            ],
        ]
        lines.append("| " + " | ".join(cell(value) for value in values) + " |")
    lines.extend(
        [
            "",
            "## Configuration details",
            "",
            "Rates include not-reached stages in their eligible denominator. "
            "The JSON report also includes reached-only rates, stage counts, "
            "repairs, retries, labels, telemetry and the original evidence.",
            "",
        ]
    )
    for index, group in enumerate(report.groups, 1):
        lines.extend(
            [
                f"### Group {index}",
                "",
                "```json",
                group.configuration.model_dump_json(indent=2),
                "```",
                "",
            ]
        )
    return "\n".join(lines)


def main():
    """Evaluate a JSON file without starting a server or calling models."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--markdown", type=Path)
    parser.add_argument("--fail-on-failure", action="store_true")
    args = parser.parse_args()
    try:
        request = BenchmarkRequest.model_validate(
            parse_json(args.input.read_text(encoding="utf-8"))
        )
        report = evaluate_tool_loading(request)
        output = report.model_dump_json(indent=2) + "\n"
        if args.output:
            args.output.write_text(output, encoding="utf-8")
        else:
            print(output, end="")
        if args.markdown:
            args.markdown.write_text(render_markdown(report), encoding="utf-8")
    except (ValueError, OSError) as exc:
        parser.exit(2, f"Invalid benchmark input or output path: {exc}\n")
    if args.fail_on_failure and any(
        t.status != "passed" for t in report.trials
    ):
        sys.exit(1)


if __name__ == "__main__":
    main()
