"""Four-checkpoint scoring without model calls, repair, or tool execution."""

import json
from collections import defaultdict

from agentscope_eval import __version__

from .json_utils import (
    json_equal,
    parse_json,
    schema_matches,
)
from .schemas import (
    Attempt,
    AttemptResult,
    BenchmarkReport,
    BenchmarkRequest,
    Checkpoint,
    Counts,
    GroupSummary,
    TrialResult,
)

SEARCH_SCHEMA = {
    "type": "object",
    "properties": {"query": {"type": "string", "minLength": 1}},
    "required": ["query"],
    "additionalProperties": False,
}
DISPATCH_SCHEMA = {
    "type": "object",
    "properties": {
        "tool_name": {"type": "string", "minLength": 1},
        "arguments": {"type": "object"},
    },
    "required": ["tool_name", "arguments"],
    "additionalProperties": False,
}


def _checked(checks):
    failures = [name for name, passed in checks.items() if passed is False]
    return Checkpoint(
        status="failed" if failures else "passed",
        failures=failures,
        checks=checks,
    )


def _absent(upstream=True, failure="missing_call"):
    return Checkpoint(
        status="failed" if upstream else "not_reached", failures=[failure]
    )


def _passed(check):
    return check.status in ("passed", "not_applicable")


def _decode(call, expected_name=None):
    checks = {
        "call_present": True,
        "generation_complete": call.complete,
        "json_valid": None,
    }
    if expected_name is not None:
        checks["entrypoint_correct"] = call.name == expected_name
    try:
        value = parse_json(call.arguments)
        checks["json_valid"] = True
    except (ValueError, RecursionError):
        checks["json_valid"] = False
        value = None
    return value, checks


def _search(attempt, trial, registry, k):
    if trial.configuration.variant != "discovery":
        return (Checkpoint(status="not_applicable"),) * 2
    if attempt.search_call is None:
        return _absent(), _absent(False, "missing_search_result")
    value, checks = _decode(attempt.search_call, "search_tools")
    checks["search_schema_valid"] = (
        schema_matches(value, SEARCH_SCHEMA) if checks["json_valid"] else None
    )
    generated = _checked(checks)
    if attempt.search_result is None:
        return generated, _absent(_passed(generated), "missing_search_result")

    result = attempt.search_result
    checks = {"search_succeeded": result.status == "success"}
    try:
        value = parse_json(result.raw_output)
        candidates = value["tools"]
        if not isinstance(candidates, list) or not all(
            isinstance(item, dict)
            and isinstance(item.get("name"), str)
            and isinstance(item.get("input_schema"), dict)
            for item in candidates
        ):
            raise ValueError("Invalid candidate list")
        checks["response_json_valid"] = True
    except (ValueError, KeyError, TypeError, RecursionError):
        checks["response_json_valid"] = False
        response = _checked(checks)
        response.recall_at_k = 0
        return generated, response
    names = [item["name"] for item in candidates]
    checks["unique_candidates"] = len(names) == len(set(names))
    checks["target_recalled"] = trial.target_tool in names[:k]
    checks["schemas_match_registry"] = all(
        item["name"] in registry
        and json_equal(
            item["input_schema"], registry[item["name"]].input_schema
        )
        for item in candidates
    )
    response = _checked(checks)
    response.recall_at_k = float(
        checks["search_succeeded"] and checks["target_recalled"]
    )
    return generated, response


def _generation(call, trial, registry, upstream):
    if call is None:
        return _absent(upstream)
    variant = trial.configuration.variant
    value, checks = _decode(
        call, None if variant == "native" else "execute_tool"
    )
    if not checks["json_valid"]:
        return _checked(checks)
    if variant == "native":
        tool_name, arguments = call.name, value
    else:
        checks["envelope_valid"] = schema_matches(value, DISPATCH_SCHEMA)
        if not checks["envelope_valid"]:
            return _checked(checks)
        tool_name, arguments = value["tool_name"], value["arguments"]
    checks["tool_name_correct"] = tool_name == trial.target_tool
    checks["tool_registered"] = tool_name in registry
    checks["tool_schema_valid"] = (
        schema_matches(arguments, registry[tool_name].input_schema)
        if tool_name in registry
        else None
    )
    checks["arguments_correct"] = json_equal(
        arguments, trial.expected_arguments
    )
    return _checked(checks)


def _execution(attempt, trial, generation):
    if trial.configuration.evaluation_scope == "generation":
        return Checkpoint(status="not_applicable")
    if attempt.execution is None:
        return _absent(_passed(generation), "missing_execution")
    execution = attempt.execution
    return _checked(
        {
            "execution_succeeded": execution.status == "success",
            "dispatch_correct": execution.tool_name == trial.target_tool,
            "executed_arguments_correct": json_equal(
                execution.arguments, trial.expected_arguments
            ),
            "output_correct": json_equal(
                execution.output, trial.expected_output
            ),
        }
    )


def _attempt(attempt, trial, registry, k, index):
    search_generation, search_response = _search(attempt, trial, registry, k)
    upstream = _passed(search_generation) and _passed(search_response)
    generation = _generation(attempt.call, trial, registry, upstream)
    repaired = None
    if attempt.repaired_arguments is not None:
        repaired_call = attempt.call.model_copy(
            update={"arguments": attempt.repaired_arguments}
        )
        repaired = _generation(repaired_call, trial, registry, upstream)
    effective = repaired if repaired is not None else generation
    execution = _execution(attempt, trial, effective)
    raw_pass = upstream and _passed(generation) and _passed(execution)
    return AttemptResult(
        index=index,
        evidence=attempt,
        search_generation=search_generation,
        search_response=search_response,
        generation=generation,
        repaired_generation=repaired,
        execution=execution,
        first_pass=raw_pass and repaired is None,
        effective_pass=upstream and _passed(effective) and _passed(execution),
    )


def _counts(statuses):
    totals = {
        name: statuses.count(name)
        for name in ("passed", "failed", "not_reached", "not_applicable")
    }
    reached = totals["passed"] + totals["failed"]
    eligible = reached + totals["not_reached"]
    return Counts(
        **totals,
        eligible=eligible,
        rate=totals["passed"] / eligible if eligible else None,
        reached_rate=totals["passed"] / reached if reached else None,
    )


def _bool_status(value):
    return "passed" if value else "failed"


def _check_status(checkpoint, check_name):
    if checkpoint.status in ("not_reached", "not_applicable"):
        return checkpoint.status
    return _bool_status(checkpoint.checks.get(check_name) is True)


def _summarize(trials, configuration=None, scenario=None):
    measures = defaultdict(list)
    stages = defaultdict(list)
    for trial in trials:
        first = trial.attempts[0]
        for name in ("json_valid", "tool_schema_valid", "arguments_correct"):
            measures[f"first_{name}"].append(
                _check_status(first.generation, name)
            )
        measures["first_generation_success"].append(first.generation.status)
        measures["first_envelope_valid"].append(
            "not_applicable"
            if trial.configuration.variant == "native"
            else _check_status(first.generation, "envelope_valid")
        )
        measures["first_tool_name_correct"].append(
            _check_status(first.generation, "tool_name_correct")
        )
        measures["first_attempt_success"].append(
            _bool_status(first.first_pass and trial.outcome == "completed")
        )
        measures["eventual_success"].append(
            _bool_status(trial.status == "passed")
        )
        repaired = [
            a for a in trial.attempts if a.repaired_generation is not None
        ]
        measures["repair_used"].append(_bool_status(bool(repaired)))
        measures["repair_recovery"].append(
            _bool_status(
                any(
                    a.generation.status != "passed"
                    and a.repaired_generation.status == "passed"
                    for a in repaired
                )
            )
            if repaired
            else "not_applicable"
        )
        retried = len(trial.attempts) > 1
        measures["retry_used"].append(_bool_status(retried))
        measures["retry_recovery"].append(
            _bool_status(not first.effective_pass and trial.status == "passed")
            if retried
            else "not_applicable"
        )
        for attempt in trial.attempts:
            for stage in (
                "search_generation",
                "search_response",
                "generation",
                "execution",
            ):
                stages[stage].append(getattr(attempt, stage).status)
            measures["search_recall_at_k"].append(
                attempt.search_response.status
                if attempt.search_response.status
                in ("not_reached", "not_applicable")
                else _bool_status(attempt.search_response.recall_at_k == 1)
            )
    latency = [
        t.telemetry.duration_ms
        for t in trials
        if t.telemetry.duration_ms is not None
    ]
    telemetry = {
        "latency_samples": len(latency),
        "mean_duration_ms": sum(latency) / len(latency) if latency else None,
        "observed_search_calls": sum(
            a.evidence.search_call is not None
            for t in trials
            for a in t.attempts
        ),
        "observed_tool_calls": sum(
            a.evidence.call is not None for t in trials for a in t.attempts
        ),
    }
    for field in (
        "input_tokens",
        "output_tokens",
        "cache_read_tokens",
        "model_calls",
    ):
        values = [
            getattr(t.telemetry, field)
            for t in trials
            if getattr(t.telemetry, field) is not None
        ]
        telemetry[f"{field}_samples"] = len(values)
        telemetry[f"total_{field}"] = sum(values) if values else None
    return GroupSummary(
        configuration=configuration,
        scenario=scenario,
        runs=len(trials),
        errors=sum(t.status == "error" for t in trials),
        metrics={name: _counts(values) for name, values in measures.items()},
        checkpoints={name: _counts(values) for name, values in stages.items()},
        telemetry=telemetry,
    )


def evaluate_tool_loading(request: BenchmarkRequest) -> BenchmarkReport:
    """Score raw trials and group results without dropping failed samples.

    Args:
        request: A validated registry, labeled trials and search cutoff.

    Returns:
        A report containing original evidence and explicit rate denominators.
    """
    registry = {tool.name: tool for tool in request.tools}
    trials = []
    for trial in request.trials:
        attempts = [
            _attempt(attempt, trial, registry, request.search_k, index)
            for index, attempt in enumerate(trial.attempts or [Attempt()], 1)
        ]
        status = (
            "error"
            if trial.outcome != "completed"
            else (
                "passed"
                if any(a.effective_pass for a in attempts)
                else "failed"
            )
        )
        trials.append(
            TrialResult(
                run_id=trial.run_id,
                case_id=trial.case_id,
                input=trial.input,
                scenario=trial.scenario,
                configuration=trial.configuration,
                repetition=trial.repetition,
                target_tool=trial.target_tool,
                expected_arguments=trial.expected_arguments,
                expected_output=trial.expected_output,
                captured_attempts=len(trial.attempts),
                outcome=trial.outcome,
                status=status,
                attempts=attempts,
                telemetry=trial.telemetry,
            )
        )
    grouped = defaultdict(list)
    for trial in trials:
        key = json.dumps(trial.configuration.model_dump(), sort_keys=True)
        grouped[key].append(trial)
    groups = []
    for members in grouped.values():
        config = members[0].configuration
        groups.append(_summarize(members, config))
        scenarios = defaultdict(list)
        for member in members:
            scenarios[member.scenario].append(member)
        groups.extend(
            _summarize(items, config, name)
            for name, items in scenarios.items()
        )
    return BenchmarkReport(
        evaluator_version=__version__,
        search_k=request.search_k,
        tools=request.tools,
        trials=trials,
        summary=_summarize(trials),
        groups=groups,
    )
