"""Dataset contracts and integration with the existing generation scorer."""

import json
from copy import deepcopy

import pytest

from evaluations.tool_loading.dataset import GenerationDataset, load_dataset
from evaluations.tool_loading.evaluate import evaluate_tool_loading
from evaluations.tool_loading.schemas import (
    Attempt,
    BenchmarkRequest,
    Configuration,
    RawCall,
    Telemetry,
)


def configuration(variant="native", **overrides):
    """Mark all test observations explicitly as synthetic fixtures."""
    return Configuration(
        **{
            "provider": "fixture",
            "model": "dataset-contract-test",
            "source": "fixture",
            "variant": variant,
            "evaluation_scope": "generation",
            "schema_placement": "tools" if variant == "native" else "history",
            "agentscope_version": "test",
            **overrides,
        }
    )


def test_bundled_dataset_scores_both_conventions_without_execution():
    dataset = load_dataset()
    assert len(dataset.cases) == 16
    assert len(dataset.tools) == 4
    trials = []
    for variant in ("native", "dispatcher"):
        for case in dataset.cases:
            # Oracle observations are test-only; never export model results.
            arguments = (
                case.expected_arguments
                if variant == "native"
                else {
                    "tool_name": case.target_tool,
                    "arguments": case.expected_arguments,
                }
            )
            call = RawCall(
                call_id="call-1",
                name=case.target_tool
                if variant == "native"
                else "execute_tool",
                arguments=json.dumps(arguments, ensure_ascii=False),
            )
            trials.append(
                dataset.make_trial(
                    case.case_id,
                    configuration=configuration(variant),
                    run_id=f"{variant}-{case.case_id}",
                    attempts=[Attempt(call=call)],
                )
            )
    report = evaluate_tool_loading(
        BenchmarkRequest(tools=dataset.tools, trials=trials)
    )
    assert report.summary.runs == 32
    assert all(trial.status == "passed" for trial in report.trials)
    assert all(
        trial.attempts[0].execution.status == "not_applicable"
        for trial in report.trials
    )


@pytest.mark.parametrize(
    "change, message",
    [
        ("duplicate_case", "case_id must be unique"),
        ("duplicate_tool", "Tool names must be unique"),
        ("unknown_tool", "Unknown target tool"),
        ("invalid_arguments", "violate tool schema"),
    ],
)
def test_rejects_invalid_dataset_labels(change, message):
    data = load_dataset().model_dump()
    if change == "duplicate_case":
        data["cases"].append(deepcopy(data["cases"][0]))
    elif change == "duplicate_tool":
        data["tools"].append(deepcopy(data["tools"][0]))
    elif change == "unknown_tool":
        data["cases"][0]["target_tool"] = "unknown"
    else:
        data["cases"][0]["expected_arguments"] = {"text": False}
    with pytest.raises(ValueError, match=message):
        GenerationDataset.model_validate(data)


@pytest.mark.parametrize(
    "raw",
    [
        '{"tool_name":"record_text","arguments":',
        '{"tool_name":"record_text","arguments":{"text":"wrong"}}',
        json.dumps(
            {"tool_name": "record_text", "arguments": '{"text":"hello"}'}
        ),
    ],
)
def test_real_capture_is_preserved_and_failed_not_replaced_with_labels(raw):
    dataset = load_dataset()
    trial = dataset.make_trial(
        "text_plain",
        configuration=configuration("dispatcher"),
        run_id="broken-call",
        attempts=[
            Attempt(
                call=RawCall(
                    call_id="call-1", name="execute_tool", arguments=raw
                )
            )
        ],
    )
    report = evaluate_tool_loading(
        BenchmarkRequest(tools=dataset.tools, trials=[trial])
    )
    assert report.trials[0].status == "failed"
    assert report.trials[0].attempts[0].evidence.call.arguments == raw
    assert report.trials[0].attempts[0].repaired_generation is None


def test_missing_call_and_provider_error_remain_in_denominator():
    dataset = load_dataset()
    trials = [
        dataset.make_trial(
            "text_plain",
            configuration=configuration(),
            run_id=f"run-{index}",
            repetition=index,
            attempts=[],
            outcome=outcome,
            telemetry=Telemetry(duration_ms=12),
        )
        for index, outcome in enumerate(("completed", "provider_error"))
    ]
    report = evaluate_tool_loading(
        BenchmarkRequest(tools=dataset.tools, trials=trials)
    )
    assert [trial.status for trial in report.trials] == ["failed", "error"]
    assert report.summary.metrics["first_generation_success"].eligible == 2
    assert [trial.captured_attempts for trial in report.trials] == [0, 0]
    assert report.trials[1].repetition == 1
    assert report.trials[1].telemetry.duration_ms == 12


@pytest.mark.parametrize(
    "case_id, config, message",
    [
        ("unknown", configuration(), "Unknown case_id"),
        (
            "text_plain",
            configuration(evaluation_scope="trajectory"),
            "generation scope only",
        ),
        ("text_plain", configuration("discovery"), "native and dispatcher"),
    ],
)
def test_rejects_unsupported_experiments(case_id, config, message):
    with pytest.raises(ValueError, match=message):
        load_dataset().make_trial(
            case_id, configuration=config, run_id="test", attempts=[]
        )


def test_loader_rejects_duplicate_json_keys(tmp_path):
    path = tmp_path / "invalid.json"
    path.write_text('{"cases":[],"cases":[]}', encoding="utf-8")
    with pytest.raises(ValueError, match="Duplicate key"):
        load_dataset(path)
