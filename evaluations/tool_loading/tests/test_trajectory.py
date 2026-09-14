"""Tests for the complete search-to-execution benchmark profile."""

import asyncio
import json

import pytest

from evaluations.tool_loading.evaluate import evaluate_tool_loading
from evaluations.tool_loading.schemas import (
    Attempt,
    Configuration,
    RawCall,
    SearchResult,
)
from evaluations.tool_loading.trajectory import (
    execute_by_name,
    load_trajectory_dataset,
    search_catalog,
    search_response,
)


def configuration():
    return Configuration(
        provider="fixture",
        model="deterministic",
        agentscope_version="fixture",
        variant="discovery",
        evaluation_scope="trajectory",
        schema_placement="history",
        tool_choice="auto",
        catalog_version="test",
        prompt_version="test",
        source="fixture",
    )


def test_dataset_reference_searches_and_executors():
    dataset = load_trajectory_dataset()

    assert len(dataset.tools) == 16
    assert len(dataset.cases) == 12
    for case in dataset.cases:
        candidates = search_catalog(dataset, case.reference_query)
        assert candidates[0].name == case.target_tool
        execution = execute_by_name(
            dataset, case.target_tool, case.expected_arguments
        )
        assert execution.status == "success"
        assert execution.output == case.expected_output


def test_complete_trajectory_scores_all_four_checkpoints():
    dataset = load_trajectory_dataset()
    config = configuration()
    trials = []
    for case in dataset.cases:
        search_id = f"search:{case.case_id}"
        execute_id = f"execute:{case.case_id}"
        result = search_response(search_catalog(dataset, case.reference_query))
        envelope = {
            "tool_name": case.target_tool,
            "arguments": case.expected_arguments,
        }
        execution = execute_by_name(
            dataset, case.target_tool, case.expected_arguments
        )
        execution.call_id = execute_id
        trial = dataset.make_trial(
            case.case_id,
            configuration=config,
            run_id=case.case_id,
            attempts=[
                Attempt(
                    search_call=RawCall(
                        call_id=search_id,
                        name="search_tools",
                        arguments=json.dumps(
                            {"query": case.reference_query},
                            ensure_ascii=False,
                        ),
                    ),
                    search_result=SearchResult(
                        call_id=search_id,
                        status="success",
                        raw_output=result,
                    ),
                    call=RawCall(
                        call_id=execute_id,
                        name="execute_tool",
                        arguments=json.dumps(envelope, ensure_ascii=False),
                    ),
                    execution=execution,
                )
            ],
        )
        trials.append(trial)

    report = evaluate_tool_loading(dataset.request(trials))

    assert report.summary.metrics["eventual_success"].passed == 12
    for name in (
        "search_generation",
        "search_response",
        "generation",
        "execution",
    ):
        assert report.summary.checkpoints[name].passed == 12
    assert report.summary.search_quality["mean_recall_at_k"] == 1
    assert report.summary.search_quality["mean_reciprocal_rank"] == 1
    assert 0 < report.summary.search_quality["mean_precision_at_k"] <= 1


def test_search_miss_blocks_the_complete_pipeline():
    dataset = load_trajectory_dataset()
    case = dataset.cases[0]
    wrong_tool = next(
        tool for tool in dataset.tools if tool.name != case.target_tool
    )
    search_id = "search:miss"
    trial = dataset.make_trial(
        case.case_id,
        configuration=configuration(),
        run_id="miss",
        attempts=[
            Attempt(
                search_call=RawCall(
                    call_id=search_id,
                    name="search_tools",
                    arguments='{"query":"irrelevant capability"}',
                ),
                search_result=SearchResult(
                    call_id=search_id,
                    status="success",
                    raw_output=search_response([wrong_tool]),
                ),
            )
        ],
    )

    result = evaluate_tool_loading(dataset.request([trial])).trials[0]

    assert result.status == "failed"
    assert result.attempts[0].search_response.status == "failed"
    assert result.attempts[0].search_response.recall_at_k == 0
    assert result.attempts[0].generation.status == "not_reached"
    assert result.attempts[0].execution.status == "not_reached"


def test_dispatch_rejects_arguments_outside_the_tool_schema():
    dataset = load_trajectory_dataset()

    execution = execute_by_name(
        dataset,
        "sum_integer_batch",
        {"values": [1, 2], "include_extrema": "yes"},
    )

    assert execution.status == "error"
    assert execution.output == {"error": "ValueError"}


def test_agentscope_runner_captures_two_real_model_calls_per_case():
    pytest.importorskip("agentscope")
    from agentscope.message import ToolCallBlock
    from agentscope.model import ChatResponse, ChatUsage

    from evaluations.tool_loading.trajectory_runner import run_trajectory

    dataset = load_trajectory_dataset()

    class FakeModel:
        stream = False

        def __init__(self):
            self.index = 0

        async def __call__(self, messages, tools, tool_choice):
            case = dataset.cases[self.index // 2]
            if self.index % 2 == 0:
                name = "search_tools"
                arguments = {"query": case.reference_query}
            else:
                name = "execute_tool"
                arguments = {
                    "tool_name": case.target_tool,
                    "arguments": case.expected_arguments,
                }
            self.index += 1
            return ChatResponse(
                content=[
                    ToolCallBlock(
                        id=f"call-{self.index}",
                        name=name,
                        input=json.dumps(arguments, ensure_ascii=False),
                    )
                ],
                is_last=True,
                usage=ChatUsage(
                    input_tokens=10,
                    output_tokens=5,
                    time=0.01,
                ),
            )

    request = asyncio.run(
        run_trajectory(
            dataset,
            FakeModel(),
            configuration(),
            max_concurrent=1,
        )
    )
    report = evaluate_tool_loading(request)

    assert report.summary.metrics["eventual_success"].passed == 12
    assert report.summary.telemetry["total_model_calls"] == 24


def test_agentscope_runner_preserves_search_when_second_call_fails():
    pytest.importorskip("agentscope")
    from agentscope.message import ToolCallBlock
    from agentscope.model import ChatResponse, ChatUsage

    from evaluations.tool_loading.trajectory_runner import run_trajectory

    dataset = load_trajectory_dataset()
    dataset = dataset.model_copy(update={"cases": dataset.cases[:1]})
    case = dataset.cases[0]

    class FailingModel:
        stream = False

        def __init__(self):
            self.calls = 0

        async def __call__(self, messages, tools, tool_choice):
            self.calls += 1
            if self.calls == 2:
                raise RuntimeError("provider unavailable")
            return ChatResponse(
                content=[
                    ToolCallBlock(
                        id="search-call",
                        name="search_tools",
                        input=json.dumps({"query": case.reference_query}),
                    )
                ],
                is_last=True,
                usage=ChatUsage(
                    input_tokens=10,
                    output_tokens=5,
                    time=0.01,
                ),
            )

    request = asyncio.run(
        run_trajectory(
            dataset,
            FailingModel(),
            configuration(),
            max_concurrent=1,
        )
    )

    trial = request.trials[0]
    assert trial.outcome == "provider_error"
    assert trial.attempts[0].search_result is not None
    assert trial.attempts[0].call is None
    assert trial.telemetry.model_calls == 2
