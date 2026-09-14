"""Run all four tool-loading checkpoints through an AgentScope chat model."""

import argparse
import asyncio
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any

from .cli import render_markdown
from .evaluate import DISPATCH_SCHEMA, SEARCH_SCHEMA, evaluate_tool_loading
from .json_utils import parse_json
from .schemas import (
    Attempt,
    BenchmarkRequest,
    Configuration,
    RawCall,
    SearchResult,
    Telemetry,
)
from .trajectory import (
    DEFAULT_TRAJECTORY_DATASET,
    TrajectoryDataset,
    execute_by_name,
    load_trajectory_dataset,
    search_catalog,
    search_response,
)


def _function_tool(name: str, description: str, parameters: dict) -> dict:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": parameters,
        },
    }


SEARCH_TOOL = _function_tool(
    "search_tools",
    "Search the tool catalog for tools relevant to the current task.",
    SEARCH_SCHEMA,
)
EXECUTE_TOOL = _function_tool(
    "execute_tool",
    "Execute a discovered tool by name with an object of arguments.",
    DISPATCH_SCHEMA,
)


def _tool_hash(tool: dict) -> str:
    return hashlib.sha256(
        json.dumps(tool, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _choice(mode: str, name: str):
    from agentscope.tool import ToolChoice

    if mode == "forced":
        return ToolChoice(mode=name)
    return ToolChoice(mode=mode)


async def _model_call(model, messages, tool, choice_mode):
    """Return one raw tool call and usage from a non-streaming model."""
    if getattr(model, "stream", None) is not False:
        raise ValueError("The trajectory runner requires model.stream=False")
    response = await model(
        messages,
        tools=[tool],
        tool_choice=_choice(choice_mode, tool["function"]["name"]),
    )
    calls = [
        block
        for block in response.content
        if getattr(block, "type", None) == "tool_call"
    ]
    call = calls[0] if calls else None
    raw = None
    if call is not None:
        raw = RawCall(
            call_id=call.id,
            name=call.name,
            arguments=call.input,
            complete=response.is_last and len(calls) == 1,
        )
    return call, raw, response.usage


def _add_usage(total: dict[str, int], usage: Any) -> None:
    if usage is None:
        return
    total["usage_responses"] += 1
    total["input_tokens"] += usage.input_tokens
    total["output_tokens"] += usage.output_tokens
    total["cache_read_tokens"] += usage.cache_input_tokens


def _search_messages(case):
    from agentscope.message import SystemMsg, UserMsg

    return [
        SystemMsg(
            "system",
            "Use search_tools to find the tool needed for the task. "
            "Use a concise query describing the required capability. "
            "Do not answer the task directly.",
        ),
        UserMsg("user", case.input),
    ]


def _execute_messages(case, search_block, result):
    from agentscope.message import (
        AssistantMsg,
        SystemMsg,
        TextBlock,
        ToolResultBlock,
        ToolResultState,
        UserMsg,
    )

    return [
        SystemMsg(
            "system",
            "First search for the required tool. After receiving the search "
            "result, call execute_tool exactly once. Set tool_name to a "
            "returned tool and arguments to an object satisfying that tool's "
            "input schema. Preserve all requested values exactly.",
        ),
        UserMsg("user", case.input),
        AssistantMsg(
            "assistant",
            [
                search_block,
                ToolResultBlock(
                    id=search_block.id,
                    name="search_tools",
                    output=[TextBlock(text=result)],
                    state=ToolResultState.SUCCESS,
                ),
            ],
        ),
    ]


async def _run_case(
    dataset: TrajectoryDataset,
    case,
    model,
    configuration: Configuration,
    repetition: int,
    semaphore: asyncio.Semaphore,
):
    async with semaphore:
        started = time.perf_counter()
        usage = {
            "input_tokens": 0,
            "output_tokens": 0,
            "cache_read_tokens": 0,
            "model_calls": 0,
            "usage_responses": 0,
        }
        attempts = []
        attempt = None
        tool_hashes = []
        outcome = "completed"
        try:
            tool_hashes.append(_tool_hash(SEARCH_TOOL))
            usage["model_calls"] += 1
            search_block, search_call, first_usage = await _model_call(
                model,
                _search_messages(case),
                SEARCH_TOOL,
                configuration.tool_choice or "auto",
            )
            _add_usage(usage, first_usage)
            if search_call is not None:
                search_call.call_id = f"search:{search_call.call_id}"
            attempt = Attempt(search_call=search_call)
            if search_call is not None and search_call.name == "search_tools":
                try:
                    search_args = parse_json(search_call.arguments)
                    query = search_args["query"]
                    if not isinstance(query, str) or not query.strip():
                        raise ValueError("query must be a non-empty string")
                except (KeyError, TypeError, ValueError):
                    query = None
                if query is not None:
                    raw_result = search_response(
                        search_catalog(dataset, query, dataset.search_k)
                    )
                    attempt.search_result = SearchResult(
                        call_id=search_call.call_id,
                        status="success",
                        raw_output=raw_result,
                    )
                    tool_hashes.append(_tool_hash(EXECUTE_TOOL))
                    usage["model_calls"] += 1
                    (
                        _execute_block,
                        execute_call,
                        second_usage,
                    ) = await _model_call(
                        model,
                        _execute_messages(case, search_block, raw_result),
                        EXECUTE_TOOL,
                        configuration.tool_choice or "auto",
                    )
                    _add_usage(usage, second_usage)
                    if execute_call is not None:
                        execute_call.call_id = (
                            f"execute:{execute_call.call_id}"
                        )
                    attempt.call = execute_call
                    if execute_call is not None and (
                        execute_call.name == "execute_tool"
                    ):
                        try:
                            envelope = parse_json(execute_call.arguments)
                            name = envelope["tool_name"]
                            arguments = envelope["arguments"]
                            if not isinstance(name, str) or not isinstance(
                                arguments, dict
                            ):
                                raise TypeError("invalid execute envelope")
                        except (KeyError, TypeError, ValueError):
                            pass
                        else:
                            execution = execute_by_name(
                                dataset, name, arguments
                            )
                            execution.call_id = execute_call.call_id
                            attempt.execution = execution
        except TimeoutError:
            outcome = "timeout"
        except asyncio.CancelledError:
            outcome = "cancelled"
        except Exception as exc:  # noqa: BLE001
            outcome = "provider_error"
            print(
                f"ERROR {case.case_id} r{repetition}: {type(exc).__name__}",
                flush=True,
            )
        if attempt is not None:
            attempts.append(attempt)
        telemetry = Telemetry(
            duration_ms=round((time.perf_counter() - started) * 1000, 2),
            input_tokens=(
                usage["input_tokens"] if usage["usage_responses"] else None
            ),
            output_tokens=(
                usage["output_tokens"] if usage["usage_responses"] else None
            ),
            cache_read_tokens=(
                usage["cache_read_tokens"]
                if usage["usage_responses"]
                else None
            ),
            model_calls=usage["model_calls"] or None,
            top_level_tools_hashes=tool_hashes,
        )
        return dataset.make_trial(
            case.case_id,
            configuration=configuration,
            run_id=f"{case.case_id}-r{repetition}",
            attempts=attempts,
            repetition=repetition,
            outcome=outcome,
            telemetry=telemetry,
        )


async def run_trajectory(
    dataset: TrajectoryDataset,
    model,
    configuration: Configuration,
    *,
    repetitions: int = 1,
    max_concurrent: int = 4,
) -> BenchmarkRequest:
    """Run all cases and return captured evidence ready for scoring."""
    if repetitions < 1:
        raise ValueError("repetitions must be positive")
    if max_concurrent < 1:
        raise ValueError("max_concurrent must be positive")
    semaphore = asyncio.Semaphore(max_concurrent)
    trials = await asyncio.gather(
        *(
            _run_case(
                dataset,
                case,
                model,
                configuration,
                repetition,
                semaphore,
            )
            for repetition in range(repetitions)
            for case in dataset.cases
        )
    )
    return dataset.request(trials)


def _deepseek_model(args):
    from agentscope.credential import DeepSeekCredential
    from agentscope.model import DeepSeekChatModel

    key = os.environ.get(args.api_key_env, "")
    if not key:
        raise ValueError(f"Environment variable {args.api_key_env} is empty")
    credential_values = {"api_key": key}
    if args.base_url:
        credential_values["base_url"] = args.base_url
    return DeepSeekChatModel(
        credential=DeepSeekCredential(**credential_values),
        model=args.model,
        stream=False,
        max_retries=1,
        parameters=DeepSeekChatModel.Parameters(
            max_tokens=args.max_tokens,
            thinking_enable=False,
            temperature=args.temperature,
        ),
        client_kwargs={"timeout": args.timeout},
    )


async def _main(args):
    import agentscope

    dataset = load_trajectory_dataset(args.dataset)
    model = _deepseek_model(args)
    configuration = Configuration(
        provider="deepseek",
        model=args.model,
        model_version=args.model_version,
        variant="discovery",
        evaluation_scope="trajectory",
        schema_placement="history",
        agentscope_version=agentscope.__version__,
        strict_output=False,
        tool_choice=args.tool_choice,
        temperature=args.temperature,
        catalog_version=dataset.version_tag,
        prompt_version=dataset.version_tag,
        source="recorded",
    )
    request = await run_trajectory(
        dataset,
        model,
        configuration,
        repetitions=args.repetitions,
        max_concurrent=args.max_concurrent,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        request.model_dump_json(indent=2) + "\n", encoding="utf-8"
    )
    report = evaluate_tool_loading(request)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(
            report.model_dump_json(indent=2) + "\n", encoding="utf-8"
        )
    if args.markdown:
        args.markdown.parent.mkdir(parents=True, exist_ok=True)
        args.markdown.write_text(render_markdown(report), encoding="utf-8")
    print(
        f"Completed {len(request.trials)} trajectories: "
        f"{report.summary.metrics['eventual_success'].passed} passed, "
        f"{report.summary.errors} provider errors."
    )


def main():
    """Run the bundled full trajectory dataset with DeepSeek."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset", type=Path, default=DEFAULT_TRAJECTORY_DATASET
    )
    parser.add_argument("--model", default="deepseek-v4-flash")
    parser.add_argument("--model-version")
    parser.add_argument("--api-key-env", default="DEEPSEEK_API_KEY")
    parser.add_argument("--base-url")
    parser.add_argument("--repetitions", type=int, default=1)
    parser.add_argument("--max-concurrent", type=int, default=4)
    parser.add_argument("--max-tokens", type=int, default=8192)
    parser.add_argument("--temperature", type=float, default=0)
    parser.add_argument("--timeout", type=float, default=120)
    parser.add_argument(
        "--tool-choice", choices=("auto", "required", "forced"), default="auto"
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--markdown", type=Path)
    asyncio.run(_main(parser.parse_args()))


if __name__ == "__main__":
    main()
