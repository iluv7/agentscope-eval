"""Capture unmodified AgentScope events before argument repair/validation."""

import hashlib
import json
from copy import deepcopy
from typing import Any

from agentscope_eval.benchmarks.schemas import (
    Attempt,
    Execution,
    RawCall,
    SearchResult,
    Telemetry,
)


class ToolLoadingRecorder:
    """Observe one reply without executing tools or changing agent events.

    Args:
        reply_id: Reply to capture, or None to select the first reply start.
    """

    def __init__(self, reply_id: str | None = None):
        self.reply_id = reply_id
        self._calls: dict[str, RawCall] = {}
        self._outputs: dict[str, str] = {}
        self._states: dict[str, str] = {}
        self._repairs: dict[str, str] = {}
        self._executions: dict[str, Execution] = {}
        self._seen_events: set[str] = set()
        self._usage: list[dict] = []
        self._tools_hashes: list[str] = []

    @property
    def calls(self) -> list[RawCall]:
        """Return snapshots of calls in their generation order."""
        return [call.model_copy(deep=True) for call in self._calls.values()]

    def observe(self, event: Any) -> None:
        """Append one native AgentScope event or serialized event dictionary.

        Args:
            event: An event from reply_stream, including failed tool calls.

        Raises:
            ValueError: If deltas/end events precede a matching call start.
        """
        data = (
            event.model_dump(mode="json")
            if hasattr(event, "model_dump")
            else event
        )
        if self.reply_id is None and data.get("type") == "REPLY_START":
            self.reply_id = data["reply_id"]
        if self.reply_id is None or data.get("reply_id") != self.reply_id:
            return
        event_id = data.get("id")
        if event_id and event_id in self._seen_events:
            return
        kind = data["type"]
        call_id = data.get("tool_call_id")
        if kind == "TOOL_CALL_START":
            if call_id in self._calls:
                raise ValueError(f"Duplicate tool call start: {call_id}")
            self._calls[call_id] = RawCall(
                call_id=call_id,
                name=data["tool_call_name"],
                arguments="",
                complete=False,
            )
        elif kind in ("TOOL_CALL_DELTA", "TOOL_CALL_END"):
            if call_id not in self._calls:
                raise ValueError(f"Missing tool call start: {call_id}")
            if kind == "TOOL_CALL_DELTA":
                self._calls[call_id].arguments += data["delta"]
            else:
                self._calls[call_id].complete = True
        elif kind == "TOOL_RESULT_TEXT_DELTA":
            self._outputs[call_id] = (
                self._outputs.get(call_id, "") + data["delta"]
            )
        elif kind == "TOOL_RESULT_END":
            self._states[call_id] = data["state"]
        elif kind == "MODEL_CALL_END":
            self._usage.append(deepcopy(data))
        if event_id:
            self._seen_events.add(event_id)

    def record_repair(self, call_id: str, arguments: str | dict) -> None:
        """Record post-repair arguments separately from raw output.

        Args:
            call_id: Original call identifier.
            arguments: The complete argument object/string after framework
                repair, including the dispatcher envelope if applicable.
        """
        if call_id not in self._calls:
            raise ValueError(f"Unknown call ID: {call_id}")
        self._repairs[call_id] = (
            arguments
            if isinstance(arguments, str)
            else json.dumps(arguments, ensure_ascii=False, allow_nan=False)
        )

    def record_execution(self, execution: Execution) -> None:
        """Record observed real-tool arguments and output at the backend.

        Args:
            execution: Evidence captured at the actual dispatch/tool boundary.
                Do not infer this from model-generated arguments.
        """
        if execution.call_id not in self._calls:
            raise ValueError(f"Unknown call ID: {execution.call_id}")
        self._executions[execution.call_id] = execution.model_copy(deep=True)

    def record_model_tools(self, tools: list[dict]) -> None:
        """Fingerprint top-level tools; this is not a cache-hit measurement."""
        serialized = json.dumps(tools, sort_keys=True, allow_nan=False)
        self._tools_hashes.append(
            hashlib.sha256(serialized.encode()).hexdigest()
        )

    def attempt(
        self, call_id: str | None, search_call_id: str | None = None
    ) -> Attempt:
        """Build an attempt with explicit search/execute correlation.

        Args:
            call_id: Generated native/dispatcher call, or None if absent.
            search_call_id: The preceding search call for a discovery attempt.

        Returns:
            An independent snapshot retaining malformed arguments verbatim.
        """
        search_result = None
        if search_call_id is not None and (
            search_call_id in self._outputs or search_call_id in self._states
        ):
            search_result = SearchResult(
                call_id=search_call_id,
                status=self._states.get(search_call_id, "running"),
                raw_output=self._outputs.get(search_call_id, ""),
            )
        return Attempt(
            call=self._calls[call_id] if call_id is not None else None,
            search_call=self._calls[search_call_id]
            if search_call_id is not None
            else None,
            search_result=search_result,
            repaired_arguments=self._repairs.get(call_id),
            execution=self._executions.get(call_id),
        ).model_copy(deep=True)

    def telemetry(self, duration_ms: float | None = None) -> Telemetry:
        """Export measured latency and usage from observed end events."""
        values = {}
        for target, source in (
            ("input_tokens", "input_tokens"),
            ("output_tokens", "output_tokens"),
            ("cache_read_tokens", "cache_input_tokens"),
        ):
            values[target] = (
                sum(item[source] for item in self._usage)
                if self._usage and all(source in item for item in self._usage)
                else None
            )
        return Telemetry(
            **values,
            duration_ms=duration_ms,
            model_calls=len(self._usage) if self._usage else None,
            top_level_tools_hashes=list(self._tools_hashes),
        )
