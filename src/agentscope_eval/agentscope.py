"""Convert AgentScope 2.x recorded reply messages to evaluation cases."""

import json
from typing import Annotated, Any, Literal

from pydantic import BaseModel, Field

from agentscope_eval.schemas import (
    Contract,
    EvalCase,
    EvaluateRequest,
    FinishReason,
    MetricSpec,
    ToolCall,
    ToolState,
)


class TextRecord(BaseModel):
    """The text portion of an AgentScope TextBlock."""

    type: Literal["text"]
    text: str


class CallRecord(BaseModel):
    """AgentScope stores the tool arguments as a JSON string."""

    type: Literal["tool_call"]
    id: str
    name: str
    input: str


class ResultRecord(BaseModel):
    """A tool result, associated by ID rather than arrival order."""

    type: Literal["tool_result"]
    id: str
    name: str
    output: Any
    state: ToolState


class OtherRecord(BaseModel):
    """Blocks that do not belong in the answer text."""

    type: Literal["thinking", "hint", "data"]


BlockRecord = Annotated[
    TextRecord | CallRecord | ResultRecord | OtherRecord,
    Field(discriminator="type"),
]


class ReplyRecord(BaseModel):
    """A full reply accumulated with AgentScope Msg.append_event()."""

    id: str
    name: str
    role: Literal["assistant"]
    content: list[BlockRecord] = Field(max_length=2000)
    finished_reason: FinishReason


class AgentScopeCase(Contract):
    """A recorded AgentScope reply plus evaluation-only reference data."""

    case_id: str
    input: str
    reply: ReplyRecord
    expected_output: str | None = None
    expected_tools: list[ToolCall] | None = None
    retrieval_context: list[str] | None = None

    def to_eval_case(self) -> EvalCase:
        """Extract final text, tool arguments and execution outcome."""
        calls: dict[str, CallRecord] = {}
        results: dict[str, ResultRecord] = {}
        final_text: list[str] = []
        for block in self.reply.content:
            if isinstance(block, TextRecord):
                final_text.append(block.text)
            elif isinstance(block, CallRecord):
                if block.id in calls:
                    raise ValueError(f"Duplicate tool call ID: {block.id}")
                calls[block.id] = block
                final_text.clear()
            elif isinstance(block, ResultRecord):
                if block.id in results:
                    raise ValueError(f"Duplicate tool result ID: {block.id}")
                results[block.id] = block
                final_text.clear()
        if results.keys() - calls.keys():
            raise ValueError("Tool results require matching tool call IDs")

        tools = []
        states = []
        for call_id, call in calls.items():
            try:
                arguments = json.loads(call.input)
            except ValueError as exc:
                raise ValueError(
                    f"Tool {call.name}: input must be a complete JSON object"
                ) from exc
            if not isinstance(arguments, dict):
                raise ValueError(f"Tool {call.name}: input must be an object")
            result = results.get(call_id)
            if result is not None and result.name != call.name:
                raise ValueError(f"Tool name mismatch for ID {call_id}")
            tools.append(
                ToolCall(
                    name=call.name,
                    input_parameters=arguments,
                    output=result.output if result else None,
                )
            )
            states.append(result.state if result else "running")

        return EvalCase(
            case_id=self.case_id,
            input=self.input,
            actual_output="\n".join(final_text),
            expected_output=self.expected_output,
            tools_called=tools,
            expected_tools=self.expected_tools,
            retrieval_context=self.retrieval_context,
            finished_reason=self.reply.finished_reason,
            tool_result_states=states,
        )


class AgentScopeRequest(Contract):
    """Batch contract for the AgentScope-specific endpoint."""

    cases: list[AgentScopeCase] = Field(min_length=1, max_length=100)
    metrics: list[MetricSpec] = Field(min_length=1, max_length=6)

    def to_evaluate_request(self) -> EvaluateRequest:
        """Normalize records and apply all metric requirements."""
        return EvaluateRequest(
            cases=[case.to_eval_case() for case in self.cases],
            metrics=self.metrics,
        )


def from_agentscope(
    *, case_id: str, input: str, reply: Any, **references: Any
) -> EvalCase:
    """Convert a native Msg or its JSON dump without executing an agent.

    Args:
        case_id: A unique identifier for this evaluation case.
        input: The user request associated with the recorded reply.
        reply: A full recorded AgentScope Msg or its model_dump() dictionary.
        **references: expected_output, expected_tools or retrieval_context.

    Returns:
        A normalized evaluation case ready for Evaluator.evaluate().

    Raises:
        ValueError: If the record is incomplete or structurally inconsistent.
    """
    if hasattr(reply, "model_dump"):
        reply = reply.model_dump(mode="json")
    return AgentScopeCase(
        case_id=case_id, input=input, reply=reply, **references
    ).to_eval_case()
