"""Stateless scoring using fresh DeepEval metrics per case."""

import asyncio
import logging
import math
import time
from uuid import uuid4

import deepeval
from deepeval.metrics import (
    AnswerRelevancyMetric,
    FaithfulnessMetric,
    GEval,
    ToolCorrectnessMetric,
)
from deepeval.test_case import LLMTestCase, SingleTurnParams, ToolCallParams
from deepeval.test_case import ToolCall as DeepEvalToolCall

from agentscope_eval import __version__
from agentscope_eval.config import Settings
from agentscope_eval.judge import JsonJudge
from agentscope_eval.schemas import (
    LLM_METRICS,
    CaseResult,
    EvalCase,
    EvaluateRequest,
    EvaluateResponse,
    MetricResult,
    MetricSpec,
    Summary,
)

logger = logging.getLogger(__name__)


class Evaluator:
    """Bound active metrics across requests within one server process."""

    def __init__(self, settings: Settings, judge: JsonJudge):
        self.settings = settings
        self.judge = judge
        self._slots = asyncio.Semaphore(settings.max_concurrent)

    def _build_metric(self, spec: MetricSpec):
        common = {"model": self.judge, "threshold": spec.threshold}
        if spec.name == "tool_correctness":
            return ToolCorrectnessMetric(
                **common,
                should_exact_match=True,
                evaluation_params=[ToolCallParams.INPUT_PARAMETERS],
            )
        if spec.name == "answer_correctness":
            return GEval(
                **common,
                name="Answer correctness",
                evaluation_params=[
                    SingleTurnParams.INPUT,
                    SingleTurnParams.ACTUAL_OUTPUT,
                    SingleTurnParams.EXPECTED_OUTPUT,
                ],
                evaluation_steps=[
                    "Identify the user's requirements and reference facts.",
                    "Compare the actual answer with the reference; allow "
                    "equivalent wording and penalize errors and omissions.",
                    "Score correctness and explain the specific evidence.",
                ],
            )
        if spec.name == "answer_relevancy":
            return AnswerRelevancyMetric(**common)
        if spec.name == "faithfulness":
            return FaithfulnessMetric(**common)
        raise ValueError(f"Unsupported DeepEval metric: {spec.name}")

    @staticmethod
    def _to_test_case(case: EvalCase) -> LLMTestCase:
        def convert(calls):
            if calls is None:
                return None
            return [DeepEvalToolCall(**call.model_dump()) for call in calls]

        return LLMTestCase(
            input=case.input,
            actual_output=case.actual_output,
            expected_output=case.expected_output,
            tools_called=convert(case.tools_called),
            expected_tools=convert(case.expected_tools),
            retrieval_context=case.retrieval_context,
        )

    async def _score(self, case: EvalCase, spec: MetricSpec) -> MetricResult:
        async with self._slots:
            started = time.perf_counter()
            try:
                async with asyncio.timeout(
                    self.settings.metric_timeout_seconds
                ):
                    if spec.name == "execution_success":
                        score = float(
                            case.finished_reason == "completed"
                            and all(
                                state == "success"
                                for state in case.tool_result_states
                            )
                        )
                        reason = (
                            f"Reply ended as {case.finished_reason}; "
                            f"tool result states: {case.tool_result_states}."
                        )
                    elif spec.name == "exact_match":
                        score = float(
                            case.actual_output == case.expected_output
                        )
                        reason = (
                            "Output exactly matches the reference."
                            if score
                            else "Output differs from the reference."
                        )
                    else:
                        metric = self._build_metric(spec)
                        await metric.a_measure(
                            self._to_test_case(case), _show_indicator=False
                        )
                        score = float(metric.score)
                        reason = metric.reason
                    if not math.isfinite(score) or not 0 <= score <= 1:
                        raise ValueError("Metric returned an invalid score")
                return MetricResult(
                    name=spec.name,
                    threshold=spec.threshold,
                    status="passed" if score >= spec.threshold else "failed",
                    score=score,
                    reason=reason,
                    duration_ms=round(
                        (time.perf_counter() - started) * 1000, 2
                    ),
                )
            except Exception as exc:
                # Do not expose provider errors, credentials, or prompts.
                code = (
                    "timeout"
                    if isinstance(exc, TimeoutError)
                    else ("evaluation_error")
                )
                logger.warning(
                    "Metric %s failed: %s", spec.name, type(exc).__name__
                )
                return MetricResult(
                    name=spec.name,
                    threshold=spec.threshold,
                    status="error",
                    error=code,
                    duration_ms=round(
                        (time.perf_counter() - started) * 1000, 2
                    ),
                )

    async def evaluate(self, request: EvaluateRequest) -> EvaluateResponse:
        """Score a validated batch, preserving case and metric order."""
        requires_judge = any(m.name in LLM_METRICS for m in request.metrics)
        if requires_judge and self.judge.client is None:
            raise ValueError("Judge is not configured")

        async def score_case(case):
            results = await asyncio.gather(
                *(self._score(case, spec) for spec in request.metrics)
            )
            statuses = {result.status for result in results}
            status = (
                "error"
                if "error" in statuses
                else ("failed" if "failed" in statuses else "passed")
            )
            return CaseResult(
                case_id=case.case_id, status=status, metrics=results
            )

        results = await asyncio.gather(
            *(score_case(case) for case in request.cases)
        )
        passed = sum(result.status == "passed" for result in results)
        failed = sum(result.status == "failed" for result in results)
        errors = sum(result.status == "error" for result in results)
        return EvaluateResponse(
            evaluation_id=str(uuid4()),
            evaluator_version=__version__,
            deepeval_version=deepeval.__version__,
            judge_model=self.judge.name if requires_judge else None,
            results=results,
            summary=Summary(
                total=len(results),
                passed=passed,
                failed=failed,
                errors=errors,
                pass_rate=passed / len(results),
            ),
        )
