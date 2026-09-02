"""Exercise real metrics without calling a paid model endpoint."""

import asyncio
import json
import unittest
from pathlib import Path
from unittest.mock import patch

import httpx
from fastapi.testclient import TestClient
from openai import AsyncOpenAI

from agentscope_eval.api import create_app
from agentscope_eval.config import Settings
from agentscope_eval.engine import Evaluator
from agentscope_eval.judge import JsonJudge
from agentscope_eval.schemas import EvaluateRequest

ROOT = Path(__file__).resolve().parents[1]


def settings(**kwargs):
    return Settings(_env_file=None, judge_model="", judge_api_key="", **kwargs)


def payload(metric="exact_match"):
    return {
        "cases": [
            {
                "case_id": "one",
                "input": "Say hello",
                "actual_output": "hello",
                "expected_output": "hello",
                "tools_called": [],
                "expected_tools": [],
                "retrieval_context": ["hello"],
            }
        ],
        "metrics": [{"name": metric}],
    }


class ApiTests(unittest.TestCase):
    def setUp(self):
        self.client = self.enterContext(TestClient(create_app(settings())))

    def test_real_tool_metric_example(self):
        request = json.loads((ROOT / "examples/tools.json").read_text())
        response = self.client.post("/v1/evaluate", json=request)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.json()["summary"],
            {
                "total": 2,
                "passed": 1,
                "failed": 1,
                "errors": 0,
                "pass_rate": 0.5,
            },
        )
        self.assertIsNone(response.json()["judge_model"])

    def test_empty_tools_are_valid_but_missing_tools_are_not(self):
        request = payload("tool_correctness")
        result = self.client.post("/v1/evaluate", json=request).json()
        self.assertEqual(result["results"][0]["metrics"][0]["score"], 1)
        del request["cases"][0]["tools_called"]
        self.assertEqual(
            self.client.post("/v1/evaluate", json=request).status_code, 422
        )

    def test_tool_order_and_extra_calls_fail(self):
        for actual in [["b", "a"], ["a", "b", "c"]]:
            with self.subTest(actual=actual):
                request = payload("tool_correctness")
                request["cases"][0]["tools_called"] = [
                    {"name": name} for name in actual
                ]
                request["cases"][0]["expected_tools"] = [
                    {"name": name} for name in ["a", "b"]
                ]
                result = self.client.post("/v1/evaluate", json=request).json()
                self.assertEqual(result["summary"]["failed"], 1)

    def test_exact_match_preserves_whitespace(self):
        request = payload()
        request["cases"][0]["actual_output"] = "hello "
        result = self.client.post("/v1/evaluate", json=request).json()
        self.assertEqual(result["summary"]["failed"], 1)

    def test_request_validation(self):
        changes = [
            lambda r: r["cases"].append(r["cases"][0].copy()),
            lambda r: r["metrics"].append(r["metrics"][0].copy()),
            lambda r: r["metrics"][0].update(name="task_completion"),
            lambda r: r["metrics"][0].update(threshold=2),
            lambda r: r["cases"][0].pop("expected_output"),
            lambda r: r["cases"][0].update(typo="oops"),
        ]
        for change in changes:
            with self.subTest(change=change):
                request = payload()
                change(request)
                self.assertEqual(
                    self.client.post("/v1/evaluate", json=request).status_code,
                    422,
                )

    def test_faithfulness_requires_evidence(self):
        request = payload("faithfulness")
        request["cases"][0]["retrieval_context"] = [" "]
        self.assertEqual(
            self.client.post("/v1/evaluate", json=request).status_code, 422
        )

    def test_llm_metrics_require_configuration(self):
        response = self.client.post(
            "/v1/evaluate", json=payload("answer_correctness")
        )
        self.assertEqual(response.status_code, 503)
        self.assertFalse(self.client.get("/health").json()["judge_configured"])
        self.assertEqual(
            len(self.client.get("/v1/metrics").json()["metrics"]), 6
        )


class EngineTests(unittest.IsolatedAsyncioTestCase):
    async def test_real_llm_metrics_and_json_transport(self):
        async def transport(request):
            body = json.loads(request.content)
            self.assertEqual(body["model"], "test-judge")
            self.assertEqual(body["response_format"], {"type": "json_object"})
            system = body["messages"][0]["content"]
            schema = json.loads(system.split("Match this JSON schema: ")[1])
            fields = schema["properties"]
            if "score" in fields:
                data = {"score": 9, "reason": "Matches the reference."}
            elif "statements" in fields:
                data = {"statements": ["hello"]}
            elif "truths" in fields:
                data = {"truths": ["hello"]}
            elif "claims" in fields:
                data = {"claims": ["hello"]}
            elif "verdicts" in fields:
                data = {
                    "verdicts": [{"verdict": "yes", "reason": "Supported"}]
                }
            else:
                data = {"reason": "Supported by the evidence."}
            return httpx.Response(
                200,
                json={
                    "id": "test",
                    "object": "chat.completion",
                    "created": 0,
                    "model": "test-judge",
                    "choices": [
                        {
                            "index": 0,
                            "finish_reason": "stop",
                            "message": {
                                "role": "assistant",
                                "content": json.dumps(data),
                            },
                        }
                    ],
                },
            )

        async with httpx.AsyncClient(
            transport=httpx.MockTransport(transport)
        ) as http:
            async with AsyncOpenAI(
                api_key="test-only", http_client=http
            ) as client:
                evaluator = Evaluator(
                    settings(), JsonJudge("test-judge", client)
                )
                request = payload()
                request["metrics"] = [
                    {"name": name}
                    for name in [
                        "answer_correctness",
                        "answer_relevancy",
                        "faithfulness",
                    ]
                ]
                result = await evaluator.evaluate(
                    EvaluateRequest.model_validate(request)
                )
                self.assertEqual(result.summary.passed, 1, result.model_dump())
                self.assertEqual(result.results[0].metrics[0].score, 0.9)
                self.assertEqual(result.judge_model, "test-judge")

    async def test_metric_failure_does_not_hide_other_scores(self):
        evaluator = Evaluator(settings(), JsonJudge("", None))
        request = payload()
        request["metrics"].append({"name": "tool_correctness"})
        with patch.object(
            evaluator,
            "_build_metric",
            side_effect=RuntimeError("secret-value"),
        ):
            result = await evaluator.evaluate(
                EvaluateRequest.model_validate(request)
            )
        self.assertEqual(result.results[0].metrics[0].score, 1)
        self.assertIsNone(result.results[0].metrics[1].score)
        self.assertEqual(result.summary.errors, 1)
        self.assertEqual(result.summary.pass_rate, 0)
        self.assertNotIn("secret-value", result.model_dump_json())

    async def test_timeout_is_an_error(self):
        class SlowMetric:
            async def a_measure(self, *args, **kwargs):
                await asyncio.sleep(1)

        evaluator = Evaluator(
            settings(metric_timeout_seconds=0.01), JsonJudge("", None)
        )
        with patch.object(
            evaluator, "_build_metric", return_value=SlowMetric()
        ):
            result = await evaluator.evaluate(
                EvaluateRequest.model_validate(payload("tool_correctness"))
            )
        metric = result.results[0].metrics[0]
        self.assertEqual(metric.error, "timeout")
        self.assertIsNone(metric.score)

    async def test_concurrency_limit_is_shared_across_requests(self):
        active = 0
        peak = 0

        class TrackingMetric:
            score = 1.0
            reason = "ok"

            async def a_measure(self, *args, **kwargs):
                nonlocal active, peak
                active += 1
                peak = max(peak, active)
                try:
                    await asyncio.sleep(0.01)
                finally:
                    active -= 1

        evaluator = Evaluator(settings(max_concurrent=2), JsonJudge("", None))
        request = EvaluateRequest.model_validate(payload("tool_correctness"))
        with patch.object(
            evaluator, "_build_metric", side_effect=lambda _: TrackingMetric()
        ):
            results = await asyncio.gather(
                *(evaluator.evaluate(request) for _ in range(8))
            )
        self.assertEqual(peak, 2)
        self.assertTrue(all(result.summary.passed == 1 for result in results))
