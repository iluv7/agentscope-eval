"""AgentScope message compatibility and execution outcome regression tests."""

import json
import unittest
from pathlib import Path

from fastapi.testclient import TestClient

from agentscope_eval.agentscope import AgentScopeCase
from agentscope_eval.api import create_app
from agentscope_eval.config import Settings


def fixture():
    path = Path(__file__).resolve().parents[1] / "examples/agentscope.json"
    return json.loads(path.read_text())


class AgentScopeTests(unittest.TestCase):
    def setUp(self):
        config = Settings(_env_file=None, judge_model="", judge_api_key="")
        self.client = self.enterContext(TestClient(create_app(config)))

    def test_native_msg_fixture_passes(self):
        response = self.client.post("/v1/agentscope/evaluate", json=fixture())
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.json()["summary"]["passed"], 1)
        self.assertEqual(len(response.json()["results"][0]["metrics"]), 3)

    def test_final_answer_excludes_plans_and_thinking(self):
        case = AgentScopeCase.model_validate(fixture()["cases"][0])
        normalized = case.to_eval_case()
        self.assertEqual(normalized.actual_output, case.expected_output)
        self.assertEqual(
            normalized.tools_called[0].input_parameters, {"order_id": "A123"}
        )

    def test_results_are_joined_by_id_not_arrival_order(self):
        record = fixture()["cases"][0]
        record["reply"]["content"] = [
            {"type": "tool_call", "id": "a", "name": "first", "input": "{}"},
            {"type": "tool_call", "id": "b", "name": "second", "input": "{}"},
            {
                "type": "tool_result",
                "id": "b",
                "name": "second",
                "state": "success",
                "output": "second-result",
            },
            {
                "type": "tool_result",
                "id": "a",
                "name": "first",
                "state": "success",
                "output": "first-result",
            },
            {"type": "text", "text": "done"},
        ]
        normalized = AgentScopeCase.model_validate(record).to_eval_case()
        self.assertEqual(
            [tool.output for tool in normalized.tools_called],
            ["first-result", "second-result"],
        )

    def test_failed_agent_execution_is_a_failed_metric(self):
        for reason in ["interrupted", "error", "exceed_max_iters"]:
            with self.subTest(reason=reason):
                request = fixture()
                request["cases"][0]["reply"]["finished_reason"] = reason
                response = self.client.post(
                    "/v1/agentscope/evaluate", json=request
                )
                result = response.json()
                self.assertEqual(result["summary"]["failed"], 1)
                self.assertEqual(result["summary"]["errors"], 0)
                self.assertEqual(
                    result["results"][0]["metrics"][0]["score"], 0
                )

    def test_denied_failed_and_missing_results_do_not_pass_execution(self):
        for state in ["denied", "error", "interrupted", "running", None]:
            with self.subTest(state=state):
                request = fixture()
                blocks = request["cases"][0]["reply"]["content"]
                if state is None:
                    blocks.pop(3)
                else:
                    blocks[3]["state"] = state
                result = self.client.post(
                    "/v1/agentscope/evaluate", json=request
                ).json()
                self.assertEqual(
                    result["results"][0]["metrics"][0]["score"], 0
                )

    def test_no_tool_reply_can_succeed(self):
        request = fixture()
        case = request["cases"][0]
        case["reply"]["content"] = [{"type": "text", "text": "hello"}]
        case["expected_output"] = "hello"
        case["expected_tools"] = []
        result = self.client.post(
            "/v1/agentscope/evaluate", json=request
        ).json()
        self.assertEqual(result["summary"]["passed"], 1)

    def test_incomplete_or_inconsistent_records_are_rejected(self):
        mutations = [
            lambda r: r.update(finished_reason=None),
            lambda r: r["content"][2].update(input="not json"),
            lambda r: r["content"][2].update(input="[]"),
            lambda r: r["content"][3].update(id="unknown"),
            lambda r: r["content"][3].update(name="other-tool"),
            lambda r: r["content"].append(r["content"][2].copy()),
        ]
        for mutate in mutations:
            with self.subTest(mutate=mutate):
                request = fixture()
                mutate(request["cases"][0]["reply"])
                response = self.client.post(
                    "/v1/agentscope/evaluate", json=request
                )
                self.assertEqual(response.status_code, 422)
