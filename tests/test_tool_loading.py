"""Regression tests for raw JSON reliability and four-stage denominators."""

import copy
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from fastapi.testclient import TestClient
from pydantic import ValidationError

from agentscope_eval.api import create_app
from agentscope_eval.benchmarks.agentscope import ToolLoadingRecorder
from agentscope_eval.benchmarks.cli import render_markdown
from agentscope_eval.benchmarks.schemas import BenchmarkRequest, Execution
from agentscope_eval.benchmarks.tool_loading import evaluate_tool_loading
from agentscope_eval.config import Settings

ROOT = Path(__file__).resolve().parents[1]
TOOL = {
    "name": "save",
    "input_schema": {
        "type": "object",
        "properties": {
            "text": {"type": "string"},
            "items": {"type": "array", "items": {"type": "integer"}},
            "settings": {
                "type": "object",
                "properties": {"enabled": {"type": "boolean"}},
                "required": ["enabled"],
                "additionalProperties": False,
            },
        },
        "required": ["text", "items", "settings"],
        "additionalProperties": False,
    },
}
ARGS = {
    "text": 'quote " backslash \\ newline\n雪',
    "items": [1, 2],
    "settings": {"enabled": True},
}


def request(variant="dispatcher"):
    arguments = (
        ARGS
        if variant == "native"
        else {"tool_name": "save", "arguments": ARGS}
    )
    attempt = {
        "call": {
            "call_id": "call1",
            "name": "save" if variant == "native" else "execute_tool",
            "arguments": json.dumps(arguments),
        },
        "execution": {
            "call_id": "call1",
            "tool_name": "save",
            "arguments": copy.deepcopy(ARGS),
            "status": "success",
            "output": {"saved": True},
        },
    }
    if variant == "discovery":
        attempt["search_call"] = {
            "call_id": "search1",
            "name": "search_tools",
            "arguments": '{"query":"save text"}',
        }
        attempt["search_result"] = {
            "call_id": "search1",
            "status": "success",
            "raw_output": json.dumps({"tools": [TOOL]}),
        }
    return {
        "tools": [copy.deepcopy(TOOL)],
        "trials": [
            {
                "run_id": "run1",
                "case_id": "case1",
                "input": "Save the supplied text, items and settings exactly.",
                "scenario": "escaping",
                "configuration": {
                    "provider": "fixture",
                    "model": "synthetic",
                    "variant": variant,
                    "schema_placement": "tools"
                    if variant == "native"
                    else "history",
                    "agentscope_version": "2.0.7.post1",
                    "source": "fixture",
                },
                "target_tool": "save",
                "expected_arguments": copy.deepcopy(ARGS),
                "expected_output": {"saved": True},
                "attempts": [attempt],
            }
        ],
    }


def score(payload):
    return evaluate_tool_loading(BenchmarkRequest.model_validate(payload))


class BenchmarkTests(unittest.TestCase):
    def test_generation_scope_does_not_require_execution(self):
        payload = request()
        trial = payload["trials"][0]
        trial["configuration"]["evaluation_scope"] = "generation"
        del trial["attempts"][0]["execution"]
        del trial["expected_output"]
        result = score(payload)
        self.assertEqual(result.trials[0].status, "passed")
        self.assertEqual(
            result.summary.checkpoints["execution"].not_applicable, 1
        )
        self.assertIsNone(result.summary.checkpoints["execution"].rate)

    def test_generation_scope_cannot_silently_discard_execution(self):
        payload = request()
        payload["trials"][0]["configuration"]["evaluation_scope"] = (
            "generation"
        )
        with self.assertRaises(ValidationError):
            BenchmarkRequest.model_validate(payload)

    def test_synthetic_fixture_has_expected_failure_and_recovery_counts(self):
        payload = json.loads(
            (ROOT / "examples/tool_loading/fixture.json").read_text()
        )
        report = score(payload)
        self.assertEqual(report.summary.runs, 18)
        self.assertEqual(report.summary.metrics["eventual_success"].passed, 15)
        self.assertEqual(
            report.summary.metrics["first_attempt_success"].passed, 13
        )
        self.assertEqual(report.summary.metrics["repair_recovery"].passed, 1)
        self.assertEqual(report.summary.metrics["retry_recovery"].passed, 1)

    def test_native_dispatcher_and_discovery_success(self):
        for variant in ["native", "dispatcher", "discovery"]:
            with self.subTest(variant=variant):
                result = score(request(variant))
                self.assertEqual(result.trials[0].status, "passed")
                self.assertEqual(
                    result.summary.metrics["first_attempt_success"].rate, 1
                )
                self.assertIsNone(result.summary.telemetry["mean_duration_ms"])

    def test_malformed_json_is_a_failed_observation_not_http_422(self):
        payload = request()
        raw = '{"tool_name":"save","arguments":'
        payload["trials"][0]["attempts"][0] = {
            "call": {
                "call_id": "broken",
                "name": "execute_tool",
                "arguments": raw,
            }
        }
        config = Settings(_env_file=None, judge_model="", judge_api_key="")
        with TestClient(create_app(config)) as client:
            response = client.post(
                "/v1/benchmarks/tool-loading/evaluate", json=payload
            )
        self.assertEqual(response.status_code, 200, response.text)
        trial = response.json()["trials"][0]
        self.assertEqual(trial["status"], "failed")
        self.assertEqual(
            trial["attempts"][0]["evidence"]["call"]["arguments"], raw
        )
        self.assertFalse(
            trial["attempts"][0]["generation"]["checks"]["json_valid"]
        )

    def test_strict_json_duplicate_keys_and_non_finite_numbers(self):
        for raw in [
            '{"a":1,"a":2}',
            '{"a":NaN}',
            '{"a":Infinity}',
            '{"a":1e999}',
            "```json\n{}\n```",
            '{"a":1,}',
        ]:
            with self.subTest(raw=raw):
                payload = request()
                payload["trials"][0]["attempts"][0]["call"]["arguments"] = raw
                result = score(payload)
                self.assertEqual(
                    result.summary.metrics["first_json_valid"].rate, 0
                )

    def test_stringified_nested_arguments_fail_envelope_validation(self):
        payload = request()
        payload["trials"][0]["attempts"][0]["call"]["arguments"] = json.dumps(
            {"tool_name": "save", "arguments": json.dumps(ARGS)}
        )
        generation = score(payload).trials[0].attempts[0].generation
        self.assertTrue(generation.checks["json_valid"])
        self.assertFalse(generation.checks["envelope_valid"])

    def test_schema_and_value_errors_are_distinct(self):
        for changed, schema_valid in [
            ({"text": 9}, False),
            ({"text": "wrong"}, True),
        ]:
            with self.subTest(changed=changed):
                payload = request()
                arguments = {**ARGS, **changed}
                payload["trials"][0]["attempts"][0]["call"]["arguments"] = (
                    json.dumps({"tool_name": "save", "arguments": arguments})
                )
                check = score(payload).trials[0].attempts[0].generation
                self.assertEqual(
                    check.checks["tool_schema_valid"], schema_valid
                )
                self.assertFalse(check.checks["arguments_correct"])

    def test_boolean_is_not_equal_to_one(self):
        payload = request()
        arguments = copy.deepcopy(ARGS)
        arguments["settings"]["enabled"] = 1
        payload["trials"][0]["attempts"][0]["call"]["arguments"] = json.dumps(
            {"tool_name": "save", "arguments": arguments}
        )
        result = score(payload).trials[0].attempts[0].generation
        self.assertFalse(result.checks["arguments_correct"])

    def test_repair_does_not_overwrite_first_pass(self):
        payload = request()
        attempt = payload["trials"][0]["attempts"][0]
        attempt["repaired_arguments"] = attempt["call"]["arguments"]
        attempt["call"]["arguments"] = '{"tool_name":'
        result = score(payload)
        for name in ["first_json_valid", "first_attempt_success"]:
            self.assertEqual(result.summary.metrics[name].rate, 0)
        for name in ["repair_recovery", "eventual_success"]:
            self.assertEqual(result.summary.metrics[name].rate, 1)
        self.assertEqual(
            result.trials[0].attempts[0].evidence.call.arguments,
            '{"tool_name":',
        )

    def test_retry_is_not_first_attempt_success(self):
        payload = request()
        good = payload["trials"][0]["attempts"][0]
        payload["trials"][0]["attempts"] = [
            {
                "call": {
                    "call_id": "bad",
                    "name": "execute_tool",
                    "arguments": "{",
                }
            },
            good,
        ]
        result = score(payload)
        self.assertEqual(
            result.summary.metrics["first_attempt_success"].rate, 0
        )
        self.assertEqual(result.summary.metrics["retry_recovery"].rate, 1)
        self.assertEqual(result.summary.checkpoints["generation"].eligible, 2)

    def test_missing_call_is_counted_and_downstream_not_reached(self):
        payload = request()
        payload["trials"][0]["attempts"] = []
        result = score(payload)
        self.assertEqual(result.trials[0].captured_attempts, 0)
        self.assertEqual(result.summary.metrics["first_json_valid"].failed, 1)
        self.assertEqual(
            result.summary.metrics["first_json_valid"].eligible, 1
        )
        self.assertEqual(
            result.summary.checkpoints["execution"].not_reached, 1
        )
        self.assertEqual(result.summary.checkpoints["execution"].rate, 0)
        self.assertIsNone(result.summary.checkpoints["execution"].reached_rate)
        self.assertIsNone(result.summary.checkpoints["search_response"].rate)

    def test_failed_search_does_not_drop_the_trial(self):
        payload = request("discovery")
        payload["trials"][0]["attempts"] = [
            {
                "search_call": {
                    "call_id": "s",
                    "name": "search_tools",
                    "arguments": "bad",
                }
            }
        ]
        result = score(payload)
        self.assertEqual(
            result.summary.checkpoints["search_response"].not_reached, 1
        )
        self.assertEqual(
            result.summary.checkpoints["generation"].not_reached, 1
        )
        self.assertEqual(
            result.summary.metrics["eventual_success"].eligible, 1
        )

    def test_recall_and_schema_integrity_are_separate(self):
        payload = request("discovery")
        corrupted = copy.deepcopy(TOOL)
        corrupted["input_schema"]["required"] = []
        payload["trials"][0]["attempts"][0]["search_result"]["raw_output"] = (
            json.dumps({"tools": [corrupted]})
        )
        result = score(payload)
        response = result.trials[0].attempts[0].search_response
        self.assertEqual(response.recall_at_k, 1)
        self.assertFalse(response.checks["schemas_match_registry"])
        self.assertEqual(result.trials[0].status, "failed")

    def test_search_cutoff_and_bad_result(self):
        payload = request("discovery")
        other = {"name": "other", "input_schema": TOOL["input_schema"]}
        payload["tools"].append(other)
        payload["search_k"] = 1
        payload["trials"][0]["attempts"][0]["search_result"]["raw_output"] = (
            json.dumps({"tools": [other, TOOL]})
        )
        self.assertEqual(
            score(payload).summary.metrics["search_recall_at_k"].rate, 0
        )
        payload["trials"][0]["attempts"][0]["search_result"]["raw_output"] = (
            "not json"
        )
        self.assertEqual(
            score(payload).trials[0].attempts[0].search_response.status,
            "failed",
        )

    def test_execution_requires_real_output_and_dispatch_evidence(self):
        for change in [
            {"tool_name": "other"},
            {"arguments": {}},
            {"output": {"saved": False}},
            {"status": "denied"},
        ]:
            with self.subTest(change=change):
                payload = request()
                payload["trials"][0]["attempts"][0]["execution"].update(change)
                result = score(payload)
                self.assertEqual(
                    result.trials[0].attempts[0].generation.status, "passed"
                )
                self.assertEqual(result.trials[0].status, "failed")

    def test_missing_execution_is_a_failure_after_valid_generation(self):
        payload = request()
        del payload["trials"][0]["attempts"][0]["execution"]
        self.assertEqual(
            score(payload).summary.checkpoints["execution"].failed, 1
        )

    def test_provider_errors_are_separate_but_stay_in_denominator(self):
        payload = request()
        payload["trials"][0].update(outcome="provider_error", attempts=[])
        report = score(payload)
        self.assertEqual(report.summary.errors, 1)
        self.assertEqual(
            report.summary.metrics["eventual_success"].eligible, 1
        )
        self.assertEqual(report.summary.metrics["eventual_success"].rate, 0)

    def test_grouping_keeps_configuration_changes_separate(self):
        payload = request()
        second = copy.deepcopy(payload["trials"][0])
        second["run_id"] = "run2"
        second["configuration"]["strict_output"] = True
        payload["trials"].append(second)
        result = score(payload)
        self.assertEqual(
            len([g for g in result.groups if g.scenario is None]), 2
        )
        self.assertEqual(result.summary.runs, 2)
        self.assertIn("fixture", render_markdown(result))
        self.assertEqual(result.trials[0].expected_arguments, ARGS)

    def test_invalid_benchmark_contracts_are_rejected(self):
        for mutate in [
            lambda p: p["trials"].append(copy.deepcopy(p["trials"][0])),
            lambda p: p["trials"][0].update(expected_arguments={}),
            lambda p: p["trials"][0]["attempts"][0]["execution"].update(
                call_id="wrong"
            ),
            lambda p: p["tools"][0].update(
                input_schema={"type": "not-a-type"}
            ),
            lambda p: p["tools"][0].update(
                input_schema={"$ref": "https://example.com/schema"}
            ),
            lambda p: p["tools"][0].update(
                input_schema={
                    "type": "object",
                    "properties": {"unused": {"$ref": "#/$defs/missing"}},
                }
            ),
            lambda p: p["trials"][0].update(expected_output=float("nan")),
        ]:
            with self.subTest(mutate=mutate):
                payload = request()
                mutate(payload)
                with self.assertRaises(ValidationError):
                    BenchmarkRequest.model_validate(payload)

    def test_local_schema_references_are_supported(self):
        payload = request()
        payload["tools"][0]["input_schema"] = {
            "$defs": {"args": TOOL["input_schema"]},
            "$ref": "#/$defs/args",
        }
        self.assertEqual(score(payload).trials[0].status, "passed")

    def test_cli_json_markdown_and_failure_exit_code(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "input.json"
            output = Path(directory) / "report.json"
            markdown = Path(directory) / "report.md"
            payload = request()
            payload["trials"][0]["attempts"] = []
            source.write_text(json.dumps(payload))
            process = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "agentscope_eval.benchmarks.cli",
                    str(source),
                    "--output",
                    str(output),
                    "--markdown",
                    str(markdown),
                    "--fail-on-failure",
                ],
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(process.returncode, 1, process.stderr)
            self.assertEqual(
                json.loads(output.read_text())["summary"]["runs"], 1
            )
            self.assertIn("0/1", markdown.read_text())


class RecorderTests(unittest.TestCase):
    def test_records_raw_fragments_repair_and_real_execution_separately(self):
        recorder = ToolLoadingRecorder("reply1")
        events = [
            {"type": "TOOL_CALL_START", "tool_call_name": "execute_tool"},
            {"type": "TOOL_CALL_DELTA", "delta": '{"tool_name":'},
            {"type": "TOOL_CALL_END"},
        ]
        for index, event in enumerate(events):
            event.update(
                reply_id="reply1", tool_call_id="call1", id=str(index)
            )
            recorder.observe(event)
            recorder.observe(event)
        before = recorder.attempt("call1")
        self.assertEqual(before.call.arguments, '{"tool_name":')
        self.assertIsNone(before.execution)
        recorder.record_repair(
            "call1", {"tool_name": "save", "arguments": ARGS}
        )
        recorder.record_execution(
            Execution(
                call_id="call1",
                tool_name="save",
                arguments=ARGS,
                status="success",
                output={"saved": True},
            )
        )
        after = recorder.attempt("call1")
        self.assertEqual(after.call.arguments, before.call.arguments)
        self.assertIsNotNone(after.repaired_arguments)
        self.assertIsNone(before.repaired_arguments)
        self.assertEqual(after.execution.tool_name, "save")

    def test_reply_filtering_truncation_and_optional_telemetry(self):
        recorder = ToolLoadingRecorder()
        recorder.observe({"type": "REPLY_START", "reply_id": "r"})
        recorder.observe(
            {
                "type": "TOOL_CALL_START",
                "reply_id": "other",
                "tool_call_id": "ignored",
                "tool_call_name": "x",
            }
        )
        recorder.observe(
            {
                "type": "TOOL_CALL_START",
                "reply_id": "r",
                "tool_call_id": "c",
                "tool_call_name": "x",
            }
        )
        recorder.observe(
            {
                "type": "TOOL_CALL_DELTA",
                "reply_id": "r",
                "tool_call_id": "c",
                "delta": "{}",
            }
        )
        self.assertEqual(len(recorder.calls), 1)
        self.assertFalse(recorder.attempt("c").call.complete)
        self.assertIsNone(recorder.telemetry().input_tokens)
        recorder.observe(
            {
                "type": "MODEL_CALL_END",
                "reply_id": "r",
                "input_tokens": 100,
                "output_tokens": 10,
                "cache_input_tokens": 80,
            }
        )
        recorder.record_model_tools([TOOL])
        recorder.record_model_tools([TOOL])
        telemetry = recorder.telemetry()
        self.assertEqual(telemetry.cache_read_tokens, 80)
        self.assertEqual(
            telemetry.top_level_tools_hashes[0],
            telemetry.top_level_tools_hashes[1],
        )

    def test_search_result_mapping_uses_call_ids(self):
        recorder = ToolLoadingRecorder("r")
        for call_id in ["a", "b"]:
            recorder.observe(
                {
                    "type": "TOOL_CALL_START",
                    "reply_id": "r",
                    "tool_call_id": call_id,
                    "tool_call_name": "search_tools",
                }
            )
        for call_id, output in [("b", "second"), ("a", "first")]:
            recorder.observe(
                {
                    "type": "TOOL_RESULT_TEXT_DELTA",
                    "reply_id": "r",
                    "tool_call_id": call_id,
                    "delta": output,
                }
            )
            recorder.observe(
                {
                    "type": "TOOL_RESULT_END",
                    "reply_id": "r",
                    "tool_call_id": call_id,
                    "state": "success",
                }
            )
        self.assertEqual(
            recorder.attempt(None, "a").search_result.raw_output, "first"
        )
