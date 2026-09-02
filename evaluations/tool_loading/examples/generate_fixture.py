"""Generate synthetic observations for exercising the evaluator, not models."""

import json
from copy import deepcopy
from pathlib import Path

TOOL = {
    "name": "submit_payload",
    "input_schema": {
        "$defs": {
            "node": {
                "type": "object",
                "properties": {
                    "label": {"type": "string"},
                    "children": {
                        "type": "array",
                        "items": {"$ref": "#/$defs/node"},
                    },
                },
                "required": ["label", "children"],
                "additionalProperties": False,
            },
        },
        "type": "object",
        "properties": {
            "text": {"type": "string"},
            "items": {"type": "array", "items": {"type": "string"}},
            "options": {
                "type": "object",
                "properties": {
                    "enabled": {"type": "boolean"},
                    "limit": {"type": "integer"},
                    "note": {"type": ["string", "null"]},
                },
                "required": ["enabled", "limit", "note"],
                "additionalProperties": False,
            },
            "tree": {"$ref": "#/$defs/node"},
        },
        "required": ["text", "items", "options", "tree"],
        "additionalProperties": False,
    },
}
BASE = {
    "text": "hello",
    "items": [],
    "options": {"enabled": True, "limit": 3, "note": None},
    "tree": {"label": "root", "children": []},
}


def trial(scenario, arguments, variant="dispatcher", scope="generation"):
    """Build a synthetic successful observation with explicit provenance."""
    raw = (
        arguments
        if variant == "native"
        else {
            "tool_name": TOOL["name"],
            "arguments": arguments,
        }
    )
    attempt = {
        "call": {
            "call_id": "call-1",
            "name": TOOL["name"] if variant == "native" else "execute_tool",
            "arguments": json.dumps(raw, ensure_ascii=False),
        }
    }
    if scope == "trajectory":
        attempt["execution"] = {
            "call_id": "call-1",
            "tool_name": TOOL["name"],
            "arguments": deepcopy(arguments),
            "status": "success",
            "output": {"accepted": True},
        }
    if variant == "discovery":
        attempt["search_call"] = {
            "call_id": "search-1",
            "name": "search_tools",
            "arguments": '{"query":"submit a structured payload"}',
        }
        attempt["search_result"] = {
            "call_id": "search-1",
            "status": "success",
            "raw_output": json.dumps({"tools": [TOOL]}),
        }
    return {
        "run_id": f"{variant}-{scenario}",
        "case_id": scenario,
        "input": "Submit this payload exactly, preserving all values:\n"
        + json.dumps(arguments, ensure_ascii=False),
        "scenario": scenario,
        "configuration": {
            "provider": "fixture",
            "model": "synthetic-observations",
            "source": "fixture",
            "variant": variant,
            "evaluation_scope": scope,
            "schema_placement": "tools" if variant == "native" else "history",
            "agentscope_version": "2.0.7.post1",
            "catalog_version": "fixture-v1",
            "prompt_version": "fixture-v1",
        },
        "target_tool": TOOL["name"],
        "expected_arguments": deepcopy(arguments),
        "expected_output": {"accepted": True}
        if scope == "trajectory"
        else None,
        "attempts": [attempt],
    }


def build_fixture():
    """Return argument stress cases and injected failure/recovery examples."""
    deep = {"label": "leaf", "children": []}
    for index in range(8):
        deep = {"label": f"level-{index}", "children": [deep]}
    scenarios = {
        "escaping": {
            **BASE,
            "text": 'quote " slash \\ path C:\\tmp\nline\tend',
        },
        "unicode": {**BASE, "text": "你好 café 🚀", "items": ["α", "β"]},
        "code_string": {
            **BASE,
            "text": 'def f():\n    return {"key": "value"}\n',
        },
        "arrays": {**BASE, "items": ["", "a", '"quoted"', "line\nbreak"]},
        "nested_depth_8": {**BASE, "tree": deep},
        "long_string": {**BASE, "text": ('line "quote" \\ end\n' * 80)},
    }
    trials = [
        trial(name, arguments, variant)
        for variant in ["native", "dispatcher"]
        for name, arguments in scenarios.items()
    ]

    repaired = trial("repair", BASE)
    attempt = repaired["attempts"][0]
    attempt["repaired_arguments"] = attempt["call"]["arguments"]
    attempt["call"]["arguments"] = '{"tool_name":"submit_payload","arguments":'
    trials.append(repaired)

    retry = trial("retry", BASE)
    retry["attempts"].insert(
        0,
        {
            "call": {
                "call_id": "failed-attempt",
                "name": "execute_tool",
                "arguments": "{",
            }
        },
    )
    trials.append(retry)

    missing = trial("no_call", BASE)
    missing["attempts"] = []
    trials.append(missing)

    trials.append(trial("full_pipeline", BASE, "discovery", "trajectory"))
    missed = trial("search_miss", BASE, "discovery", "trajectory")
    missed["attempts"][0]["search_result"]["raw_output"] = '{"tools":[]}'
    del missed["attempts"][0]["call"]
    del missed["attempts"][0]["execution"]
    trials.append(missed)

    failed = trial("tool_error", BASE, "discovery", "trajectory")
    failed["attempts"][0]["execution"].update(
        status="error", output="injected failure"
    )
    trials.append(failed)
    return {"tools": [TOOL], "trials": trials, "search_k": 5}


if __name__ == "__main__":
    destination = Path(__file__).with_name("fixture.json")
    destination.write_text(
        json.dumps(build_fixture(), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(destination)
