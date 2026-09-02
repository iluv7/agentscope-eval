"""Strict parsing and JSON Schema checks for benchmark evidence."""

import json
import math
from typing import Any

from jsonschema import Draft202012Validator
from referencing import Registry, Resource
from referencing.jsonschema import DRAFT202012


def parse_json(raw: str) -> Any:
    """Parse without repair, rejecting duplicate keys and non-JSON numbers."""

    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError(f"Duplicate key: {key}")
            result[key] = value
        return result

    def number(value):
        parsed = float(value)
        if not math.isfinite(parsed):
            raise ValueError("Non-finite number")
        return parsed

    def constant(value):
        raise ValueError(f"Non-JSON number: {value}")

    return json.loads(
        raw,
        object_pairs_hook=pairs,
        parse_float=number,
        parse_constant=constant,
    )


def json_equal(left: Any, right: Any) -> bool:
    """Compare decoded JSON values, keeping booleans distinct from numbers."""
    if isinstance(left, bool) or isinstance(right, bool):
        return type(left) is type(right) and left == right
    if isinstance(left, dict) and isinstance(right, dict):
        return left.keys() == right.keys() and all(
            json_equal(left[key], right[key]) for key in left
        )
    if isinstance(left, list) and isinstance(right, list):
        return len(left) == len(right) and all(
            json_equal(a, b) for a, b in zip(left, right)
        )
    return left == right


def check_schema(schema: dict) -> dict:
    """Accept draft 2020-12 schemas with local references only."""
    dialect = schema.get("$schema")
    if dialect not in (None, "https://json-schema.org/draft/2020-12/schema"):
        raise ValueError("Only JSON Schema draft 2020-12 is supported")

    references = []

    def inspect(value, root=False):
        if isinstance(value, dict):
            if not root and "$id" in value:
                raise ValueError("Nested schema identifiers are unsupported")
            for key, child in value.items():
                if key in ("$ref", "$dynamicRef") and isinstance(child, str):
                    if not child.startswith("#"):
                        raise ValueError("Schema references must be local")
                    references.append(child)
                inspect(child)
        elif isinstance(value, list):
            for child in value:
                inspect(child)

    inspect(schema, root=True)
    Draft202012Validator.check_schema(schema)
    resolver = Registry().resolver_with_root(
        Resource.from_contents(schema, default_specification=DRAFT202012)
    )
    for reference in references:
        resolver.lookup(reference)
    return schema


def schema_matches(value: Any, schema: dict) -> bool:
    """Validate JSON against the registered schema without coercion."""
    return Draft202012Validator(schema, registry=Registry()).is_valid(value)
