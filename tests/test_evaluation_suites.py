"""Integration coverage for adding suites without editing the service."""

import importlib
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

import evaluations
from agentscope_eval.api import create_app
from agentscope_eval.config import Settings


class SuiteDiscoveryTests(unittest.TestCase):
    def test_new_suite_is_served_and_cli_only_suite_is_ignored(self):
        names = ["_discovery_test_api", "_discovery_test_cli"]
        with tempfile.TemporaryDirectory() as directory:
            for name in names:
                suite = Path(directory) / name
                suite.mkdir()
                (suite / "__init__.py").write_text("")
            (Path(directory) / names[0] / "api.py").write_text(
                "from fastapi import APIRouter\n"
                "router = APIRouter()\n"
                "@router.get('/test/new-suite')\n"
                "def result():\n"
                "    return {'suite': 'discovered'}\n"
            )
            try:
                with patch.object(
                    evaluations,
                    "__path__",
                    [*evaluations.__path__, directory],
                ):
                    importlib.invalidate_caches()
                    settings = Settings(
                        _env_file=None, judge_model="", judge_api_key=""
                    )
                    with TestClient(create_app(settings)) as client:
                        response = client.get("/test/new-suite")
                        self.assertEqual(response.status_code, 200)
                        self.assertEqual(
                            response.json(), {"suite": "discovered"}
                        )
                        paths = client.get("/openapi.json").json()["paths"]
                        self.assertIn(
                            "/v1/benchmarks/tool-loading/evaluate", paths
                        )
            finally:
                for name in names:
                    sys.modules.pop(f"evaluations.{name}.api", None)
                    sys.modules.pop(f"evaluations.{name}", None)
