"""Evaluate AgentScope execution records and answer quality."""

import os

# Keep the standalone service local; no DeepEval login or artifact store.
for _key, _value in {
    "DEEPEVAL_TELEMETRY_OPT_OUT": "1",
    "DEEPEVAL_FILE_SYSTEM": "READ_ONLY",
    "DEEPEVAL_DISABLE_DOTENV": "1",
    "DEEPEVAL_DISABLE_LEGACY_KEYFILE": "1",
    "CONFIDENT_TRACE_INTERNAL": "0",
    "CONFIDENT_TRACE_FLUSH": "0",
}.items():
    os.environ.setdefault(_key, _value)

__version__ = "0.1.0"
