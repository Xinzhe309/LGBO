"""Runtime configuration for the public LGBO release.

Do not commit private API keys. Set API_KEY in the environment before online
runs, or pass --offline to run the deterministic smoke path.
"""

from __future__ import annotations

import os


BASE_URL = os.getenv("BASE_URL", "https://chat.intern-ai.org.cn")
ENDPOINT = os.getenv("ENDPOINT", "/api/v1/chat/completions")
MODEL_NAME = os.getenv("MODEL_NAME", os.getenv("INTERN_MODEL_NAME", "intern-s1"))
API_KEY = os.getenv("API_KEY", "")

MAX_TOKENS = int(os.getenv("MAX_TOKENS", "8192"))
TEMP = float(os.getenv("LLM_TEMP", "0.2"))
DO_SAMPLE = bool(int(os.getenv("DO_SAMPLE", "0")))
REQUEST_TIMEOUT = int(os.getenv("REQUEST_TIMEOUT", "600"))

BATCH_Q = int(os.getenv("BATCH_Q", "3"))
N_INIT = int(os.getenv("N_INIT", "2"))
PRINT_LIMIT = int(os.getenv("PRINT_LIMIT", "3000"))


def require_api_key() -> str:
    if not API_KEY:
        raise RuntimeError("API_KEY is not set. Export API_KEY or run with --offline.")
    return API_KEY
