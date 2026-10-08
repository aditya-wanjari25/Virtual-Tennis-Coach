"""Guardrails on free text the player sends us.

Only a length cap for now. It's not about rejecting long questions -- 2000
characters is far more than anyone types into a chat box -- it's that
ChatRequest.message went straight into an LLM call with no bound at all, so a
single scripted request could run up an arbitrary token bill.

Enforced as a Pydantic constraint on the request model rather than as a check
in the endpoint, so FastAPI rejects it during validation with a 422 and the
route never sees it.
"""

from __future__ import annotations

import os


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ[name])
    except (KeyError, ValueError):
        return default


MAX_CHAT_CHARS = _env_int("MAX_CHAT_CHARS", 2000)
