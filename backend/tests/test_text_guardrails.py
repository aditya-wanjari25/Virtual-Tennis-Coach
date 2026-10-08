"""Tests for the chat message cap.

Enforced as a Pydantic constraint rather than a check in the endpoint, so what
needs proving is that FastAPI rejects an oversized body during validation --
i.e. before the route function runs and before anything reaches an LLM.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from app.guardrails import MAX_CHAT_CHARS
from app.main import ChatRequest, app

# No `with` block: TestClient only runs lifespan as a context manager, and the
# case below is refused during request validation, so the app never needs a
# database, a Langfuse client or a checkpointer pool to serve it.
client = TestClient(app)

CHAT_URL = "/videos/00000000-0000-0000-0000-000000000000/chat"


def test_cap_is_generous_enough_for_a_real_question():
    """A ceiling on abuse, not a budget for users -- nobody types 2000
    characters into a chat box, and a cap that cut into real questions would be
    a worse bug than the one it fixes."""
    assert MAX_CHAT_CHARS >= 500


def test_accepts_a_normal_question():
    body = ChatRequest(message="Was my elbow bent at contact?")
    assert body.message.startswith("Was my elbow")


def test_accepts_a_message_exactly_at_the_cap():
    assert len(ChatRequest(message="x" * MAX_CHAT_CHARS).message) == MAX_CHAT_CHARS


def test_rejects_a_message_one_character_over_the_cap():
    with pytest.raises(ValidationError):
        ChatRequest(message="x" * (MAX_CHAT_CHARS + 1))


def test_rejects_an_empty_message():
    """min_length=1 as well as max_length: an empty turn would still cost a
    full LLM call to answer with nothing."""
    with pytest.raises(ValidationError):
        ChatRequest(message="")


def test_http_rejects_an_oversized_message_during_validation():
    """Proves the cap fires at the HTTP boundary, not just on the model.

    Deliberately no happy-path counterpart here: a valid body would get past
    validation and into the route's job lookup, which needs a live database --
    environment coupling this suite is better off without. The route behaviour
    past validation isn't what the guardrail is responsible for.
    """
    res = client.post(CHAT_URL, json={"message": "x" * (MAX_CHAT_CHARS + 1)})
    assert res.status_code == 422
    assert "message" in res.text
