"""Tests for the Haiku-backed guardrails.

The classifier calls themselves are stubbed. What's worth testing here isn't
whether Haiku labels a sentence correctly -- that's a prompt-quality question,
checked by running it against real messages -- but the logic wrapped around the
label: that an unparseable answer fails open rather than guessing, that a
refused message never reaches the agent, that the repair attempt is bounded,
and that medical advice doesn't ship when repair fails.
"""

from __future__ import annotations

import pytest

from app.agent import analyst, chat
from app.guardrails import classify
from app.guardrails.classify import (
    MEDICAL_FALLBACK,
    REFUSALS,
    REPAIR_INSTRUCTIONS,
    ChatVerdict,
    FeedbackVerdict,
    review_feedback,
    screen_chat_message,
)


# --- label parsing and failing open ----------------------------------------

@pytest.mark.parametrize("raw,expected", [
    ("OK", ChatVerdict.OK),
    ("MEDICAL", ChatVerdict.MEDICAL),
    ("OFF_TOPIC", ChatVerdict.OFF_TOPIC),
    ("INJECTION", ChatVerdict.INJECTION),
    ("medical", ChatVerdict.MEDICAL),       # case
    ("MEDICAL.", ChatVerdict.MEDICAL),      # trailing punctuation
    (" INJECTION ", ChatVerdict.INJECTION),
])
def test_screening_parses_labels(raw, expected, monkeypatch):
    monkeypatch.setattr(classify, "_ask", lambda *a: raw)
    assert screen_chat_message("anything") is expected


@pytest.mark.parametrize("raw", [
    None,                                    # the call failed
    "I think this message is probably fine",  # prose instead of a label
    "",
    "MAYBE",
])
def test_screening_fails_open_on_anything_unparseable(raw, monkeypatch):
    """Fail open is the deliberate choice here, unlike the ffprobe gate.

    This check sits on top of a system prompt that already asks for the same
    behaviour, so one unscreened message is a smaller harm than every player
    losing chat because the classifier had a bad minute. A verdict invented
    from a malformed reply would be worse than abstaining.
    """
    monkeypatch.setattr(classify, "_ask", lambda *a: raw)
    assert screen_chat_message("my elbow hurts") is ChatVerdict.OK


@pytest.mark.parametrize("raw,expected", [
    ("OK", FeedbackVerdict.OK),
    ("MEDICAL", FeedbackVerdict.MEDICAL),
    ("UNGROUNDED", FeedbackVerdict.UNGROUNDED),
])
def test_review_parses_labels(raw, expected, monkeypatch):
    monkeypatch.setattr(classify, "_ask", lambda *a: raw)
    assert review_feedback("draft", "evidence") is expected


def test_review_fails_open(monkeypatch):
    monkeypatch.setattr(classify, "_ask", lambda *a: None)
    assert review_feedback("draft", "evidence") is FeedbackVerdict.OK


# --- completeness invariants ------------------------------------------------

def test_every_refusable_verdict_has_player_facing_copy():
    """A verdict without copy is a KeyError in the request path, which is how
    adding a category later turns into a 500 for the player who triggered it."""
    for verdict in ChatVerdict:
        if verdict is not ChatVerdict.OK:
            assert REFUSALS[verdict].strip()


def test_every_failing_verdict_has_a_repair_instruction():
    for verdict in FeedbackVerdict:
        if verdict is not FeedbackVerdict.OK:
            assert REPAIR_INSTRUCTIONS[verdict].strip()


# --- the analysis repair loop ----------------------------------------------

@pytest.fixture
def drafts(monkeypatch):
    """Records calls to the analysis model and returns canned drafts."""
    calls: list[dict] = []

    def fake_draft(user_content, previous=None, repair=None):
        calls.append({"previous": previous, "repair": repair})
        return f"draft-{len(calls)}"

    monkeypatch.setattr(analyst, "_draft", fake_draft)
    return calls


def _verdicts(monkeypatch, *sequence):
    """Make review_feedback return each verdict in turn."""
    remaining = list(sequence)
    monkeypatch.setattr(
        analyst, "review_feedback", lambda *a: remaining.pop(0) if remaining else FeedbackVerdict.OK
    )


def test_a_clean_draft_ships_without_a_repair(drafts, monkeypatch):
    _verdicts(monkeypatch, FeedbackVerdict.OK)
    assert analyst.analyze([{"contact_time_s": 1.0}]) == "draft-1"
    assert len(drafts) == 1


def test_a_failing_draft_is_repaired_once_and_ships(drafts, monkeypatch):
    _verdicts(monkeypatch, FeedbackVerdict.MEDICAL, FeedbackVerdict.OK)
    assert analyst.analyze([{"contact_time_s": 1.0}]) == "draft-2"
    assert len(drafts) == 2


def test_the_repair_call_sees_the_rejected_draft_and_the_reason(drafts, monkeypatch):
    """Asking for a rewrite while withholding the draft is just a reroll. The
    model needs both what it wrote and what was wrong with it."""
    _verdicts(monkeypatch, FeedbackVerdict.MEDICAL, FeedbackVerdict.OK)
    analyst.analyze([{"contact_time_s": 1.0}])
    assert drafts[1]["previous"] == "draft-1"
    assert drafts[1]["repair"] == REPAIR_INSTRUCTIONS[FeedbackVerdict.MEDICAL]


def test_medical_advice_does_not_ship_when_repair_fails(drafts, monkeypatch):
    """The one verdict that withholds rather than degrades."""
    _verdicts(monkeypatch, FeedbackVerdict.MEDICAL, FeedbackVerdict.MEDICAL)
    assert analyst.analyze([{"contact_time_s": 1.0}]) == MEDICAL_FALLBACK


def test_ungrounded_feedback_still_ships_when_repair_fails(drafts, monkeypatch):
    """Overreaching coaching is a worse write-up, not a harmful one, and
    shipping nothing after the player waited for an analysis is its own
    failure. Logged, not withheld."""
    _verdicts(monkeypatch, FeedbackVerdict.UNGROUNDED, FeedbackVerdict.UNGROUNDED)
    assert analyst.analyze([{"contact_time_s": 1.0}]) == "draft-2"


def test_repair_is_bounded_to_one_attempt(drafts, monkeypatch):
    """An unbounded loop on a background job is how a cheap check becomes an
    expensive one."""
    _verdicts(monkeypatch, *[FeedbackVerdict.UNGROUNDED] * 10)
    analyst.analyze([{"contact_time_s": 1.0}])
    assert len(drafts) == 2


# --- screening in the chat path --------------------------------------------

class FakeGraph:
    """Stands in for the compiled graph: records invokes and state updates."""

    def __init__(self) -> None:
        self.invoked: list[dict] = []
        self.updates: list[dict] = []

    def invoke(self, payload, config=None):
        self.invoked.append(payload)
        return {"messages": [type("M", (), {"content": "agent reply"})()]}

    def update_state(self, config, values, as_node=None):
        self.updates.append(values)


@pytest.fixture
def graph(monkeypatch) -> FakeGraph:
    fake = FakeGraph()
    monkeypatch.setattr(chat, "_graph", lambda: fake)
    return fake


@pytest.mark.parametrize("verdict", [
    ChatVerdict.MEDICAL,
    ChatVerdict.OFF_TOPIC,
    ChatVerdict.INJECTION,
])
def test_a_refused_message_never_reaches_the_agent(verdict, graph, monkeypatch):
    """The whole point of screening before invoke: a refused message touches
    no model, no tools and no video."""
    monkeypatch.setattr(chat, "screen_chat_message", lambda m: verdict)
    reply = chat.ask("job-1", "my elbow hurts")
    assert reply == REFUSALS[verdict]
    assert graph.invoked == []


def test_a_refused_turn_is_recorded_in_the_conversation(graph, monkeypatch):
    """The client refetches history after every send, so a reply that was only
    returned over HTTP would appear once and then vanish, leaving the player
    looking at their own question with no answer under it."""
    monkeypatch.setattr(chat, "screen_chat_message", lambda m: ChatVerdict.MEDICAL)
    chat.ask("job-1", "my elbow hurts")

    assert len(graph.updates) == 1
    recorded = graph.updates[0]["messages"]
    assert recorded[0].content == "my elbow hurts"
    assert recorded[1].content == REFUSALS[ChatVerdict.MEDICAL]
    assert graph.updates[0]["job_id"] == "job-1"


def test_an_allowed_message_reaches_the_agent(graph, monkeypatch):
    monkeypatch.setattr(chat, "screen_chat_message", lambda m: ChatVerdict.OK)
    assert chat.ask("job-1", "which swing was best?") == "agent reply"
    assert len(graph.invoked) == 1
    assert graph.updates == []


# --- record_turn against a real checkpointer -------------------------------
# FakeGraph proves ask() calls update_state; it can't prove update_state
# actually appends through the add_messages reducer and reads back out of
# history(). InMemorySaver exercises the real thing without needing Postgres.

def _memory_graph():
    from langgraph.checkpoint.memory import InMemorySaver
    from langgraph.graph import END, START, StateGraph

    graph = StateGraph(chat.ChatState)
    graph.add_node("agent", lambda state: {"messages": []})
    graph.add_edge(START, "agent")
    graph.add_edge("agent", END)
    return graph.compile(checkpointer=InMemorySaver())


def test_record_turn_round_trips_through_history(monkeypatch):
    graph = _memory_graph()
    monkeypatch.setattr(chat, "_graph", lambda: graph)
    config = {"configurable": {"thread_id": "job-x"}}

    chat.record_turn(graph, config, "job-x", "my elbow hurts", "see a physio")

    assert chat.history("job-x") == [
        {"role": "user", "content": "my elbow hurts"},
        {"role": "assistant", "content": "see a physio"},
    ]


def test_refused_turns_accumulate_rather_than_replace(monkeypatch):
    """add_messages is a reducer, so each update appends. If it replaced, a
    second refusal would wipe the conversation before it."""
    graph = _memory_graph()
    monkeypatch.setattr(chat, "_graph", lambda: graph)
    monkeypatch.setattr(chat, "screen_chat_message", lambda m: ChatVerdict.OFF_TOPIC)

    chat.ask("job-y", "capital of France?")
    chat.ask("job-y", "write me a poem")

    turns = chat.history("job-y")
    assert len(turns) == 4
    assert [t["role"] for t in turns] == ["user", "assistant", "user", "assistant"]
    assert turns[0]["content"] == "capital of France?"
    assert turns[2]["content"] == "write me a poem"


# --- the tracing helpers must never break a guardrail ----------------------

def test_tracing_helpers_swallow_failures(monkeypatch):
    """record() and score() run inside the request path, so an error from the
    observability layer would surface to a player as a failed request. Losing a
    span is the right trade against losing the response it describes."""
    from app import tracing

    class Broken:
        def update_current_span(self, **kw):
            raise RuntimeError("langfuse is down")

        def score_current_span(self, **kw):
            raise RuntimeError("langfuse is down")

    monkeypatch.setattr(tracing, "get_client", lambda: Broken())
    tracing.record(input="x", output="y")        # must not raise
    tracing.score("chat_screen", "MEDICAL")      # must not raise


def test_screening_still_works_when_langfuse_is_down(monkeypatch):
    """The guardrail's verdict must not depend on the trace landing.

    Breaks the Langfuse client itself and leaves the real record()/score()
    helpers in place, so the guarding in tracing.py is what has to hold.
    Stubbing the helpers out instead would prove nothing.
    """
    from app import tracing

    class Broken:
        def update_current_span(self, **kw):
            raise RuntimeError("langfuse is down")

        def score_current_span(self, **kw):
            raise RuntimeError("langfuse is down")

    monkeypatch.setattr(tracing, "get_client", lambda: Broken())
    monkeypatch.setattr(classify, "_ask", lambda *a: "MEDICAL")
    assert screen_chat_message("my elbow hurts") is ChatVerdict.MEDICAL
