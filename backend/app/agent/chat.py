"""Chat agent -- follow-up questions about an analysed video.

This is the part of the project that genuinely needs LangGraph. Unlike the
analysis path (a fixed sequence, now a plain function), chat:

  - LOOPS: agent -> tools -> agent -> ... until the model stops asking for tools
  - BRANCHES: a conditional edge decides tool-call vs done, per turn
  - has PERSISTENT STATE: the conversation survives between HTTP requests,
    checkpointed in Postgres and keyed by thread_id
  - lets the MODEL decide what happens next, rather than us hardcoding it

    START -> agent --(tool calls?)--> tools --+
               ^                              |
               +------------------------------+
               |
               +--(no)--> END
"""

from __future__ import annotations

import os
from functools import lru_cache
from typing import Annotated, TypedDict

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import AnyMessage, SystemMessage
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from psycopg_pool import ConnectionPool

from app.agent.chat_tools import CHAT_TOOLS

MODEL = "claude-sonnet-5"

SYSTEM_PROMPT = """\
You are the tennis coach who analysed this player's practice video, answering \
their follow-up questions. You already gave them written feedback; they can see \
it. Now they're asking about it.

You have tools to look things up. Use them rather than guessing:
- get_swing_metrics -- precise measurements for one swing
- get_observations -- what was seen in the video during the original analysis
- rewatch_swing -- watch a swing again to answer something specific

If a question is about something you weren't given, rewatch the swing rather
than speculating. If the footage genuinely can't answer it -- wrong camera
angle, occluded, out of frame -- say so plainly. Never invent a detail about
their technique.

Two things to carry over from the original feedback:
- Don't quote raw measurements at them. The numbers are for your reasoning.
  Our rotation figures are 2D projected angles from a back view, not the
  rotation a coach means -- absolute values aren't meaningful to a player.
  Translate to what they'd see or feel.
- Talk like a coach, not a report. Short, direct, specific. It's a
  conversation, so match their question's scope: a one-line question gets a
  one-line answer, not a lecture.
"""


class ChatState(TypedDict):
    # add_messages is a reducer: nodes return only NEW messages and LangGraph
    # appends them, rather than each node rebuilding the whole history.
    messages: Annotated[list[AnyMessage], add_messages]
    # Read by the tools via InjectedState; never shown to the model.
    job_id: str


@lru_cache(maxsize=1)
def _checkpointer() -> PostgresSaver:
    """Postgres-backed conversation persistence, one pool for the process.

    PostgresSaver.from_conn_string is a context manager, which doesn't suit a
    checkpointer that has to outlive a single request -- so we own the pool and
    construct the saver directly. Note it wants a raw libpq URL; DATABASE_URL is
    SQLAlchemy-flavoured (postgresql+psycopg://) and needs the driver stripped.
    """
    dsn = os.environ["DATABASE_URL"].replace("postgresql+psycopg://", "postgresql://")
    pool = ConnectionPool(conninfo=dsn, max_size=5, kwargs={"autocommit": True})
    saver = PostgresSaver(pool)
    saver.setup()  # idempotent; creates the checkpoint tables on first run
    _pools.append(pool)
    return saver


# Pools opened by _checkpointer(), so the app can close them on shutdown.
# Without this, psycopg_pool logs "couldn't stop thread ... within 5.0 seconds"
# at exit because its worker threads are still running.
_pools: list[ConnectionPool] = []


def close_pools() -> None:
    for pool in _pools:
        pool.close()
    _pools.clear()


@lru_cache(maxsize=1)
def _graph():
    model = ChatAnthropic(
        model=MODEL,
        max_tokens=2048,
        api_key=os.environ["ANTHROPIC_API_KEY"],
    ).bind_tools(CHAT_TOOLS)

    def agent(state: ChatState) -> dict:
        # System prompt is prepended per call rather than stored in state, so it
        # stays editable without rewriting every checkpointed conversation.
        response = model.invoke([SystemMessage(content=SYSTEM_PROMPT), *state["messages"]])
        return {"messages": [response]}

    graph = StateGraph(ChatState)
    graph.add_node("agent", agent)
    graph.add_node("tools", ToolNode(CHAT_TOOLS))

    graph.add_edge(START, "agent")
    # tools_condition routes to "tools" when the last message has tool calls,
    # otherwise to END. This is the conditional edge the analysis path never had.
    graph.add_conditional_edges("agent", tools_condition, {"tools": "tools", END: END})
    graph.add_edge("tools", "agent")  # the cycle

    return graph.compile(checkpointer=_checkpointer())


def _text_of(message) -> str:
    """Flatten a message's content to plain text.

    LangChain message content is either a string or a list of content blocks --
    Sonnet 5 runs adaptive thinking by default, so a reply often arrives as
    [{'type': 'thinking', ...}, {'type': 'text', 'text': ...}]. Returning that
    list raw fails response validation and would leak reasoning to the client.
    """
    content = message.content
    if isinstance(content, str):
        return content
    return "".join(
        block.get("text", "")
        for block in content
        if isinstance(block, dict) and block.get("type") == "text"
    )


def ask(job_id: str, message: str) -> str:
    """Send one user message; returns the assistant's reply.

    History is loaded and saved automatically by the checkpointer, keyed on
    thread_id -- so the caller passes only the new message, not the transcript.
    """
    config = {"configurable": {"thread_id": job_id}}
    result = _graph().invoke(
        {"messages": [{"role": "user", "content": message}], "job_id": job_id},
        config=config,
    )
    return _text_of(result["messages"][-1])


def history(job_id: str) -> list[dict]:
    """Past turns for a conversation, for rendering the UI on page load."""
    config = {"configurable": {"thread_id": job_id}}
    state = _graph().get_state(config)
    if not state.values:
        return []

    out: list[dict] = []
    for m in state.values.get("messages", []):
        role = getattr(m, "type", None)
        if role not in ("human", "ai"):
            continue  # skip tool calls/results -- internal, not conversation
        text = _text_of(m)
        if text.strip():
            out.append({"role": "user" if role == "human" else "assistant", "content": text})
    return out
