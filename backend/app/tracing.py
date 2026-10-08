"""Small wrappers around the Langfuse client.

Everything here swallows its own exceptions. Tracing calls sit inside the
request path now -- the chat guardrail runs before every reply, and the upload
route records its own outcome -- so an error from the observability layer would
surface to a player as a failed request. Losing a span is the right trade
against losing the response it describes.

The existing update_current_generation calls in analyst and perception are
unguarded; those run inside a background job where a raised exception becomes a
failed job rather than a 500, so the exposure is different.
"""

from __future__ import annotations

import logging

from langfuse import get_client

logger = logging.getLogger(__name__)


def record(**fields) -> None:
    """Attach input/output/metadata to the current span."""
    try:
        get_client().update_current_span(**fields)
    except Exception:
        logger.debug("Could not record span fields", exc_info=True)


def score(name: str, value: str, comment: str | None = None) -> None:
    """Attach a categorical score to the current span.

    A score rather than only a metadata field because scores are what Langfuse
    aggregates: "how many messages did we refuse this week, and for what" is a
    chart over these, where the same value buried in metadata is a text search.
    """
    try:
        get_client().score_current_span(
            name=name, value=value, data_type="CATEGORICAL", comment=comment
        )
    except Exception:
        logger.debug("Could not record span score", exc_info=True)
