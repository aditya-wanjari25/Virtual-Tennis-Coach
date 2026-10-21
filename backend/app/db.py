"""SQLAlchemy engine/session setup. One module, imported everywhere that
needs a DB session -- keeps engine creation (which reads DATABASE_URL)
in exactly one place."""

from __future__ import annotations

import os
from functools import lru_cache

from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker


class Base(DeclarativeBase):
    pass


# Driver-less prefixes a hosting platform might hand us. Fly's `postgres
# attach` sets DATABASE_URL to `postgres://...`; other providers use
# `postgresql://...`. Under SQLAlchemy the first fails to resolve a dialect at
# all and the second resolves to psycopg2, which this project does not install
# -- only psycopg v3. Either way the app dies on its first database call, and
# in the container that call is `alembic upgrade head` in the CMD, so the
# machine never finishes booting.
_DRIVERLESS_PREFIXES = ("postgres://", "postgresql://")
_SQLALCHEMY_PREFIX = "postgresql+psycopg://"


def sqlalchemy_url() -> str:
    """DATABASE_URL in the form SQLAlchemy needs, whatever form it arrives in.

    Normalising here rather than at deploy time means the app works with the
    value the platform sets, instead of depending on someone remembering to
    rewrite a secret by hand. A URL that already names a driver is left alone
    -- that is an explicit choice and not ours to override.
    """
    url = os.environ["DATABASE_URL"]
    if "+" in url.split("://", 1)[0]:
        return url
    for prefix in _DRIVERLESS_PREFIXES:
        if url.startswith(prefix):
            return _SQLALCHEMY_PREFIX + url[len(prefix):]
    return url


def libpq_url() -> str:
    """DATABASE_URL in the plain form libpq wants.

    LangGraph's PostgresSaver talks to psycopg directly and rejects the
    SQLAlchemy-flavoured `+psycopg` dialect suffix, so it needs the inverse of
    the above. Both conversions live here so there is one place that knows how
    this URL is spelled.
    """
    return sqlalchemy_url().replace(_SQLALCHEMY_PREFIX, "postgresql://", 1)


@lru_cache
def get_engine():
    # Lazy + cached: reads DATABASE_URL on first use, not at import time,
    # so callers can load_dotenv() first regardless of import order.
    return create_engine(sqlalchemy_url())


def get_session() -> Session:
    return sessionmaker(bind=get_engine())()
