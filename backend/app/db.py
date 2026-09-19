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


@lru_cache
def get_engine():
    # Lazy + cached: reads DATABASE_URL on first use, not at import time,
    # so callers can load_dotenv() first regardless of import order.
    return create_engine(os.environ["DATABASE_URL"])


def get_session() -> Session:
    return sessionmaker(bind=get_engine())()
