"""SQLAlchemy engine, sessions, and schema lifecycle helpers."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator

from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from .config import Settings, get_settings
from .models import Base


def create_engine_for_url(database_url: str, *, echo: bool = False) -> Engine:
    """Create a synchronous SQLAlchemy engine for PostgreSQL or SQLite.

    The application uses synchronous transactions around graph side effects.  A
    synchronous engine also keeps the persistence layer usable by migration and
    recovery scripts.  ``check_same_thread`` is only relevant for SQLite tests;
    PostgreSQL receives normal pooled connections.
    """

    connect_args: dict[str, object] = {}
    kwargs: dict[str, object] = {
        "echo": echo,
        "future": True,
        "pool_pre_ping": True,
    }
    if database_url.startswith("sqlite"):
        connect_args["check_same_thread"] = False
        kwargs["connect_args"] = connect_args
        # A single connection makes in-memory SQLite visible across sessions and
        # is harmless for the short-lived local test/runtime use case.
        if ":memory:" in database_url:
            from sqlalchemy.pool import StaticPool

            kwargs["poolclass"] = StaticPool
    return create_engine(database_url, **kwargs)


def create_engine_from_settings(settings: Settings | None = None) -> Engine:
    settings = settings or get_settings()
    return create_engine_for_url(settings.database_url, echo=settings.database_echo)


def session_factory(engine: Engine) -> sessionmaker[Session]:
    """Build a non-expiring session factory for request/worker transactions."""

    return sessionmaker(bind=engine, class_=Session, expire_on_commit=False, autoflush=False)


def init_db(engine: Engine) -> None:
    """Create missing tables.

    Production startup should run Alembic migrations.  ``create_all`` is useful
    for first-run local setup, tests, and recovery when no migration runner is
    installed yet; it never drops or alters existing tables.
    """

    Base.metadata.create_all(engine)


def drop_db(engine: Engine) -> None:
    """Drop all application tables (test/dev helper only)."""

    Base.metadata.drop_all(engine)


@contextmanager
def session_scope(factory: sessionmaker[Session]) -> Iterator[Session]:
    """Commit on success and roll back on any exception."""

    session = factory()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def make_session_factory(settings: Settings | None = None) -> tuple[Engine, sessionmaker[Session]]:
    """Convenience constructor used by the API bootstrap."""

    engine = create_engine_from_settings(settings)
    return engine, session_factory(engine)

