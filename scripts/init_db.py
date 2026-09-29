"""Initialize the local database schema.

Usage after installing the project::

    python scripts/init_db.py

For production deployments prefer ``alembic upgrade head``.  This script is a
small first-run helper and intentionally never drops existing tables.
"""

from __future__ import annotations

from app.core.config import get_settings
from app.core.db import create_engine_from_settings, init_db


def main() -> None:
    settings = get_settings().ensure_runtime()
    engine = create_engine_from_settings(settings)
    init_db(engine)
    print(f"database initialized: {settings.database_url}")


if __name__ == "__main__":
    main()

