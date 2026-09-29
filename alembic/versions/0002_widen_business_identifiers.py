"""Widen prefixed business identifiers for PostgreSQL deployments.

The first development migration used UUID-sized VARCHAR columns.  The API uses
readable prefixes (``task_``, ``resume_`` and ``snapshot_``), so existing
PostgreSQL installations need those columns widened.  SQLite does not enforce
VARCHAR length and is intentionally left untouched.
"""

from alembic import op
import sqlalchemy as sa


revision = "0002_widen_business_identifiers"
down_revision = "0001_initial"
branch_labels = None
depends_on = None


def upgrade() -> None:
    bind = op.get_bind()
    if bind.dialect.name == "sqlite":
        return
    inspector = sa.inspect(bind)
    for table in inspector.get_table_names():
        columns = inspector.get_columns(table)
        for column in columns:
            name = str(column.get("name", ""))
            type_ = column.get("type")
            length = getattr(type_, "length", None)
            # Only widen UUID-sized identifier columns.  Hashes and ordinary
            # short strings retain their original constraints.
            if length == 36 and (name == "id" or name.endswith("_id")):
                op.alter_column(
                    table,
                    name,
                    existing_type=sa.String(length=36),
                    type_=sa.String(length=128),
                    existing_nullable=column.get("nullable", True),
                )


def downgrade() -> None:
    bind = op.get_bind()
    if bind.dialect.name == "sqlite":
        return
    inspector = sa.inspect(bind)
    for table in inspector.get_table_names():
        columns = inspector.get_columns(table)
        for column in columns:
            name = str(column.get("name", ""))
            type_ = column.get("type")
            length = getattr(type_, "length", None)
            if length == 128 and (name == "id" or name.endswith("_id")):
                op.alter_column(
                    table,
                    name,
                    existing_type=sa.String(length=128),
                    type_=sa.String(length=36),
                    existing_nullable=column.get("nullable", True),
                )
