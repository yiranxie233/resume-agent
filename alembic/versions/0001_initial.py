"""Create the initial resume-agent schema.

The canonical table definitions live in ``app.core.models``.  Calling
``create_all`` here keeps this initial migration concise while still making it
safe to run repeatedly: SQLAlchemy only creates missing tables.  Subsequent
migrations should use normal Alembic operations so production data is never
silently rebuilt.
"""

from alembic import op

from app.core.models import Base

revision = "0001_initial"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    bind = op.get_bind()
    Base.metadata.create_all(bind=bind)


def downgrade() -> None:
    bind = op.get_bind()
    Base.metadata.drop_all(bind=bind)
