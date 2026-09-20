"""add job stage

Revision ID: 146832c0b99f
Revises: 8738bcd0e60b
Create Date: 2026-09-20 17:30:24.550814

NOTE: autogenerate originally emitted DROP statements for LangGraph's
checkpoint tables (checkpoints, checkpoint_blobs, checkpoint_writes,
checkpoint_migrations) in this migration. Those tables are created and owned by
PostgresSaver.setup() at application start, not by Alembic -- autogenerate saw
tables absent from our models' metadata and concluded they should be deleted.

Locally that succeeded (silently wiping chat history, which setup() then
recreated). On a fresh database the tables don't exist yet, so DROP INDEX
raised and the container crash-looped. Those statements are removed here, and
alembic/env.py now filters non-Alembic tables out of autogenerate so this
cannot recur.
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision: str = '146832c0b99f'
down_revision: Union[str, Sequence[str], None] = '8738bcd0e60b'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.add_column('jobs', sa.Column('stage', sa.String(), nullable=True))


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_column('jobs', 'stage')
