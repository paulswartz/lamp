"""add calendar_dates table

Revision ID: c514f5b5e389
Revises: be4c10c548f0
Create Date: 2023-04-05 16:04:39.041095

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = "c514f5b5e389"
down_revision = "be4c10c548f0"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # ### commands auto generated by Alembic - please adjust! ###
    op.create_table(
        "static_calendar_dates",
        sa.Column("pk_id", sa.Integer(), nullable=False),
        sa.Column("service_id", sa.String(length=128), nullable=False),
        sa.Column("date", sa.Integer(), nullable=False),
        sa.Column("exception_type", sa.SmallInteger(), nullable=False),
        sa.Column("holiday_name", sa.String(length=128), nullable=True),
        sa.Column("timestamp", sa.Integer(), nullable=False),
        sa.PrimaryKeyConstraint("pk_id"),
    )
    op.create_index(
        op.f("ix_static_calendar_dates_date"),
        "static_calendar_dates",
        ["date"],
        unique=False,
    )
    op.create_index(
        op.f("ix_static_calendar_dates_service_id"),
        "static_calendar_dates",
        ["service_id"],
        unique=False,
    )
    op.create_index(
        op.f("ix_static_calendar_dates_timestamp"),
        "static_calendar_dates",
        ["timestamp"],
        unique=False,
    )
    # ### end Alembic commands ###


def downgrade() -> None:
    # ### commands auto generated by Alembic - please adjust! ###
    op.drop_index(
        op.f("ix_static_calendar_dates_timestamp"),
        table_name="static_calendar_dates",
    )
    op.drop_index(
        op.f("ix_static_calendar_dates_service_id"),
        table_name="static_calendar_dates",
    )
    op.drop_index(
        op.f("ix_static_calendar_dates_date"),
        table_name="static_calendar_dates",
    )
    op.drop_table("static_calendar_dates")
    # ### end Alembic commands ###
