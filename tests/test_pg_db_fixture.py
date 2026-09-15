"""Smoke test for the pg_db fixture (Phase 0 of the schema_translate_map fix).

omop-graph had no Postgres test fixture at all before this -- this proves
the newly-added one actually works, not just that it's wired up.
"""

import sqlalchemy as sa


def test_pg_db_yields_a_working_connection_and_session(pg_db):
    assert pg_db.connection.execute(sa.text("SELECT 1")).scalar() == 1
    assert pg_db.session.connection() is pg_db.connection


def test_pg_db_rolls_back_between_tests(pg_db):
    """A second, independent test using the same fixture must not see
    anything from a prior test -- proving isolation, not just connectivity."""
    exists = pg_db.connection.execute(
        sa.text("SELECT to_regclass('pg_db_fixture_smoke_test')")
    ).scalar()
    assert exists is None
    pg_db.connection.execute(sa.text("CREATE TABLE pg_db_fixture_smoke_test (id INT)"))
    # Never committed -- rolled back automatically when this test ends.
