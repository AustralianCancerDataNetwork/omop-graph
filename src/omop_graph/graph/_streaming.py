"""Server-side streaming of large query results."""

from __future__ import annotations

from sqlalchemy import Select


def stream(stmt: Select, batch_size: int = 10_000) -> Select:
    """
    Return *stmt* set to fetch its rows from a server-side cursor in batches.

    Only dialects with server-side cursors (psycopg, psycopg2) act on it, the others
    ignore it. It costs a named cursor per result, so use it for results that can be
    large or that a consumer may stop reading early, not for small lookups. The batch
    size replaces any ``yield_per`` configured on the engine or session.

    Parameters
    ----------
    stmt : Select
        The statement to stream.
    batch_size : int
        Rows fetched per round trip.

    Returns
    -------
    Select
        The statement with ``stream_results`` and ``yield_per`` set.
    """
    return stmt.execution_options(stream_results=True, yield_per=batch_size)
