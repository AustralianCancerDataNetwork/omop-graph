from sqlalchemy import select

from omop_graph.graph._streaming import stream


def test_stream_sets_a_server_side_cursor_with_the_batch_size() -> None:
    options = stream(select(1), batch_size=500).get_execution_options()

    assert options == {"stream_results": True, "yield_per": 500}
