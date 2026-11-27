def test_cli_imports_with_orm_loader_three():
    from omop_graph.cli import app

    assert app is not None
