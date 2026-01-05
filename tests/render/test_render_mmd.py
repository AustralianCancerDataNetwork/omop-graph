def test_path_mermaid_snapshot(kg, example_path):
    from kg_core.render import render_path

    out = render_path(kg, example_path, format="mmd")

    assert out.startswith("graph LR")
    assert "-->|Is a|" in out
