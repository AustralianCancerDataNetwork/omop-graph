def test_path_text_snapshot(kg, example_path):
    from omop_graph.render import render_path

    out = render_path(kg, example_path, format="text")

    assert out == """\
Aspirin --[Is a]--> Antiplatelet agent
Antiplatelet agent --[Is a]--> Drug
"""
