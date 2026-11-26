# Rendering

`omop_graph.render` turns paths, subgraphs and traces into text, HTML or Mermaid.

```python
from omop_graph.render import render_path, render_subgraph, render_trace, render_explained_path

print(render_path(kg, path))
render_subgraph(kg, subgraph, format="mmd")
```

| Function | Renders | Formats |
|---|---|---|
| `render_path` | A `GraphPath` | `text`, `html`, `mmd` |
| `render_subgraph` | A `Subgraph` | `text`, `html`, `mmd` |
| `render_trace` | A `GraphTrace` | `text`, `html` |
| `render_explained_path` | A `PathExplanation` | `text`, `html` |

* The default format is `auto`, which gives HTML in a Jupyter notebook and text elsewhere.
* The Mermaid output is the text of a flowchart that can be pasted into any Mermaid renderer.
* `bind_default_renderers(kg)` makes `GraphPath`, `Subgraph` and `GraphTrace` display as HTML in a notebook cell. The oaklib adapter calls it when it builds its graph.
* A format a function does not support gives its text output.
