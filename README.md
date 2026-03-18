# gp-coords-widget

Continuous parallel coordinates widget for visualising distributions over functions — GP priors, posterior samples, ensemble forecasts.

## Install

```bash
pip install gp-coords-widget
```

Or directly from source:

```bash
pip install "gp-coords-widget @ git+https://github.com/josephsmann/gp_widget.git"
```

## Usage

Create the widget once (with no data), then inject data imperatively. This decouples the widget's lifetime from your data so brush constraints survive slider or parameter changes.

```python
import marimo as mo
from gp_coords_widget import ContinuousParallelCoords

# Cell 1 — create widget once; brush constraints survive data updates
widget = mo.ui.anywidget(ContinuousParallelCoords(
    height=420, width=1000,
    x_label="x", y_label="f(x)",
))
widget

# Cell 2 — expose underlying object (plain Python, not a UIElement)
underlying = widget.widget

# Cell 3 — update data whenever inputs change; brush constraints are preserved
underlying.x_values = x_values   # list[float] — shared input grid
underlying.y_samples = y_samples  # list[list[float]] — shape (n_samples, n_x)
```

**Interactions**
- **Click** in the plot → add a constraint at that (x, y)
- **Drag body** → reposition the constraint (horizontal = x, vertical = y-shift)
- **Drag top/bottom handle** → resize the y-range
- **Double-click** → remove the constraint

**Reading results**

```python
widget.filtered_indices   # list[int] — indices of samples passing all constraints
widget.filtered_samples   # list[list[float]] — the passing rows
widget.filtered_as_numpy  # numpy array, shape (k, n_x)
widget.filtered_as_polars # polars DataFrame
```

## Demos

| Notebook | Description |
|----------|-------------|
| `gp_parallel_coords_demo.py` | GP prior samples with six kernel choices (RBF, Matérn 1/2 / 3/2 / 5/2, Periodic, Linear) |
| `cherry_blossom_gp.py` | McElreath's cherry blossom data (Statistical Rethinking ch. 14) — interactive GP posterior conditioning on 1200 years of Kyoto bloom records |
