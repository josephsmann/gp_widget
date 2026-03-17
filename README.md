# gp-coords-widget

Continuous parallel coordinates widget for visualising distributions over functions — GP priors, posterior samples, ensemble forecasts.

## Install

```bash
pip install "gp-coords-widget @ git+https://github.com/josephsmann/gp_widget.git"
```

## Usage

```python
import marimo as mo
from gp_coords_widget import ContinuousParallelCoords

widget = mo.ui.anywidget(ContinuousParallelCoords(
    x_values,   # list of floats — the shared input grid
    y_samples,  # list of lists — shape (n_samples, len(x_values))
))
widget
```

**Interactions**
- **Click** in the plot → add a constraint at that (x, y)
- **Drag body** → reposition the constraint (horizontal = x, vertical = y-shift)
- **Drag top/bottom handle** → resize the y-range
- **Double-click** → remove the constraint

`widget.filtered_indices` — indices of samples consistent with all constraints.

## Demos

| Notebook | Description |
|----------|-------------|
| `gp_parallel_coords_demo.py` | GP prior samples with RBF kernel |
| `cherry_blossom_gp.py` | McElreath's cherry blossom data (Statistical Rethinking ch. 14) |
