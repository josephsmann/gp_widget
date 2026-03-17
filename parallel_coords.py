# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "marimo",
#     "numpy",
#     "polars",
#     "scikit-learn",
#     "wigglystuff==0.2.37",
# ]
# ///

import marimo

__generated_with = "0.21.0"
app = marimo.App()


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import polars as pl
    from sklearn.datasets import load_iris
    from wigglystuff import ParallelCoordinates

    return ParallelCoordinates, load_iris, mo, np, pl


@app.cell
def _(mo):
    mo.md("""
    ## ParallelCoordinates

    Interactive parallel coordinates plot powered by HiPlot.
    Brush on axes to filter, drag axis labels to reorder, and right-click an axis to color by it.
    """)
    return


@app.cell
def _(ParallelCoordinates, load_iris, mo, pl):
    iris = load_iris()
    iris_df = pl.DataFrame(
        {name: iris.data[:, i] for i, name in enumerate(iris.feature_names)}
    ).with_columns(pl.Series("target", iris.target))

    widget = mo.ui.anywidget(ParallelCoordinates(
        iris_df,
        height=300,
        width=700,
        color_by="target",
        color_map={0: "teal", 1: "orange", 2: "crimson"},
    ))
    widget
    return (widget,)


@app.cell
def _(mo, widget):
    _n_filtered = len(widget.filtered_indices)
    _n_total = len(widget.data)
    mo.md(f"**Filtered:** {_n_filtered} / {_n_total} rows")
    return


@app.cell
def _(widget):
    widget.filtered_as_polars
    return


@app.cell
def _(mo):
    mo.md("""
    ### Scale Demo

    Example with **15,000 rows** and multiple dimensions to show large-data handling.
    """)
    return


@app.cell
def _(ParallelCoordinates, mo, np, pl):
    _rng = np.random.default_rng(42)
    _n_rows = 15_000
    _segments = _rng.integers(0, 4, size=_n_rows)

    big_df = pl.DataFrame(
        {
            "x0": _rng.normal(_segments * 0.7, 1.0),
            "x1": _rng.normal(_segments * 0.3, 1.1),
            "x2": _rng.normal(_segments * 0.5, 0.9),
            "x3": _rng.normal(_segments * 0.2, 1.2),
            "x4": _rng.normal(_segments * 0.4, 1.0),
            "x5": _rng.normal(_segments * 0.6, 1.0),
            "segment": [str(s) for s in _segments],
        }
    )

    widget_lg = mo.ui.anywidget(ParallelCoordinates(big_df, height=360, color_by="segment"))
    widget_lg
    return (widget_lg,)


@app.cell
def _(mo, widget_lg):
    _n_filtered = len(widget_lg.filtered_indices)
    _n_total = len(widget_lg.data)
    mo.md(f"**Scale demo filtered:** {_n_filtered} / {_n_total} rows")
    return


if __name__ == "__main__":
    app.run()
