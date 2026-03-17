# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "marimo",
#     "numpy",
#     "polars",
#     "altair",
#     "anywidget",
#     "traitlets",
#     "gp-coords-widget @ git+https://github.com/josephsmann/gp_widget.git",
# ]
# ///

import marimo

__generated_with = "0.21.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import polars as pl
    import altair as alt
    from gp_coords_widget import ContinuousParallelCoords

    return ContinuousParallelCoords, alt, mo, np, pl


@app.cell
def _(mo):
    mo.md("""
    # Continuous Parallel Coordinates for Function Distributions

    This widget extends the parallel-coordinates idea to **continuous input domains**.
    Each polyline is a function sample `f(x)`.  Placing a constraint at position `x`
    filters to samples where `f(x) ∈ [lo, hi]` — like interactive GP posterior conditioning.

    **Interactions**
    - **Click** anywhere in the plot → add a constraint centred on that `(x, y)`
    - **Drag body** of a constraint → move it (left/right = x position, up/down = y shift)
    - **Drag top / bottom handle** → resize the `y`-range
    - **Double-click** a constraint → remove it

    Switch kernels to see how the prior's smoothness changes — constraints on rough
    (Matérn 1/2) samples propagate very locally, while RBF samples are infinitely
    smooth and constraints ripple further. McElreath uses the RBF kernel in
    *Statistical Rethinking* ch. 14.
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    ## Controls
    """)
    return


@app.cell
def _(mo):
    n_samples_slider = mo.ui.slider(50, 500, value=200, step=50, label="Samples")
    lengthscale_slider = mo.ui.slider(0.05, 1.0, value=0.3, step=0.05, label="Lengthscale")
    kernel_dropdown = mo.ui.dropdown(
        options={
            "RBF (squared exponential)": "rbf",
            "Matérn 1/2 (Laplace / rough)": "matern12",
            "Matérn 3/2 (once differentiable)": "matern32",
            "Matérn 5/2 (twice differentiable)": "matern52",  # common GP library default
            "Periodic": "periodic",
            "Linear": "linear",
        },
        value="rbf",
        label="Kernel",
    )
    mo.hstack([kernel_dropdown, lengthscale_slider, n_samples_slider])
    return kernel_dropdown, lengthscale_slider, n_samples_slider


@app.cell
def _(kernel_dropdown, lengthscale_slider, n_samples_slider, np):
    _n_x = 50
    x_values = np.linspace(0, 1, _n_x)
    _ls = lengthscale_slider.value
    _d = np.abs(x_values[:, None] - x_values[None, :])  # pairwise distances

    _kernels = {
        "rbf":       lambda d: np.exp(-0.5 * d**2 / _ls**2),
        "matern12":  lambda d: np.exp(-d / _ls),
        "matern32":  lambda d: (1 + np.sqrt(3) * d / _ls) * np.exp(-np.sqrt(3) * d / _ls),
        "matern52":  lambda d: (1 + np.sqrt(5) * d / _ls + 5 * d**2 / (3 * _ls**2)) * np.exp(-np.sqrt(5) * d / _ls),
        "periodic":  lambda d: np.exp(-2 * np.sin(np.pi * d) ** 2 / _ls**2),
        "linear":    lambda d: (x_values[:, None] - 0.5) @ (x_values[None, :] - 0.5) / _ls**2,
    }

    _K = _kernels[kernel_dropdown.value](_d) + 1e-6 * np.eye(_n_x)
    _L = np.linalg.cholesky(_K)

    _rng = np.random.default_rng(42)
    _Z = _rng.standard_normal((_n_x, n_samples_slider.value))
    y_samples = (_L @ _Z).T  # (n_samples, n_x)
    return x_values, y_samples


@app.cell
def _(ContinuousParallelCoords, mo):
    # Created once — brush constraints survive kernel/slider changes.
    widget = mo.ui.anywidget(ContinuousParallelCoords(
        [0.0], [[0.0]],
        height=420,
        width=1000,
        x_label="x",
        y_label="f(x)",
    ))
    widget
    return (widget,)


@app.cell
def _(widget):
    underlying = widget.widget
    return (underlying,)


@app.cell
def _(underlying, x_values, y_samples):
    underlying.x_values = x_values.tolist()
    underlying.y_samples = y_samples.tolist()
    return


@app.cell
def _(mo, widget):
    _n_filt  = len(widget.filtered_indices)
    _n_total = len(widget.y_samples)
    _pct = 100 * _n_filt / _n_total if _n_total else 0
    mo.md(f"**{_n_filt} / {_n_total}** samples pass current constraints ({_pct:.1f} %)")
    return


@app.cell
def _(mo):
    mo.md("""
    ## Filtered sample statistics
    """)
    return


@app.cell
def _(alt, mo, np, pl, widget, x_values):
    mo.stop(len(widget.filtered_indices) == 0, mo.md("*No samples pass the current constraints.*"))

    _filt = np.array([widget.y_samples[i] for i in widget.filtered_indices])
    _mean = _filt.mean(axis=0)
    _std  = _filt.std(axis=0)

    _stats_df = pl.DataFrame({
        "x":    x_values.tolist(),
        "mean": _mean.tolist(),
        "lo":   (_mean - _std).tolist(),
        "hi":   (_mean + _std).tolist(),
    })

    _band = (
        alt.Chart(_stats_df)
        .mark_area(opacity=0.25, color="#4477aa")
        .encode(x="x:Q", y="lo:Q", y2="hi:Q")
    )
    _line = (
        alt.Chart(_stats_df)
        .mark_line(color="#4477aa", strokeWidth=2)
        .encode(x=alt.X("x:Q", title="x"), y=alt.Y("mean:Q", title="f(x)"),
                tooltip=["x:Q", "mean:Q"])
    )
    (_band + _line).properties(
        title="Mean ± 1 std of filtered samples",
        width=680, height=200,
    )
    return


@app.cell
def _(mo):
    mo.md("""
    ## Raw filtered data
    """)
    return


@app.cell
def _(mo, pl, widget, x_values):
    mo.stop(len(widget.filtered_indices) == 0, mo.md("*No samples selected.*"))

    _rows = [widget.y_samples[i] for i in widget.filtered_indices[:100]]
    _cols = {f"{x:.3f}": [row[j] for row in _rows]
             for j, x in enumerate(x_values[::8])}  # every 8th column
    _cols["sample_idx"] = widget.filtered_indices[:100]

    # show mean and std per column for quick inspection
    _df = pl.DataFrame(_cols)
    _summary = _df.select(
        pl.all().exclude("sample_idx")
    ).describe()
    _summary
    return


if __name__ == "__main__":
    app.run()
