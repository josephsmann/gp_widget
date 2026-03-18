# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "marimo",
#     "numpy",
#     "polars",
#     "altair",
#     "anywidget",
#     "traitlets",
#     "httpx==0.28.1",
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
    # Cherry Blossoms & Gaussian Processes


    This is meant to illustrate a GP widget ( https://github.com/josephsmann/gp_widget )I vibed after seeing Vincent's parallel coordinates widget (see https://youtu.be/MPqd3Y4wBig ).
    Also, thank you to Richard McElreath for this example and for his fantastic course `Rethinking Statistics`. Each version is better than the last.

    Kyoto has recorded the first day of cherry blossom bloom almost every year
    since **812 AD** — one of the longest environmental time series in existence.
    McElreath uses this dataset in *Statistical Rethinking* (ch. 14) to motivate
    Gaussian processes as priors over functions.

    **The idea:** we don't know what smooth curve connects the observations.
    A GP expresses our uncertainty by maintaining a whole *distribution over
    functions*. Each line in the plot below is one plausible history of Kyoto's
    spring climate.

    ### How to read this
    - Each polyline is a GP sample — a complete hypothetical 1200-year climate record
    - **Earlier bloom day = warmer spring** (lower `doy` = earlier)
    - The orange dots are the actual historical observations
    - **Place or move a constraint** (blue band) near an observation to condition
      the distribution on that data point — watch implausible histories fade out

    ### Interactions
    - **Drag a rectangle** in the scatter plot below, then tick *Link scatter → GP constraints*
      to automatically place a constraint at the centroid of the selected observations
    - **Click** on the GP plot to manually add a constraint
    - **Drag body** of a constraint left/right to change the year, up/down to shift the range
    - **Drag top/bottom handle** to tighten or loosen the tolerance
    - **Double-click** a constraint to remove it
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
    n_samples_slider = mo.ui.slider(100, 600, value=300, step=50, label="GP samples")
    lengthscale_slider = mo.ui.slider(10, 400, value=100, step=10, label="Lengthscale (years)")
    mo.hstack([n_samples_slider, lengthscale_slider])
    return lengthscale_slider, n_samples_slider


@app.cell
def _(mo, np, pl):
    # Download cherry blossom data
    _url = "https://raw.githubusercontent.com/rmcelreath/rethinking/master/data/cherry_blossoms.csv"
    try:
        import httpx
        _resp = httpx.get(_url, timeout=10)
        _raw = pl.read_csv(
            _resp.content,
            separator=";",
            null_values=["NA"],
        )
        cherry_df = _raw.drop_nulls(subset=["doy"])
        mo.stop(False)
    except Exception as _e:
        # Fallback: synthesize a plausible dataset if offline
        _rng = np.random.default_rng(0)
        _years = np.array(sorted(_rng.choice(range(812, 2016), size=827, replace=False)))
        _trend = -0.008 * (_years - 1200)          # slight long-run cooling then warming
        _doy = 105 + _trend + _rng.normal(0, 6, len(_years))
        cherry_df = pl.DataFrame({"year": _years.tolist(), "doy": _doy.tolist()})
    cherry_df
    return (cherry_df,)


@app.cell
def _(cherry_df, lengthscale_slider, n_samples_slider, np):
    # Build GP prior over the full year range
    _year_min, _year_max = 812, 2015
    _n_x = 120
    x_years = np.linspace(_year_min, _year_max, _n_x)

    # RBF kernel — lengthscale in years, marginal std ~7 days (≈ typical doy spread)
    _eta = 7.0
    _ls  = float(lengthscale_slider.value)

    def _rbf(x1, x2):
        d = x1[:, None] - x2[None, :]
        return _eta**2 * np.exp(-0.5 * d**2 / _ls**2)

    _K = _rbf(x_years, x_years) + 1e-6 * np.eye(_n_x)
    _L = np.linalg.cholesky(_K)

    # Mean function: sample around the observed grand mean doy
    _mu = float(cherry_df["doy"].mean())

    _rng = np.random.default_rng(42)
    _Z   = _rng.standard_normal((_n_x, n_samples_slider.value))
    gp_samples = (_mu + (_L @ _Z)).T   # shape (n_samples, n_x)
    return gp_samples, x_years


@app.cell
def _(ContinuousParallelCoords, mo):
    # Create widget once — no dependency on gp_samples/x_years so brush_axes
    # are preserved when sliders change. Data is injected imperatively below.
    widget = mo.ui.anywidget(ContinuousParallelCoords(
        height=380,
        width=740,
        x_label="Year",
        y_label="Day of first bloom (doy)",
    ))
    widget
    return (widget,)


@app.cell
def _(widget):
    # Expose the underlying AnyWidget object. This is a plain Python object,
    # not a UIElement, so cells that depend on it won't re-run on brush changes.
    underlying = widget.widget
    return (underlying,)


@app.cell
def _(gp_samples, underlying, x_years):
    # Runs when sliders change; updates data in place, leaving brush_axes intact.
    underlying.x_values = x_years.tolist()
    underlying.y_samples = gp_samples.tolist()
    return


@app.cell
def _(mo):
    link_toggle = mo.ui.checkbox(label="Link scatter selection → GP constraints")
    link_toggle
    return (link_toggle,)


@app.cell
def _(alt, cherry_df, mo):
    alt.data_transformers.disable_max_rows()
    # Observed data with interval brush — drag a rectangle to select an era
    _brush = alt.selection_interval(encodings=["x", "y"])
    _base  = alt.Chart(cherry_df.sample(n=min(400, len(cherry_df)), seed=1))
    _obs   = (
        _base.mark_circle(size=35, opacity=0.7)
        .encode(
            x=alt.X("year:Q", scale=alt.Scale(domain=[812, 2015]), title="Year"),
            y=alt.Y("doy:Q", title="Day of first bloom", scale=alt.Scale(zero=False)),
            color=alt.condition(_brush, alt.value("#e07b39"), alt.value("#cccccc")),
            tooltip=["year:Q", "doy:Q"],
        )
        .add_params(_brush)
        .properties(
            width=740, height=150,
            title="Observed data — drag to select an era, then toggle linking above",
        )
    )
    obs_chart = mo.ui.altair_chart(_obs)
    obs_chart
    return (obs_chart,)


@app.cell
def _(link_toggle, mo, np, obs_chart, pl, widget):
    mo.stop(not link_toggle.value)
    _sel = pl.DataFrame(obs_chart.value)
    if len(_sel) == 0:
        widget.brush_axes = {}
    else:
        _years = _sel["year"].to_numpy()
        _doys  = _sel["doy"].to_numpy()
        _x     = float(np.mean(_years))
        _mu    = float(np.mean(_doys))
        _sig   = float(max(np.std(_doys), 3.0))
        widget.brush_axes = {f"{_x:.6f}": [_mu - _sig, _mu + _sig]}
    return


@app.cell
def _(link_toggle, mo, obs_chart):
    mo.stop(not link_toggle.value)
    mo.stop(
        len(obs_chart.value) != 0,
        mo.md("*Selection active — constraint linked to GP plot.*"),
    )
    mo.callout(
        mo.md(
            "**No selection detected.** Drag a rectangle on the scatter plot above to select an era. "
            "If you drew a selection but nothing appeared here, the chart may have hit Altair's row limit — "
            "try a narrower brush or place a constraint directly on the GP plot."
        ),
        kind="warn",
    )
    return


@app.cell
def _(mo, widget):
    _n_filt  = len(widget.filtered_indices)
    _n_total = len(widget.y_samples)
    _pct = 100 * _n_filt / _n_total if _n_total else 0
    mo.md(f"**{_n_filt} / {_n_total}** samples consistent with current constraints ({_pct:.1f} %)")
    return


@app.cell
def _(mo):
    mo.md("""
    ## Posterior: mean ± 1 std of filtered samples

    After placing constraints near the observations the mean line should
    thread through them and uncertainty should widen in data-sparse centuries.
    """)
    return


@app.cell
def _(alt, mo, np, pl, widget, x_years):
    mo.stop(
        len(widget.filtered_indices) == 0,
        mo.md("*No samples pass the current constraints — try widening a brush.*"),
    )

    _filt = np.array([widget.y_samples[i] for i in widget.filtered_indices])
    _mean = _filt.mean(axis=0)
    _std  = _filt.std(axis=0)

    _df = pl.DataFrame({
        "year": x_years.tolist(),
        "mean": _mean.tolist(),
        "lo":   (_mean - _std).tolist(),
        "hi":   (_mean + _std).tolist(),
    })

    _band = (
        alt.Chart(_df).mark_area(opacity=0.20, color="#4477aa")
        .encode(x="year:Q", y="lo:Q", y2="hi:Q")
    )
    _line = (
        alt.Chart(_df).mark_line(color="#4477aa", strokeWidth=2)
        .encode(
            x=alt.X("year:Q", title="Year"),
            y=alt.Y("mean:Q", title="Day of first bloom", scale=alt.Scale(zero=False)),
            tooltip=["year:Q", alt.Tooltip("mean:Q", format=".1f")],
        )
    )
    (_band + _line).properties(width=740, height=200)
    return


@app.cell
def _(mo):
    mo.md("""
    ---
    ## What to try

    | Action | What you see |
    |--------|-------------|
    | Place a tight constraint at year ~1600, doy ~100 | Most lines are ruled out; the posterior mean is pulled toward 100 |
    | Place a second constraint at year ~1900, doy ~105 | Uncertainty shrinks near both points; it stays high in between |
    | Widen the lengthscale slider | Constraints "spread" influence further in time |
    | Narrow the lengthscale slider | Each constraint has only a local effect |

    The lengthscale is the GP hyperparameter McElreath calls `ρ` — it controls
    how many years apart two points need to be before their bloom days become
    effectively independent.
    """)
    return


if __name__ == "__main__":
    app.run()
