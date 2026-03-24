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
    import anywidget
    import traitlets
    from gp_coords_widget import ContinuousParallelCoords

    return ContinuousParallelCoords, alt, anywidget, mo, np, pl, traitlets


@app.cell
def _(mo):
    mo.md(r"""
    # Gaussian Process Kernel Explorer

    version 0.11

    A Gaussian Process is fully specified by its **kernel** (covariance function) `k(x₁, x₂)`.
    The kernel encodes assumptions about smoothness, periodicity, and correlation length.

    For stationary kernels, covariance depends only on the lag `r = |x₁ − x₂|`:

    $$k(x_1, x_2) = k(r), \quad r = |x_1 - x_2|$$

    **Draw your own kernel** below using the spline editor, then see what GP sample functions
    it produces in the parallel coordinates view.
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    ## Draw a Kernel Function

    Sketch `k(r)` — covariance as a function of lag `r = |x₁ − x₂|`.

    - **Click** to add control points
    - **Drag** to move them
    - **Double-click** a point to remove it

    The curve is evaluated as a Catmull-Rom spline. Values are clamped to `[0, 1]`.
    A valid kernel should start near **1** at `r = 0` and decay toward **0**.
    """)
    return


@app.cell
def _(anywidget, mo, traitlets):
    class KernelSplineWidget(anywidget.AnyWidget):
        _esm = """
    function render({ model, el }) {
      const W = 560, H = 280;
      const M = { top: 15, right: 20, bottom: 44, left: 58 };
      const iW = W - M.left - M.right;
      const iH = H - M.top - M.bottom;
      const X_MAX = 1.0, Y_MAX = 2.0;

      const sx = x => (x / X_MAX) * iW;
      const sy = y => iH - (y / Y_MAX) * iH;
      const dx = px => (px / iW) * X_MAX;
      const dy = py => (1 - py / iH) * Y_MAX;

      const ns = "http://www.w3.org/2000/svg";
      const svgEl = (tag, attrs = {}) => {
        const e = document.createElementNS(ns, tag);
        for (const [k, v] of Object.entries(attrs)) e.setAttribute(k, v);
        return e;
      };

      const svg = svgEl("svg", { width: W, height: H, class: "ksw-root" });
      svg.style.cssText = "user-select:none;touch-action:none;display:block;cursor:crosshair;";
      const g = svgEl("g", { transform: `translate(${M.left},${M.top})` });
      svg.appendChild(g);

      g.appendChild(svgEl("rect", { x: 0, y: 0, width: iW, height: iH, class: "ksw-bg" }));

      const gridG = svgEl("g", { class: "ksw-grid" });
      for (let y = 0; y <= 2.001; y += 0.5) {
        gridG.appendChild(svgEl("line", { x1: 0, y1: sy(y), x2: iW, y2: sy(y) }));
      }
      for (let x = 0; x <= 1.001; x += 0.2) {
        gridG.appendChild(svgEl("line", { x1: sx(x), y1: 0, x2: sx(x), y2: iH }));
      }
      g.appendChild(gridG);

      const fillPath = svgEl("path", { class: "ksw-fill", fill: "none" });
      g.appendChild(fillPath);
      const splinePath = svgEl("path", { class: "ksw-line", fill: "none" });
      g.appendChild(splinePath);

      const axisG = svgEl("g", { class: "ksw-axis" });
      for (let y = 0; y <= 2.001; y += 0.5) {
        const t = svgEl("text", { x: -8, y: sy(y) + 4, "text-anchor": "end", class: "ksw-label" });
        t.textContent = y.toFixed(1);
        axisG.appendChild(t);
      }
      for (let x = 0; x <= 1.001; x += 0.2) {
        const t = svgEl("text", { x: sx(x), y: iH + 18, "text-anchor": "middle", class: "ksw-label" });
        t.textContent = x.toFixed(1);
        axisG.appendChild(t);
      }
      const xTitle = svgEl("text", { x: iW / 2, y: iH + 40, "text-anchor": "middle", class: "ksw-label ksw-title" });
      xTitle.textContent = "|x₁ − x₂|  (lag r)";
      axisG.appendChild(xTitle);
      const yTitle = svgEl("text", {
        x: -(iH / 2), y: -46, "text-anchor": "middle",
        class: "ksw-label ksw-title",
        transform: "rotate(-90)"
      });
      yTitle.textContent = "covariance  k(r)";
      axisG.appendChild(yTitle);
      g.appendChild(axisG);

      const ptsG = svgEl("g");
      g.appendChild(ptsG);
      const tip = svgEl("text", { class: "ksw-tip", "text-anchor": "middle", y: -6 });
      g.appendChild(tip);

      let pts = model.get("points") || [];
      if (pts.length < 2) pts = [[0.0, 1.0], [0.5, 0.35], [1.0, 0.02]];
      let dragging = null;

      function padded(sorted) { return [sorted[0], ...sorted, sorted[sorted.length - 1]]; }

      function crInterp(p0, p1, p2, p3, t) {
        const t2 = t * t, t3 = t2 * t;
        return 0.5 * (2 * p1 + (-p0 + p2) * t + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t2 + (-p0 + 3 * p1 - 3 * p2 + p3) * t3);
      }

      function buildPath(sorted) {
        if (sorted.length < 2) return "";
        const P = padded(sorted);
        let d = `M ${sx(sorted[0][0]).toFixed(2)} ${sy(sorted[0][1]).toFixed(2)}`;
        for (let i = 0; i < sorted.length - 1; i++) {
          const p0 = P[i], p1 = P[i+1], p2 = P[i+2], p3 = P[i+3];
          for (let s = 1; s <= 20; s++) {
            const t = s / 20;
            const x = p1[0] + t * (p2[0] - p1[0]);
            const y = crInterp(p0[1], p1[1], p2[1], p3[1], t);
            d += ` L ${sx(x).toFixed(2)} ${sy(y).toFixed(2)}`;
          }
        }
        return d;
      }

      function interpolateAt(sorted, x) {
        if (x <= sorted[0][0]) return sorted[0][1];
        if (x >= sorted[sorted.length-1][0]) return sorted[sorted.length-1][1];
        let i = 0;
        while (i < sorted.length - 2 && sorted[i+1][0] < x) i++;
        const P = padded(sorted);
        const p0 = P[i], p1 = P[i+1], p2 = P[i+2], p3 = P[i+3];
        const t = (x - p1[0]) / (p2[0] - p1[0]);
        return crInterp(p0[1], p1[1], p2[1], p3[1], t);
      }

      function redraw() {
        const sorted = [...pts].sort((a, b) => a[0] - b[0]);
        const pathD = buildPath(sorted);
        splinePath.setAttribute("d", pathD);
        const last = sorted[sorted.length-1], first = sorted[0];
        fillPath.setAttribute("d", pathD + ` L ${sx(last[0]).toFixed(2)} ${sy(0).toFixed(2)} L ${sx(first[0]).toFixed(2)} ${sy(0).toFixed(2)} Z`);
        ptsG.innerHTML = "";
        for (let i = 0; i < pts.length; i++) {
          const c = svgEl("circle", {
            cx: sx(pts[i][0]), cy: sy(pts[i][1]),
            r: 6, class: "ksw-pt", "data-i": i
          });
          c.style.cursor = "grab";
          c.addEventListener("mouseenter", () => c.setAttribute("r", 8));
          c.addEventListener("mouseleave", () => { if (dragging !== i) c.setAttribute("r", 6); });
          ptsG.appendChild(c);
        }
      }

      function save() {
        const sorted = [...pts].sort((a, b) => a[0] - b[0]);
        const N = 100;
        const sampled = [];
        for (let i = 0; i < N; i++) {
          sampled.push(interpolateAt(sorted, i / (N - 1)));
        }
        model.set("points", pts.map(([x, y]) => [
          Math.round(x * 1000) / 1000,
          Math.round(y * 1000) / 1000
        ]));
        model.set("sampled_values", sampled);
        model.save_changes();
      }

      svg.addEventListener("mousedown", e => {
        const rect = svg.getBoundingClientRect();
        const rx = e.clientX - rect.left - M.left, ry = e.clientY - rect.top - M.top;
        for (let i = 0; i < pts.length; i++) {
          if (Math.hypot(sx(pts[i][0]) - rx, sy(pts[i][1]) - ry) < 10) {
            if (e.detail === 2) { if (pts.length > 2) { pts.splice(i, 1); redraw(); save(); } return; }
            dragging = i; svg.style.cursor = "grabbing"; return;
          }
        }
        pts.push([dx(rx), dy(ry)]);
        redraw(); save();
      });

      svg.addEventListener("mousemove", e => {
        if (dragging === null) return;
        const rect = svg.getBoundingClientRect();
        const rx = e.clientX - rect.left - M.left, ry = e.clientY - rect.top - M.top;
        pts[dragging] = [Math.max(0, Math.min(1, dx(rx))), dy(ry)];
        // clamp the leftmost point's x to exactly 0
        const minIdx = pts.reduce((mi, p, i) => p[0] < pts[mi][0] ? i : mi, 0);
        pts[minIdx][0] = 0;
        const v = pts[dragging];
        tip.setAttribute("x", sx(v[0]));
        tip.setAttribute("y", sy(Math.max(0, v[1])) - 10);
        tip.textContent = `r=${v[0].toFixed(2)}, k=${v[1].toFixed(3)}`;
        redraw();
      });

      const endDrag = () => {
        if (dragging !== null) { save(); dragging = null; svg.style.cursor = "crosshair"; tip.textContent = ""; }
      };
      svg.addEventListener("mouseup", endDrag);
      svg.addEventListener("mouseleave", endDrag);

      redraw(); save();
      el.appendChild(svg);
    }
    export default { render };
    """
        _css = """
    .ksw-bg { fill: #f5f6fa; }
    .ksw-grid line { stroke: #d8dce8; stroke-width: 0.5; }
    .ksw-line { stroke: #b03060; stroke-width: 2.5; stroke-linejoin: round; }
    .ksw-fill { fill: #b03060; opacity: 0.12; }
    .ksw-pt { fill: #b03060; stroke: #fff; stroke-width: 2.5; }
    .ksw-label { font: 11px/1 system-ui, sans-serif; fill: #555; }
    .ksw-title { font-size: 12px; font-weight: 600; }
    .ksw-tip { font: bold 11px system-ui, sans-serif; fill: #b03060; }
    @media (prefers-color-scheme: dark) {
      .ksw-bg { fill: #1a1d27; }
      .ksw-grid line { stroke: #2e3145; }
      .ksw-label { fill: #9099b2; }
      .ksw-pt { stroke: #1a1d27; }
      .ksw-tip { fill: #e06080; }
    }
    """
        points = traitlets.List([]).tag(sync=True)
        sampled_values = traitlets.List([]).tag(sync=True)

    kernel_spline_widget = mo.ui.anywidget(KernelSplineWidget())
    kernel_spline_widget
    return (kernel_spline_widget,)


@app.cell
def _(mo):
    kernel_source = mo.ui.radio(
        options={"Draw a custom kernel": "spline", "Use a standard kernel": "standard"},
        value="Draw a custom kernel",
        label="Kernel source",
    )
    kernel_source
    return (kernel_source,)


@app.cell
def _(mo):
    std_kernel_dd = mo.ui.dropdown(
        options={
            "RBF / Squared Exponential": "rbf",
            "Matérn 1/2 (rough)": "matern12",
            "Matérn 3/2": "matern32",
            "Matérn 5/2 (smooth)": "matern52",
            "Periodic": "periodic",
            "Linear": "linear",
        },
        value="RBF / Squared Exponential",
        label="Kernel",
    )
    amp_slider = mo.ui.slider(0.1, 3.0, value=1.0, step=0.05, label="Amplitude α")
    ls_slider = mo.ui.slider(0.05, 1.0, value=0.3, step=0.05, label="Lengthscale ℓ")
    period_slider = mo.ui.slider(0.1, 2.0, value=1.0, step=0.05, label="Period p (Periodic only)")
    bias_slider = mo.ui.slider(0.0, 2.0, value=0.0, step=0.05, label="Bias σ_b² (Linear only)")
    slope_slider = mo.ui.slider(0.1, 3.0, value=1.0, step=0.05, label="Slope σ_v² (Linear only)")
    shift_slider = mo.ui.slider(0.0, 1.0, value=0.0, step=0.05, label="Shift c (Linear only)")
    return (
        amp_slider,
        bias_slider,
        ls_slider,
        period_slider,
        shift_slider,
        slope_slider,
        std_kernel_dd,
    )


@app.cell
def _(
    amp_slider,
    bias_slider,
    kernel_source,
    ls_slider,
    mo,
    period_slider,
    shift_slider,
    slope_slider,
    std_kernel_dd,
):
    mo.stop(
        kernel_source.value != "standard",
        mo.md("_Draw a kernel shape above using the spline editor._"),
    )
    _k = std_kernel_dd.value
    _a = amp_slider.value
    _ls = ls_slider.value
    _common = mo.hstack([std_kernel_dd, amp_slider, ls_slider])
    if _k == "periodic":
        _extra = mo.hstack([period_slider])
    elif _k == "linear":
        _extra = mo.hstack([bias_slider, slope_slider, shift_slider])
    else:
        _extra = mo.md("")

    _formulas = {
        "rbf": rf"k(r) = {_a:.2f}^2 \exp\!\left(-\frac{{r^2}}{{2 \cdot {_ls:.2f}^2}}\right)",
        "matern12": rf"k(r) = {_a:.2f}^2 \exp\!\left(-\frac{{r}}{{{_ls:.2f}}}\right)",
        "matern32": rf"k(r) = {_a:.2f}^2 \left(1 + \frac{{\sqrt{{3}}\,r}}{{{_ls:.2f}}}\right)\exp\!\left(-\frac{{\sqrt{{3}}\,r}}{{{_ls:.2f}}}\right)",
        "matern52": rf"k(r) = {_a:.2f}^2 \left(1 + \frac{{\sqrt{{5}}\,r}}{{{_ls:.2f}}} + \frac{{5\,r^2}}{{3 \cdot {_ls:.2f}^2}}\right)\exp\!\left(-\frac{{\sqrt{{5}}\,r}}{{{_ls:.2f}}}\right)",
        "periodic": rf"k(r) = {_a:.2f}^2 \exp\!\left(-\frac{{2\sin^2(\pi r / {period_slider.value:.2f})}}{{{_ls:.2f}^2}}\right)",
        "linear": rf"k(x_1, x_2) = {bias_slider.value:.2f} + {slope_slider.value:.2f}\,(x_1 - {shift_slider.value:.2f})(x_2 - {shift_slider.value:.2f})",
    }
    _formula = mo.md(rf"$$\boxed{{{_formulas[_k]}}}$$")

    mo.vstack([_common, _extra, _formula])
    return


@app.cell
def _(mo):
    mo.md("""
    ## GP Samples

    Adjust the number of samples and noise level, then interact with the parallel coordinates
    to condition the GP on constraints.
    """)
    return


@app.cell
def _(mo):
    n_samples_slider = mo.ui.slider(50, 1000, value=300, step=50, label="Samples")
    n_points_slider = mo.ui.slider(20, 200, value=80, step=10, label="Points per line")
    noise_slider = mo.ui.slider(0.0, 0.1, value=0.01, step=0.001, label="Jitter ε (numerical stability)")
    mo.vstack([
        mo.hstack([n_samples_slider, n_points_slider, noise_slider]),
        mo.md("_Jitter adds ε·I to K before Cholesky to guarantee positive-definiteness. A drawn spline may not be exactly PSD — raise ε if sampling fails._"),
    ])
    return n_points_slider, n_samples_slider, noise_slider


@app.cell
def _(
    amp_slider,
    bias_slider,
    kernel_source,
    kernel_spline_widget,
    ls_slider,
    n_points_slider,
    n_samples_slider,
    noise_slider,
    np,
    period_slider,
    shift_slider,
    slope_slider,
    std_kernel_dd,
):
    _n_x = n_points_slider.value
    x_values = np.linspace(0, 1, _n_x)
    _d = np.abs(x_values[:, None] - x_values[None, :])

    if kernel_source.value == "standard":
        _k = std_kernel_dd.value
        _a = amp_slider.value
        _ls = ls_slider.value
        if _k == "rbf":
            _K = _a ** 2 * np.exp(-0.5 * _d ** 2 / _ls ** 2)
        elif _k == "matern12":
            _K = _a ** 2 * np.exp(-_d / _ls)
        elif _k == "matern32":
            _K = _a ** 2 * (1 + np.sqrt(3) * _d / _ls) * np.exp(-np.sqrt(3) * _d / _ls)
        elif _k == "matern52":
            _K = _a ** 2 * (1 + np.sqrt(5) * _d / _ls + 5 * _d ** 2 / (3 * _ls ** 2)) * np.exp(-np.sqrt(5) * _d / _ls)
        elif _k == "periodic":
            _p = period_slider.value
            _K = _a ** 2 * np.exp(-2 * np.sin(np.pi * _d / _p) ** 2 / _ls ** 2)
        else:  # linear
            _sb2 = bias_slider.value
            _sv2 = slope_slider.value
            _c = shift_slider.value
            _K = _sb2 + _sv2 * (x_values[:, None] - _c) * (x_values[None, :] - _c)
        _K = _K + noise_slider.value * np.eye(_n_x)
    else:
        _kvals = kernel_spline_widget.sampled_values
        if len(_kvals) == 0:
            _kvals = [1.0] * 100

        def _interp_kernel(r):
            _idx = r * 99
            _lo = int(_idx)
            _hi = min(_lo + 1, 99)
            _frac = _idx - _lo
            return float(_kvals[_lo]) * (1 - _frac) + float(_kvals[_hi]) * _frac

        _K = np.vectorize(_interp_kernel)(_d) + noise_slider.value * np.eye(_n_x)

    _eigvals, _eigvecs = np.linalg.eigh(_K)
    _n_negative = int((_eigvals < 0).sum())
    _eigvals_clipped = np.clip(_eigvals, 1e-8, None)
    _K = (_eigvecs * _eigvals_clipped) @ _eigvecs.T

    _L = np.linalg.cholesky(_K)
    _rng = np.random.default_rng(42)
    _Z = _rng.standard_normal((_n_x, n_samples_slider.value))
    y_samples = (_L @ _Z).T
    n_negative_eigvals = _n_negative
    K_matrix = _K
    return K_matrix, n_negative_eigvals, x_values, y_samples


@app.cell
def _(
    amp_slider,
    kernel_source,
    ls_slider,
    mo,
    n_negative_eigvals,
    n_points_slider,
    std_kernel_dd,
):
    _total = n_points_slider.value
    if n_negative_eigvals == 0:
        _msg = mo.md("✓ **Kernel is valid** — no negative eigenvalues.")
    else:
        _count = f"**{n_negative_eigvals}/{_total} negative eigenvalues**"
        if kernel_source.value == "standard":
            _kname = std_kernel_dd.element.options[std_kernel_dd.value] if hasattr(std_kernel_dd, "element") else std_kernel_dd.value
            _kname = {
                "rbf": "RBF",
                "matern12": "Matérn 1/2",
                "matern32": "Matérn 3/2",
                "matern52": "Matérn 5/2",
                "periodic": "Periodic",
                "linear": "Linear",
            }.get(std_kernel_dd.value, std_kernel_dd.value)
            if std_kernel_dd.value == "linear":
                _hint = f"The **{_kname}** kernel is rank-deficient by construction — its covariance matrix has many true zero eigenvalues that appear slightly negative due to floating-point arithmetic. Raise Jitter ε to suppress this."
            else:
                _hint = f"The **{_kname}** kernel (α={amp_slider.value:.2f}, ℓ={ls_slider.value:.2f}) is mathematically valid, but at this lengthscale the matrix is ill-conditioned and small negative eigenvalues arise from floating-point error. Raise Jitter ε or increase ℓ to stabilize."
            if n_negative_eigvals < _total * 0.1:
                _msg = mo.md(f"⚠ {_count} clipped. {_hint}")
            else:
                _msg = mo.md(f"✗ {_count} clipped. {_hint}")
        else:
            if n_negative_eigvals < _total * 0.1:
                _msg = mo.md(f"⚠ {_count} clipped. Minor distortion — try smoothing the kernel near r=0 or raising Jitter ε.")
            else:
                _msg = mo.md(f"✗ {_count} — the drawn spline is not a valid kernel. Samples may look nothing like your drawing. Draw a shape that **starts flat at r=0** and decays monotonically.")
    _msg
    return


@app.cell
def _(K_matrix, alt, mo, n_samples_slider, np, pl, x_values):
    _n = len(x_values)

    _method_text = mo.md(rf"""
    ### How GP samples are drawn

    Given $n = {_n}$ equally-spaced inputs $\mathbf{{x}} \in [0,1]$ and kernel matrix
    $K \in \mathbb{{R}}^{{n \times n}}$ with $K_{{ij}} = k(|x_i - x_j|)$:

    1. **Cholesky** — decompose $K = L L^\top$ (after jitter + eigenvalue clipping for PSD)
    2. **Standard normal** — draw $Z \in \mathbb{{R}}^{{n \times m}}$ with $Z_{{ij}} \overset{{\text{{iid}}}}{{\sim}} \mathcal{{N}}(0,1)$, giving $m = {n_samples_slider.value}$ samples
    3. **Transform** — samples are the rows of $\;(LZ)^\top$; each row $\sim \mathcal{{N}}(0, K)$

    Equivalently, for a single sample: $\mathbf{{f}} = L\mathbf{{z}}$, $\mathbf{{z}} \sim \mathcal{{N}}(0,I_n)$.
    """)

    _kdf = pl.DataFrame({
        "x1": np.repeat(x_values, _n).tolist(),
        "x2": np.tile(x_values, _n).tolist(),
        "k": K_matrix.ravel().tolist(),
    })
    _cov_chart = (
        alt.Chart(_kdf)
        .mark_rect()
        .encode(
            x=alt.X("x1:O", title="x₁", axis=alt.Axis(labels=False, ticks=False)),
            y=alt.Y("x2:O", title="x₂", axis=alt.Axis(labels=False, ticks=False)),
            color=alt.Color("k:Q", scale=alt.Scale(scheme="viridis"), title="k(x₁,x₂)"),
            tooltip=[
                alt.Tooltip("x1:Q", format=".2f"),
                alt.Tooltip("x2:Q", format=".2f"),
                alt.Tooltip("k:Q", format=".4f"),
            ],
        )
        .properties(title="Covariance matrix K", width=300, height=300)
    )

    mo.accordion({
        "Sampling method": _method_text,
        "Covariance matrix": _cov_chart,
    })
    return


@app.cell
def _(ContinuousParallelCoords, mo):
    gp_widget = mo.ui.anywidget(ContinuousParallelCoords(
        height=420,
        width=900,
        x_label="x",
        y_label="f(x)",
    ))
    gp_widget
    return (gp_widget,)


@app.cell
def _(gp_widget):
    gp_underlying = gp_widget.widget
    return (gp_underlying,)


@app.cell
def _(gp_underlying, x_values, y_samples):
    gp_underlying.x_values = x_values.tolist()
    gp_underlying.y_samples = y_samples.tolist()
    return


@app.cell
def _(gp_widget, mo):
    _n_filt = len(gp_widget.filtered_indices)
    _n_total = len(gp_widget.y_samples)
    _pct = 100 * _n_filt / _n_total if _n_total else 0
    mo.md(f"**{_n_filt} / {_n_total}** samples pass current constraints ({_pct:.1f}%)")
    return


@app.cell
def _(mo):
    mo.md("""
    ## Filtered Posterior Mean
    """)
    return


@app.cell
def _(alt, gp_widget, mo, np, pl, x_values):
    mo.stop(
        len(gp_widget.filtered_indices) == 0,
        mo.md("*No samples pass the current constraints — add a constraint by clicking in the plot above.*")
    )

    _filt = np.array([gp_widget.y_samples[i] for i in gp_widget.filtered_indices])
    _mean = _filt.mean(axis=0)
    _std = _filt.std(axis=0)

    _stats_df = pl.DataFrame({
        "x": x_values.tolist(),
        "mean": _mean.tolist(),
        "lo": (_mean - _std).tolist(),
        "hi": (_mean + _std).tolist(),
    })

    _band = (
        alt.Chart(_stats_df)
        .mark_area(opacity=0.25, color="#4477aa")
        .encode(x="x:Q", y="lo:Q", y2="hi:Q")
    )
    _line = (
        alt.Chart(_stats_df)
        .mark_line(color="#4477aa", strokeWidth=2)
        .encode(
            x=alt.X("x:Q", title="x"),
            y=alt.Y("mean:Q", title="f(x)"),
            tooltip=["x:Q", "mean:Q"],
        )
    )
    (_band + _line).properties(
        title="Mean ± 1 std of filtered samples",
        width=680, height=200,
    )
    return


@app.cell
def _(
    alt,
    amp_slider,
    bias_slider,
    kernel_source,
    kernel_spline_widget,
    ls_slider,
    np,
    period_slider,
    pl,
    shift_slider,
    slope_slider,
    std_kernel_dd,
):
    _r = np.linspace(0, 1, 200)

    if kernel_source.value == "standard":
        _k = std_kernel_dd.value
        _a = amp_slider.value
        _ls = ls_slider.value
        _std_fns = {
            "rbf":      lambda r: _a**2 * np.exp(-0.5 * r**2 / _ls**2),
            "matern12": lambda r: _a**2 * np.exp(-r / _ls),
            "matern32": lambda r: _a**2 * (1 + np.sqrt(3)*r/_ls) * np.exp(-np.sqrt(3)*r/_ls),
            "matern52": lambda r: _a**2 * (1 + np.sqrt(5)*r/_ls + 5*r**2/(3*_ls**2)) * np.exp(-np.sqrt(5)*r/_ls),
            "periodic": lambda r: _a**2 * np.exp(-2*np.sin(np.pi*r/period_slider.value)**2/_ls**2),
            "linear":   lambda r: bias_slider.value + slope_slider.value * (r - shift_slider.value)**2,
        }
        _active_k = _std_fns[_k](_r)
        _label = "Standard"
    else:
        _kvals = kernel_spline_widget.sampled_values
        if len(_kvals) == 100:
            _drawn_r = np.linspace(0, 1, 100)
            _active_r = _drawn_r
            _active_k = np.array(_kvals)
        else:
            _active_r = _r
            _active_k = np.exp(-0.5 * _r**2 / 0.3**2)
        _r = _active_r
        _label = "Drawn"

    _df = pl.DataFrame({
        "r": _r.tolist(),
        "k": _active_k.tolist(),
        "source": [_label] * len(_r),
    })
    _color_range = ["#b03060"] if kernel_source.value == "spline" else ["#4477aa"]
    alt.Chart(_df).mark_line(strokeWidth=2.5).encode(
        x=alt.X("r:Q", title="|x₁ − x₂|"),
        y=alt.Y("k:Q", title="k(r)"),
        color=alt.Color("source:N", scale=alt.Scale(range=_color_range), legend=None),
        tooltip=[alt.Tooltip("r:Q", format=".2f"), alt.Tooltip("k:Q", format=".3f")],
    ).properties(title="Kernel shape preview", width=560, height=200)
    return


if __name__ == "__main__":
    app.run()
