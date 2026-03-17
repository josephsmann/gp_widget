"""
ContinuousParallelCoords — an anywidget for visualising distributions over functions.

Unlike a standard parallel coordinates plot (which has a finite set of named axes),
this widget treats the horizontal axis as a **continuous input domain**.  Each polyline
is a function sample f(x), and the user can place brush constraints at any x-position
to interactively filter down to samples consistent with f(x) ∈ [lo, hi].

Typical use-cases
-----------------
- GP prior / posterior samples
- Ensembles of time-series predictions
- Bayesian neural-network function draws
- Any "spaghetti plot" where you want to query point-wise ranges

Interactions
------------
- Click anywhere in the plot area  → add a constraint axis centred on that (x, y)
- Drag the body of a constraint    → move it vertically
- Drag the top/bottom handle       → resize the y-range
- Double-click the body            → remove the constraint

Widget properties (Python-side)
---------------------------------
x_values         : list[float]       – shared x-grid for all samples
y_samples        : list[list[float]] – shape (n_samples, len(x_values))
colors           : list[str]         – one CSS color per sample (optional)
height           : int               – SVG height in px  (default 400)
width            : int               – SVG width  in px  (0 = auto 700)
brush_axes       : dict[str, [lo,hi]]– constraint state synced with JS
filtered_indices : list[int]         – indices of samples that pass all constraints
"""

from __future__ import annotations

import textwrap
from typing import Any

import anywidget
import traitlets

# ---------------------------------------------------------------------------
# JavaScript (ESM module)
# ---------------------------------------------------------------------------

_ESM = textwrap.dedent("""\
function render({ model, el }) {
  const M = { top: 20, right: 20, bottom: 50, left: 50 };
  let dragState = null;
  let rafPending = false;
  let lastEv = null;

  // ── scales ──────────────────────────────────────────────────────────────
  function sc() {
    const w  = model.get('width') || 700;
    const h  = model.get('height');
    const xs = model.get('x_values');
    const ss = model.get('y_samples');

    const flat = ss.flat();
    const rawLo = Math.min(...flat), rawHi = Math.max(...flat);
    const pad   = (rawHi - rawLo || 1) * 0.06;
    const yLo   = rawLo - pad,  yHi = rawHi + pad;
    const xLo   = Math.min(...xs), xHi = Math.max(...xs);
    const iW    = w - M.left - M.right;
    const iH    = h - M.top  - M.bottom;

    return {
      w, h, iW, iH, xLo, xHi, yLo, yHi,
      px : x  => M.left + (x  - xLo) / (xHi - xLo) * iW,
      py : y  => M.top  + (1  - (y - yLo) / (yHi - yLo)) * iH,
      ix : px => xLo   + (px - M.left)  / iW * (xHi - xLo),
      iy : py => yLo   + (1  - (py - M.top) / iH) * (yHi - yLo),
    };
  }

  // ── linear interpolation of sample[i] at x ──────────────────────────────
  function lerp(sample, x, xs) {
    if (x <= xs[0])               return sample[0];
    if (x >= xs[xs.length - 1])   return sample[xs.length - 1];
    for (let i = 0; i < xs.length - 1; i++) {
      if (x >= xs[i] && x <= xs[i + 1]) {
        const t = (x - xs[i]) / (xs[i + 1] - xs[i]);
        return sample[i] * (1 - t) + sample[i + 1] * t;
      }
    }
    return NaN;
  }

  // ── which samples pass all brush constraints ─────────────────────────────
  function getFiltered(brushes, xs, samples) {
    const entries = Object.entries(brushes);
    if (!entries.length) return samples.map((_, i) => i);
    return samples.reduce((acc, s, i) => {
      const ok = entries.every(([k, [lo, hi]]) => {
        const y = lerp(s, parseFloat(k), xs);
        return y >= lo && y <= hi;
      });
      if (ok) acc.push(i);
      return acc;
    }, []);
  }

  // ── SVG element helper ───────────────────────────────────────────────────
  function svg_el(tag, attrs, styles) {
    const e = document.createElementNS('http://www.w3.org/2000/svg', tag);
    if (attrs)  for (const [k, v] of Object.entries(attrs))  e.setAttribute(k, v);
    if (styles) for (const [k, v] of Object.entries(styles)) e.style[k] = v;
    return e;
  }

  // ── main draw ────────────────────────────────────────────────────────────
  function draw() {
    el.innerHTML = '';
    el.style.fontFamily = 'sans-serif';

    const S       = sc();
    const xs      = model.get('x_values');
    const samples = model.get('y_samples');
    const brushes = model.get('brush_axes');
    const colors  = model.get('colors');
    const hasBrush = Object.keys(brushes).length > 0;

    const filtered = getFiltered(brushes, xs, samples);
    const fSet     = new Set(filtered);
    // Update filtered_indices locally; save_changes is deferred to user interactions
    // to avoid a Python→JS echo that would overwrite brush_axes mid-click.
    model.set('filtered_indices', filtered);

    // root SVG
    const root = svg_el('svg', { width: S.w, height: S.h }, { userSelect: 'none' });
    el.appendChild(root);

    // hint text
    const hint = document.createElement('div');
    hint.textContent = 'Click to add constraint  ·  drag body to move  ·  drag handles to resize  ·  dbl-click to remove';
    hint.style.cssText = 'font-size:11px;color:#bbb;text-align:center;margin-top:3px';
    el.appendChild(hint);

    // chart background
    root.appendChild(svg_el('rect', {
      x: M.left, y: M.top, width: S.iW, height: S.iH,
      fill: '#f9f9f9', stroke: '#ddd',
    }));

    // y grid + labels
    for (let i = 0; i <= 5; i++) {
      const yv  = S.yLo + (i / 5) * (S.yHi - S.yLo);
      const pyv = S.py(yv);
      root.appendChild(svg_el('line', { x1: M.left, y1: pyv, x2: M.left + S.iW, y2: pyv, stroke: '#eaeaea' }));
      const t = svg_el('text', { x: M.left - 6, y: pyv + 4, 'text-anchor': 'end', 'font-size': 10, fill: '#999' });
      t.textContent = yv.toFixed(1);
      root.appendChild(t);
    }

    // x labels
    for (let i = 0; i <= 5; i++) {
      const xv  = S.xLo + (i / 5) * (S.xHi - S.xLo);
      const pxv = S.px(xv);
      const t = svg_el('text', { x: pxv, y: M.top + S.iH + 16, 'text-anchor': 'middle', 'font-size': 10, fill: '#999' });
      t.textContent = xv.toFixed(2);
      root.appendChild(t);
    }

    // axis labels
    const xLabel = svg_el('text', {
      x: M.left + S.iW / 2, y: M.top + S.iH + 38,
      'text-anchor': 'middle', 'font-size': 11, fill: '#888',
    });
    xLabel.textContent = model.get('x_label') || 'x';
    root.appendChild(xLabel);

    const yLabel = svg_el('text', {
      x: 12, y: M.top + S.iH / 2,
      'text-anchor': 'middle', 'font-size': 11, fill: '#888',
      transform: `rotate(-90, 12, ${M.top + S.iH / 2})`,
    });
    yLabel.textContent = model.get('y_label') || 'f(x)';
    root.appendChild(yLabel);

    // polylines: unfiltered first, filtered on top
    for (const pass of [false, true]) {
      for (let i = 0; i < samples.length; i++) {
        const ok = fSet.has(i);
        if (ok !== pass) continue;
        const pts = xs.map((x, j) => `${S.px(x)},${S.py(samples[i][j])}`).join(' ');
        root.appendChild(svg_el('polyline', {
          points: pts, fill: 'none',
          stroke:        (colors && colors[i]) ? colors[i] : '#4477aa',
          'stroke-width': ok ? 1.2 : 0.4,
          opacity:        ok ? 0.55 : (hasBrush ? 0.05 : 0.20),
        }));
      }
    }

    // brush axes
    const BW = 14, HH = 7;
    for (const [xKey, [lo, hi]] of Object.entries(brushes)) {
      const bx  = parseFloat(xKey);
      const bpx = S.px(bx);
      const top = S.py(hi), bot = S.py(lo);
      const bH  = Math.max(bot - top, 4);

      // dashed vertical guide
      root.appendChild(svg_el('line', {
        x1: bpx, y1: M.top, x2: bpx, y2: M.top + S.iH,
        stroke: '#4477aa', 'stroke-width': 1, 'stroke-dasharray': '4 3', opacity: 0.35,
      }));

      // body
      const body = svg_el('rect', {
        x: bpx - BW / 2, y: top, width: BW, height: bH,
        fill: 'rgba(68,119,170,0.18)', stroke: '#4477aa', 'stroke-width': 1.5, rx: 3,
      }, { cursor: 'move' });
      root.appendChild(body);

      // range labels
      const hiLbl = svg_el('text', { x: bpx + BW / 2 + 3, y: top + 9,  'font-size': 9, fill: '#4477aa' });
      hiLbl.textContent = hi.toFixed(2);
      root.appendChild(hiLbl);

      const loLbl = svg_el('text', { x: bpx + BW / 2 + 3, y: bot,      'font-size': 9, fill: '#4477aa' });
      loLbl.textContent = lo.toFixed(2);
      root.appendChild(loLbl);

      // x label below axis
      const xl = svg_el('text', {
        x: bpx, y: M.top + S.iH + 34,
        'text-anchor': 'middle', 'font-size': 10, fill: '#4477aa', 'font-weight': 'bold',
      });
      xl.textContent = `x=${bx.toFixed(3)}`;
      root.appendChild(xl);

      // top & bottom handles
      const topH = svg_el('rect', { x: bpx - BW/2, y: top - HH/2, width: BW, height: HH, fill: '#4477aa', rx: 2 }, { cursor: 'ns-resize' });
      const botH = svg_el('rect', { x: bpx - BW/2, y: bot - HH/2, width: BW, height: HH, fill: '#4477aa', rx: 2 }, { cursor: 'ns-resize' });
      root.appendChild(topH);
      root.appendChild(botH);

      // double-click removes constraint
      body.addEventListener('dblclick', e => {
        e.stopPropagation();
        const cur = Object.assign({}, model.get('brush_axes') || {});
        delete cur[xKey];
        model.set('brush_axes', cur);
        model.save_changes();
      });

      const startDrag = type => e => {
        e.stopPropagation(); e.preventDefault();
        dragState = { type, xKey, startY: e.clientY, startX: e.clientX, startRange: [lo, hi], startXValue: bx };
      };
      body.addEventListener('mousedown', startDrag('move'));
      topH.addEventListener('mousedown', startDrag('top'));
      botH.addEventListener('mousedown', startDrag('bottom'));
    }

    // click on chart area → add new constraint
    root.addEventListener('click', e => {
      if (dragState) return;
      const r   = root.getBoundingClientRect();
      const px  = e.clientX - r.left, py = e.clientY - r.top;
      if (px < M.left || px > M.left + S.iW) return;
      if (py < M.top  || py > M.top  + S.iH) return;
      const x    = S.ix(px), y = S.iy(py);
      const span = (S.yHi - S.yLo) * 0.20;
      const key  = x.toFixed(6);
      const cur  = model.get('brush_axes') || {};
      model.set('brush_axes', Object.assign({}, cur, { [key]: [y - span / 2, y + span / 2] }));
      model.save_changes();
    });
  }

  // ── drag handlers (RAF-throttled) ────────────────────────────────────────
  function onMove(e) {
    if (!dragState) return;
    lastEv = e;
    if (rafPending) return;
    rafPending = true;
    requestAnimationFrame(() => {
      rafPending = false;
      if (!dragState || !lastEv) return;
      const S  = sc();
      const dy = lastEv.clientY - dragState.startY;
      const dY = -dy / S.iH * (S.yHi - S.yLo);
      const [sLo, sHi] = dragState.startRange;
      const minSpan    = (S.yHi - S.yLo) * 0.01;
      const cur        = Object.assign({}, model.get('brush_axes') || {});
      if (dragState.type === 'move') {
        const dx    = lastEv.clientX - dragState.startX;
        const dX    = dx / S.iW * (S.xHi - S.xLo);
        const newX  = Math.max(S.xLo, Math.min(S.xHi, dragState.startXValue + dX));
        const newKey = newX.toFixed(6);
        delete cur[dragState.xKey];           // remove from old x position
        cur[newKey] = [sLo + dY, sHi + dY];  // place at new x position
        dragState.xKey = newKey;              // track so next frame deletes the right key
      } else if (dragState.type === 'top') {
        cur[dragState.xKey] = [sLo, Math.max(sHi + dY, sLo + minSpan)];
      } else if (dragState.type === 'bottom') {
        cur[dragState.xKey] = [Math.min(sLo + dY, sHi - minSpan), sHi];
      }
      // Only update local model state — no save_changes during drag to avoid
      // Python echoing stale positions back mid-drag and causing jitter.
      model.set('brush_axes', cur);
    });
  }

  function onUp() {
    if (dragState) {
      dragState = null;
      model.save_changes();   // sync final position to Python once drag ends
    }
  }

  document.addEventListener('mousemove', onMove);
  document.addEventListener('mouseup',   onUp);

  draw();
  model.on('change:x_values',  draw);
  model.on('change:y_samples', draw);
  model.on('change:brush_axes', draw);
  model.on('change:colors',    draw);
  model.on('change:height',    draw);
  model.on('change:width',     draw);

  return () => {
    document.removeEventListener('mousemove', onMove);
    document.removeEventListener('mouseup',   onUp);
  };
}

export default { render };
""")

# ---------------------------------------------------------------------------
# Python widget class
# ---------------------------------------------------------------------------

class ContinuousParallelCoords(anywidget.AnyWidget):
    """Interactive parallel coordinates widget for distributions over functions.

    Parameters
    ----------
    x_values : list[float]
        The shared x-grid at which every sample is evaluated.
    y_samples : list[list[float]]  or  2-D array-like
        Function samples, shape ``(n_samples, len(x_values))``.
    colors : list[str], optional
        One CSS color string per sample.  ``None`` uses a uniform blue.
    height : int
        Widget height in pixels (default 400).
    width : int
        Widget width in pixels (0 → auto 700).
    x_label / y_label : str
        Axis label text.
    """

    _esm = _ESM

    # synced with JS
    x_values         = traitlets.List(traitlets.Float()).tag(sync=True)
    y_samples        = traitlets.List(traitlets.List(traitlets.Float())).tag(sync=True)
    colors           = traitlets.List(traitlets.Unicode()).tag(sync=True)
    height           = traitlets.Int(400).tag(sync=True)
    width            = traitlets.Int(0).tag(sync=True)
    x_label          = traitlets.Unicode("x").tag(sync=True)
    y_label          = traitlets.Unicode("f(x)").tag(sync=True)
    brush_axes       = traitlets.Dict({}).tag(sync=True)
    filtered_indices = traitlets.List(traitlets.Int()).tag(sync=True)

    def __init__(
        self,
        x_values,
        y_samples,
        *,
        colors=None,
        height=400,
        width=0,
        x_label="x",
        y_label="f(x)",
    ):
        xs  = [float(v) for v in x_values]
        ys  = [[float(v) for v in row] for row in y_samples]
        col = list(colors) if colors is not None else []
        super().__init__(
            x_values=xs,
            y_samples=ys,
            colors=col,
            height=height,
            width=width,
            x_label=x_label,
            y_label=y_label,
            brush_axes={},
            filtered_indices=list(range(len(ys))),
        )

    # ------------------------------------------------------------------
    # Convenience accessors
    # ------------------------------------------------------------------

    @property
    def filtered_samples(self):
        """Return ``y_samples`` rows that pass all current constraints."""
        return [self.y_samples[i] for i in self.filtered_indices]

    @property
    def filtered_as_numpy(self):
        """Return filtered samples as a ``numpy`` array, shape (k, n_x)."""
        import numpy as np
        return np.array(self.filtered_samples)

    @property
    def filtered_as_polars(self):
        """Return filtered samples as a ``polars`` DataFrame."""
        import polars as pl
        xs = self.x_values
        data = {f"x={x:.4g}": [row[j] for row in self.filtered_samples]
                for j, x in enumerate(xs)}
        return pl.DataFrame(data)


def _to_list(arr) -> list:
    """Convert numpy / polars arrays to plain Python lists."""
    if hasattr(arr, "tolist"):
        return arr.tolist()
    return list(arr)
