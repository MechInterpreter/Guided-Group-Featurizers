"""Top-down GGF schematic. Node glyphs illustrate operations, not measured data."""

from pathlib import Path
import json, hashlib
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
from matplotlib.collections import PolyCollection
import pymupdf

R = Path(__file__).resolve().parent.parent
O = R / "figures"
I = R / "figure_sources/inputs"
plt.rcParams.update(
    {"font.family": "DejaVu Sans", "pdf.fonttype": 42, "svg.fonttype": "none"}
)
INK = "#1F2B33"
MUT = "#52646F"
BLUE = "#357AA5"
PURPLE = "#7954A3"
GOLD = "#BA8734"
TEAL = "#078E83"
PINK = "#F8E5E4"
CYAN = "#D7EDF5"
LIME = "#EFF0CA"
LAV = "#E6E2F1"
GREEN = "#DEEDDE"
W, H = 1200, 830
fig = plt.figure(figsize=(12.5, 7.30), facecolor="white")
ax = fig.add_axes([0, 0, 1, 1])
ax.set(xlim=(0, W), ylim=(H, 0))
ax.axis("off")
checks = []


def txt(x, y, s, size=17, color=INK, bold=False, ha="center"):
    # Express sizes at the final .91-textwidth placement, preserving print readability.
    printed_pt = 10.2 if bold and size >= 21 else (9.35 if bold else 9.02)
    native_pt = printed_pt / ((0.91 * 6.875) / 12.5)
    return ax.text(
        x,
        y,
        s,
        fontsize=native_pt,
        color=color,
        weight="bold" if bold else "normal",
        ha=ha,
        va="center",
        linespacing=1.13,
        zorder=8,
    )


def box(x, y, w, h, fill, lw=1.7, r=10):
    p = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle=f"round,pad=0,rounding_size={r}",
        facecolor=fill,
        edgecolor=INK,
        lw=lw,
        zorder=2,
    )
    ax.add_patch(p)
    return p


def arrow(points, color=INK, dash=False, lw=1.6):
    ls = (0, (4, 3)) if dash else "-"
    if len(points) > 2:
        a = np.array(points[:-1])
        ax.plot(
            a[:, 0],
            a[:, 1],
            color=color,
            lw=lw,
            ls=ls,
            zorder=4,
            solid_capstyle="round",
        )
    ax.add_patch(
        FancyArrowPatch(
            points[-2],
            points[-1],
            arrowstyle="-|>",
            mutation_scale=12,
            lw=lw,
            color=color,
            linestyle=ls,
            shrinkA=0,
            shrinkB=1,
            zorder=5,
        )
    )


def module(x, y, w, h, label, fill):
    p = box(x, y, w, h, fill)
    t = txt(x + w / 2, y + 21, label, size=16.5, bold=False)
    checks.append((p, t))
    return p


def grid(x, y, n, m, step, values=None, cmap="Blues"):
    if values is None:
        values = np.linspace(0.15, 0.95, n * m).reshape(n, m)
    for j in range(n):
        for k in range(m):
            ax.add_patch(
                Rectangle(
                    (x + k * step, y + j * step),
                    step - 1,
                    step - 1,
                    facecolor=plt.get_cmap(cmap)(values[j, k]),
                    edgecolor="none",
                    zorder=6,
                )
            )


def vec(x, y, vals, step=7, cmap="PuBuGn"):
    grid(x, y, len(vals), 1, step, np.array(vals)[:, None], cmap)


def groups(x, y, vals, selected=None, w=23, h=36, gap=9):
    for j, v in enumerate(vals):
        col = TEAL if selected is None or j in selected else "#BFCAD0"
        ax.add_patch(
            Rectangle(
                (x + j * (w + gap), y),
                w,
                h,
                facecolor="white",
                edgecolor=col,
                lw=1.1,
                zorder=6,
            )
        )
        for q, a in enumerate(v):
            ax.add_patch(
                Rectangle(
                    (x + 2 + j * (w + gap), y + 2 + q * (h - 4) / len(v)),
                    (w - 4) * abs(a),
                    max(2, (h - 5) / len(v) - 1),
                    facecolor=col,
                    alpha=0.35 + 0.6 * abs(a),
                    edgecolor="none",
                    zorder=7,
                )
            )


def strip(x, y, values, w=118, h=19, cmap="YlOrBr"):
    for j, v in enumerate(values):
        ax.add_patch(
            Rectangle(
                (x + j * w / len(values), y),
                w / len(values) - 1,
                h,
                facecolor=plt.get_cmap(cmap)(0.12 + 0.78 * v),
                edgecolor="white",
                lw=0.5,
                zorder=6,
            )
        )
    ax.add_patch(
        Rectangle((x, y), w, h, facecolor="none", edgecolor=MUT, lw=0.7, zorder=7)
    )


def surface(cx, cy, w, h, kind="bowl", color=TEAL):
    # Orthographic vector projections. Geometry is an illustration, not model data.
    def project(p):
        p = np.asarray(p)
        return np.stack(
            [
                0.86 * p[..., 0] - 0.51 * p[..., 1],
                0.25 * p[..., 0] + 0.43 * p[..., 1] - 0.87 * p[..., 2],
            ],
            axis=-1,
        )

    u = np.linspace(-1, 1, 12)
    v = np.linspace(-1, 1, 12)
    X, Y = np.meshgrid(u, v)
    if kind == "bowl":
        Z = 0.72 * (X * X + Y * Y)
    elif kind == "plane":
        Z = 0.23 * X - 0.18 * Y
    else:
        Z = 0.25 * np.sin(2 * X) * np.cos(Y)
    Q = project(np.stack([X, Y, Z], axis=-1))
    lo = Q.reshape(-1, 2).min(0)
    hi = Q.reshape(-1, 2).max(0)
    scale = np.array([w, h]) / (hi - lo)

    def pos(p):
        return (project(p) - (lo + hi) / 2) * scale + np.array([cx, cy])

    points = (Q - (lo + hi) / 2) * scale + np.array([cx, cy])
    cells = []
    for j in range(11):
        for k in range(11):
            cells.append(points[[j, j, j + 1, j + 1], [k, k + 1, k + 1, k]])
    ax.add_collection(
        PolyCollection(cells, facecolors=color, edgecolors="none", alpha=0.11, zorder=5)
    )
    for q in range(0, 12, 2):
        ax.plot(*points[q].T, color=color, alpha=0.48, lw=0.5, zorder=6)
        ax.plot(*points[:, q].T, color=color, alpha=0.48, lw=0.5, zorder=6)
    if kind == "bowl":
        tt = np.linspace(0, 1, 60)
        r = 0.94 * np.exp(-2.6 * tt)
        phi = 2.6 * np.pi * tt + 0.6
        xx = r * np.cos(phi)
        yy = r * np.sin(phi)
        zz = 0.72 * (xx * xx + yy * yy) + 0.035
        q = pos(np.stack([xx, yy, zz], axis=-1))
        ax.plot(*q.T, color=GOLD, lw=1.9, zorder=7)
        ax.scatter(*q[-1], s=16, color=GOLD, edgecolor="white", lw=0.7, zorder=8)
        arrow([q[35], q[40]], GOLD, lw=1.4)
    else:
        origin = pos([0, 0, 0])
        end = pos([0.65, 0.35, 0.23 * 0.65 - 0.18 * 0.35])
        arrow([origin, end], color, lw=1.7)


def axes_cloud(cx, cy, w, h):
    toy = np.load(I / "agf_initialization_3d_toy.npz")
    X = toy["vectors"]
    U = toy["directions"]
    X = X @ U
    X = X - X.mean(axis=0)
    X = X / X.std(axis=0) * np.array([1.15, 0.70, 0.40])
    U = np.eye(3)

    def proj(p):
        return np.stack(
            [
                0.82 * p[..., 0] - 0.57 * p[..., 1],
                0.25 * p[..., 0] + 0.37 * p[..., 1] - 0.89 * p[..., 2],
            ],
            axis=-1,
        )

    points = proj(X)
    d = proj(U.T * 2.3)
    allp = np.vstack([points, d, np.zeros((1, 2))])
    extent = np.max(np.abs(allp), axis=0)
    scale = np.array([w, h]) / 2 / extent
    points = points * scale + [cx, cy]
    d = d * scale + [cx, cy]
    ax.scatter(
        *points.T, c=X[:, 0], cmap="viridis", s=3.5, alpha=0.58, linewidths=0, zorder=5
    )
    for q, c in zip(d, [TEAL, GOLD, PURPLE]):
        arrow([(cx, cy), q], c, lw=1.8)


def curve(x, y, w, h, color=TEAL):
    ax.plot([x, x, x + w], [y, y + h, y + h], color=MUT, lw=0.8, zorder=5)
    t = np.linspace(0, 1, 50)
    ax.plot(
        x + w * t,
        y + h * (0.13 + 0.75 * (1 - np.exp(-4 * t))),
        color=color,
        lw=2,
        zorder=6,
    )
