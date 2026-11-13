from pathlib import Path
import json
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
import pymupdf

S = Path(__file__).with_name("architecture_glyphs.py")
# Reuse the existing vector glyphs. No paper assets are overwritten.
ns = {"__file__": str(S), "__name__": "glyphs"}
text = S.read_text()
exec(text, ns)
plt.close(ns["fig"])
O = Path(__file__).resolve().parent.parent / "figures"
O.mkdir(exist_ok=True)
W, H = 1440, 570
fig = plt.figure(figsize=(6.875, 2.72), facecolor="white")
ax = fig.add_axes([0, 0, 1, 1])
ax.set(xlim=(0, W), ylim=(H, 0))
ax.axis("off")
INK = ns["INK"]
MUT = ns["MUT"]
BLUE = ns["BLUE"]
PURPLE = ns["PURPLE"]
GOLD = ns["GOLD"]
TEAL = ns["TEAL"]
PINK = ns["PINK"]
CYAN = ns["CYAN"]
LIME = ns["LIME"]
LAV = ns["LAV"]
GREEN = ns["GREEN"]
labels = []
node_checks = []


def txt(x, y, s, size=8.0, color=INK, bold=False, ha="center"):
    a = ax.text(
        x,
        y,
        s,
        fontsize=size,
        color=color,
        weight="bold" if bold else "normal",
        ha=ha,
        va="center",
        linespacing=1.06,
        zorder=10,
    )
    labels.append(a)
    return a


def box(x, y, w, h, fill, lw=0.7, r=8):
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


def arrow(points, color=INK, dash=False, lw=0.75):
    ls = (0, (3, 2.5)) if dash else "-"
    if len(points) > 2:
        a = np.array(points[:-1])
        ax.plot(a[:, 0], a[:, 1], color=color, lw=lw, ls=ls, zorder=4)
    ax.add_patch(
        FancyArrowPatch(
            points[-2],
            points[-1],
            arrowstyle="-|>",
            mutation_scale=5.2,
            lw=lw,
            color=color,
            linestyle=ls,
            shrinkA=0,
            shrinkB=1,
            zorder=5,
        )
    )


def module(x, y, w, h, label, fill, size=8):
    p = box(x, y, w, h, fill)
    t = txt(x + w / 2, y + (30 if "\n" in label else 20), label, size=size)
    node_checks.append((p, t))
    return p, t


ns.update(fig=fig, ax=ax, W=W, H=H, txt=txt, box=box, arrow=arrow)
grid = ns["grid"]
groups = ns["groups"]
strip = ns["strip"]
surface = ns["surface"]
axes_cloud = ns["axes_cloud"]
curve = ns["curve"]


def axes_cloud(cx, cy, w, h):
    # Preserve the original three starting directions and their projected cloud.
    toy = np.load(ns["I"] / "agf_initialization_3d_toy.npz")
    points3 = toy["vectors"] @ toy["directions"]
    points3 -= points3.mean(axis=0)
    points3 = points3 / points3.std(axis=0) * np.array([1.15, 0.70, 0.40])

    def project(p):
        return np.stack(
            [
                0.82 * p[..., 0] - 0.57 * p[..., 1],
                0.25 * p[..., 0] + 0.37 * p[..., 1] - 0.89 * p[..., 2],
            ],
            axis=-1,
        )

    points = project(points3)
    directions = project(np.eye(3) * 2.3)
    extent = np.max(np.abs(np.vstack([points, directions, np.zeros((1, 2))])), axis=0)
    scale = np.array([w, h]) / 2 / extent
    points = points * scale + [cx, cy]
    directions = directions * scale + [cx, cy]
    ax.scatter(
        *points.T,
        c=points3[:, 0],
        cmap="viridis",
        s=0.75,
        alpha=0.50,
        linewidths=0,
        zorder=5,
    )
    for endpoint, color in zip(directions, [TEAL, GOLD, PURPLE]):
        ax.add_patch(
            FancyArrowPatch(
                (cx, cy),
                endpoint,
                arrowstyle="-|>",
                mutation_scale=7.0,
                lw=1.2,
                color=color,
                shrinkA=0,
                shrinkB=0,
                zorder=9,
            )
        )


txt(20, 18, "Guided Group Featurizers (GGF)", 10.1, bold=True, ha="left")
txt(1417, 18, "Masks are used only during training", 7.6, MUT, ha="right")
box(20, 48, 675, 154, "#F7F8F9", r=11)
txt(35, 64, "Initialization", 8.8, bold=True, ha="left")
box(716, 48, 704, 154, "#F7F8F9", r=11)
txt(730, 64, "Adaptive loss weights", 8.8, bold=True, ha="left")
txt(1405, 64, "Every 10 steps", 7.5, PURPLE, ha="right")
module(35, 87, 145, 102, "Object\nmasks", PINK, 7.5)
grid(68, 145, 5, 7, 6, np.eye(5, 7) * 0.75 + 0.12, "Greens")
module(198, 87, 145, 102, "Object\npatches", PINK, 7.5)
grid(225, 145, 5, 8, 6, cmap="Blues")
module(361, 87, 145, 102, "Starting\naxes", GREEN, 7.5)
axes_cloud(433, 166, 116, 40)
module(524, 87, 155, 102, "Warm-up", CYAN, 7.5)
curve(539, 130, 45, 38)
groups(607, 129, [[0.6, 0.3, 0.9], [0.8, 0.4, 0.5]], w=18, h=35, gap=8)
for x in [180, 343, 506]:
    arrow([(x, 140), (x + 18, 140)], GOLD if x == 180 else INK)
module(731, 87, 219, 102, "Gradients +\nloss progress", LIME, 7.7)
surface(784, 162, 77, 33, "bowl", BLUE)
for i, col in enumerate([BLUE, TEAL, GOLD]):
    q = np.linspace(0, 1, 30)
    ax.plot(
        848 + q * 80,
        149 + 5 * i + 25 * (1 - np.exp(-(2 + i) * q)),
        c=col,
        lw=0.65,
        zorder=6,
    )
module(968, 87, 209, 102, "Match target\nstrengths", LAV, 7.7)
for k, (color, pos, target) in enumerate([(TEAL, 0.35, 0.5), (GOLD, 0.77, 0.55)]):
    x = 986 + k * 91
    ax.plot([x, x + 74], [156, 156], c="#B8C2C8", lw=0.9, zorder=6)
    ax.plot(x + 74 * pos, 156, "o", color=color, ms=3.8, zorder=7)
    ax.plot([x + 74 * target] * 2, [146, 166], c=INK, lw=0.8, zorder=6)
module(1195, 87, 210, 102, "Weight Adam\n+ bounds", LAV, 7.7)
for k, (color, pos) in enumerate([(TEAL, 0.55), (GOLD, 0.80)]):
    x = 1211 + k * 92
    ax.plot([x, x + 75], [156, 156], c="#BAB1C6", lw=1.2, zorder=6)
    for e in [x, x + 75]:
        ax.plot([e, e], [149, 163], c=MUT, lw=0.6, zorder=6)
    ax.plot(x + 75 * pos, 156, "o", c=color, ms=4, zorder=7)
for x in [950, 1177]:
    arrow([(x, 141), (x + 18, 141)], PURPLE, True)

txt(550, 239, "Guided Learning", 8.8, bold=True, ha="left")
module(20, 262, 235, 94, "Frozen DINOv3", LAV, 8.2)
grid(43, 302, 4, 4, 6, cmap="YlGn")
arrow([(76, 315), (98, 315)])
strip(110, 306, [0.2, 0.8, 0.4, 0.6, 0.95, 0.3, 0.7, 0.5], w=118, h=21)
module(285, 262, 235, 94, "Linear encoder", CYAN, 8.2)
strip(300, 307, [0.2, 0.7, 0.4, 0.9, 0.3, 0.6], w=40, h=23)
arrow([(345, 318), (360, 318)])
grid(365, 300, 5, 6, 6, cmap="Blues")
arrow([(405, 318), (423, 318)])
groups(432, 301, [[0.8, 0.4, 0.6], [0.5, 0.3, 0.6], [0.9, 0.4, 0.7]], w=18, h=34, gap=7)
module(550, 262, 235, 94, "Group Top-k", LIME, 8.2)
groups(
    563,
    300,
    [[0.8, 0.4, 0.6], [0.1, 0.2, 0.1], [0.9, 0.6, 0.3]],
    selected=[0, 2],
    w=18,
    h=34,
    gap=6,
)
arrow([(636, 318), (655, 318)])
groups(
    665,
    300,
    [[0.8, 0.4, 0.6], [0, 0, 0], [0.9, 0.6, 0.3]],
    selected=[0, 2],
    w=18,
    h=34,
    gap=6,
)
module(815, 262, 235, 94, "Linear decoder", CYAN, 8.2)
surface(845, 319, 38, 29, "plane", TEAL)
txt(873, 320, "+", 9)
surface(905, 319, 38, 29, "plane", PURPLE)
arrow([(929, 319), (947, 319)])
strip(958, 308, [0.4, 0.8, 0.2, 0.6, 0.3, 0.9], w=76, h=21, cmap="PuBuGn")
module(1080, 262, 325, 94, "Reconstructed activations", GREEN, 8.2)
strip(1097, 307, [0.2, 0.8, 0.4, 0.6, 0.95, 0.3, 0.7, 0.5], w=124, h=23)
arrow([(1228, 319), (1250, 319)])
strip(
    1258,
    307,
    [0.25, 0.74, 0.45, 0.58, 0.89, 0.34, 0.68, 0.55],
    w=130,
    h=23,
    cmap="PuBuGn",
)
for x in [255, 520, 785, 1050]:
    arrow([(x, 310), (x + 30, 310)])
# Initialize the encoder and decoder, leaving the backbone frozen.
arrow([(602, 189), (602, 220), (400, 220), (400, 262)], MUT)
arrow([(602, 220), (933, 220), (933, 262)], MUT)
arrow([(137, 262), (137, 214), (270, 214), (270, 189)], MUT, lw=0.65)

module(550, 410, 340, 92, "N logistic classifier heads", PINK, 8.1)
groups(567, 449, [[0.4, 0.8, 0.5], [0.7, 0.3, 0.6]], w=15, h=30, gap=7)
arrow([(612, 464), (634, 464)])
txt(674, 464, "Norm", 8)
arrow([(710, 464), (730, 464)])
q = np.linspace(-3, 3, 60)
for col, scale, bias in [(TEAL, 1.3, -0.5), (GOLD, 0.9, 0.4), (PURPLE, 1.7, 0.1)]:
    ax.plot(
        745 + (q + 3) * 20,
        482 - 35 / (1 + np.exp(-(scale * q + bias))),
        color=col,
        lw=0.8,
        zorder=7,
    )
box(925, 410, 480, 104, PINK)
txt(1165, 430, "Joint training losses", 8.2)
for k, (label, col) in enumerate(
    [("Reconstruction", BLUE), ("Object\ncontrast", TEAL), ("Patch\ndetection", GOLD)]
):
    tx = 1005 + k * 158
    txt(tx, 467, label, 7.0 if k == 0 else 7.5)
    ax.add_patch(
        Rectangle((tx - 42, 498), 84, 6, facecolor=col, edgecolor="none", zorder=6)
    )
arrow([(667, 356), (667, 410)])
arrow([(890, 461), (925, 461)])
arrow([(854, 356), (854, 385), (1120, 385), (1120, 410)])
arrow([(1242, 356), (1242, 382), (1338, 382), (1338, 410)])
# Masks supply training targets, and loss gradients feed the separate controller.
arrow(
    [(35, 141), (9, 141), (9, 521), (1415, 521), (1415, 463), (1405, 463)],
    GOLD,
    lw=0.65,
)
arrow(
    [(1405, 445), (1430, 445), (1430, 212), (840, 212), (840, 189)],
    PURPLE,
    True,
    lw=0.65,
)
arrow(
    [(1301, 189), (1301, 232), (1415, 232), (1415, 394), (1395, 394), (1395, 410)],
    PURPLE,
    True,
    lw=0.65,
)
# Model Adam updates only the encoder, decoder and logistic classifier heads.
arrow([(925, 485), (916, 485), (916, 534), (300, 534), (300, 356)], BLUE, True, lw=0.65)
arrow([(916, 534), (900, 534), (900, 370), (933, 370), (933, 356)], BLUE, True, lw=0.65)
arrow([(720, 534), (720, 502)], BLUE, True, lw=0.65)
for x, label, col, dash in [
    (30, "Forward flow", INK, False),
    (310, "Mask targets", GOLD, False),
    (615, "Model Adam", BLUE, True),
    (962, "Weight controller", PURPLE, True),
]:
    arrow([(x, 554), (x + 35, 554)], col, dash, lw=0.65)
    txt(x + 47, 554, label, 7.6, col, ha="left")
fig.canvas.draw()
ren = fig.canvas.get_renderer()
for t in labels:
    b = t.get_window_extent(ren)
    assert b.x0 >= 0 and b.x1 <= fig.bbox.x1 and b.y0 >= 0 and b.y1 <= fig.bbox.y1, (
        t.get_text()
    )
for p, t in node_checks:
    pb = p.get_window_extent(ren)
    tb = t.get_window_extent(ren)
    assert (
        pb.x0 + 1 < tb.x0
        and tb.x1 < pb.x1 - 1
        and pb.y0 + 1 < tb.y0
        and tb.y1 < pb.y1 - 1
    ), t.get_text()
for ext in ["pdf", "svg", "png"]:
    fig.savefig(O / ("agf_architecture." + ext), dpi=600)
plt.close(fig)
(O / "agf_architecture_audit.json").write_text(
    json.dumps(
        dict(
            preview_only=False,
            paper_changed=True,
            width_inches=6.875,
            height_inches=2.72,
            min_font_pt=7.0,
            main_font_pt=8.2,
            encoder_decoder_gate_unchanged=True,
            N_category_heads=True,
            reused_existing_vector_glyphs=True,
            guided_learning_heading=True,
        ),
        indent=2,
    )
)
print("Built the GGF architecture diagram with visible starting axes.")
