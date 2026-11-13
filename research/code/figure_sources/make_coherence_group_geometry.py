"""Measured image-to-cloud panels for the response-geometry figure."""

from pathlib import Path
import json, hashlib
import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter, map_coordinates
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch
from matplotlib.path import Path as MplPath
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from mpl_toolkits.mplot3d import proj3d

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "figures"
OUT.mkdir(exist_ok=True)
INPUT = ROOT / "figure_sources/inputs/coherence_geometry"
sel = json.loads((INPUT / "selection.json").read_text())
audit = json.loads((INPUT / "pca_verification.json").read_text())
proof = json.loads((INPUT / "review_display.json").read_text())
A = np.load(INPUT / "measured_pca.npz")
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
assert sha(INPUT / "measured_pca.npz") == audit["arrays_sha256"]
assert sha(INPUT / "selection.json") == audit["portable_selection_sha256"]
assert sha(INPUT / "review_display.json") == audit["portable_display_sha256"]
assert sha(INPUT / "image_assets.json") == audit["portable_image_assets_sha256"]
images = INPUT / "images"
assets = {
    x["image_id"]: x
    for x in json.loads((INPUT / "image_assets.json").read_text())["images"]
}
index = {int(i): j for j, i in enumerate(A["image_ids"])}
W = 502.0675048828125 / 72
H = 278.6893615722656 / 72
# Divide the existing combined panel area equally, as requested by the user.
TOP = BOTTOM = (
    171.1999969482422 * 0.82 / 72
    + 129.60000610351562 * (502.0675048828125 / 495) * 0.9 / 72
) / 2
plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 7.2,
        "pdf.fonttype": 42,
        "svg.fonttype": "none",
    }
)
fig = plt.figure(figsize=(W, H), facecolor="white")


def box(x, y, w, h):
    return [x / W, y / H, w / W, h / H]


def field(values):
    v = np.asarray(values).reshape(14, 14)
    t = (np.arange(224) + 0.5) * 14 / 224 - 0.5
    yy, xx = np.meshgrid(t, t, indexing="ij")
    return np.clip(
        map_coordinates(
            gaussian_filter(v, 0.35, mode="nearest"), [yy, xx], order=3, mode="nearest"
        ),
        v.min(),
        v.max(),
    )


methods = [
    ("ggf", "GGF", "Gloves"),
    ("slics", "SLiCS", "Faces + unrelated regions"),
    ("supervised_vanilla", "Supervised Vanilla", "Rackets + unrelated regions"),
    ("vanilla", "Vanilla BSF", "Unrelated marked regions"),
]
records = []
for panel, (method, name, theme) in enumerate(methods):
    row, col = divmod(panel, 2)
    x0 = 0.035 + col * (W / 2 + 0.025)
    width = W / 2 - 0.09
    rowbottom = (BOTTOM + 0.20) if row == 0 else 0.065
    rowheight = TOP if row == 0 else BOTTOM
    ytop = rowbottom + rowheight
    c = next(
        x
        for x in sel["cases"]
        if x["case_id"]
        == next(g["case_id"] for g in audit["groups"] if g["method"] == method)
    )
    rec = next(g for g in audit["groups"] if g["method"] == method)
    p = next(g for g in proof["panels"] if g["method"] == method)
    colors = p["colors_by_view"]
    fits = p["whole_high_fit_counts"]
    fig.text(
        x0 / W,
        (ytop - 0.07) / H,
        f"({chr(97 + panel)}) {name} (group {c['unit']})",
        fontsize=8.1,
        fontweight="bold",
        va="top",
        color="#182e3c",
    )
    fig.text(
        x0 / W,
        (ytop - 0.215) / H,
        f"{theme}  |  {p['specific_pairs_yes']}/3 specific pairs",
        fontsize=6.65,
        va="top",
        color="#4c606c",
    )
    tile = 0.415
    gap = 0.025
    grid_y = rowbottom + 0.18
    cloud = A[method + "_reference_pc"]
    marked = A[method + "_marked_pc"]
    low, high = np.percentile(cloud, [2, 98], axis=0)
    # Include marked examples in the color range, so their colors do not all saturate.
    low = np.minimum(low, marked.min(0))
    high = np.maximum(high, marked.max(0))

    def rgb(points):
        return 0.04 + 0.92 * np.clip(
            (points - low) / np.maximum(high - low, 1e-12), 0, 1
        )

    z = A[method + "_coefficients"]
    norms = np.linalg.norm(z, axis=1)
    refidx = np.array(
        [index[i] * 196 + j for i in sel["reference_image_ids"] for j in range(196)]
    )
    active = refidx[norms[refidx] > 0]
    limit = float(np.percentile(norms[active], 99))
    tiles = []
    for j, view in enumerate(c["views"]):
        image_id = view["image_id"]
        path = images / f"{image_id}.png"
        assert sha(path) == assets[image_id]["crop_sha256"]
        im = np.asarray(Image.open(path).convert("RGB"), float) / 255
        off = index[image_id] * 196
        pc = A[method + "_all_pc"][off : off + 196]
        nn = norms[off : off + 196]
        strength = field(nn)
        hue = np.stack(
            [field(rgb(pc)[:, k] * nn) for k in range(3)], axis=-1
        ) / np.maximum(strength[..., None], 1e-12)
        alpha = 0.61 * np.clip(strength / limit, 0, 1) ** 0.65
        overlay = im * (1 - alpha[..., None]) + np.clip(hue, 0, 1) * alpha[..., None]
        patch = view["patches"][0]
        rr, cc = divmod(patch, 14)
        cx = (cc + 0.5) * 16
        cy = (rr + 0.5) * 16
        side = 96
        left = int(np.clip(cx - side / 2, 0, 224 - side))
        up = int(np.clip(cy - side / 2, 0, 224 - side))
        x = x0 + (j % 2) * (tile + gap)
        y = grid_y + (2 - j // 2) * (tile + gap)
        axim = fig.add_axes(box(x, y, tile, tile))
        axim.imshow(
            np.clip(overlay[up : up + side, left : left + side], 0, 1),
            interpolation="lanczos",
        )
        axim.set(xlim=(-0.5, 95.5), ylim=(95.5, -0.5))
        axim.set_xticks([])
        axim.set_yticks([])
        for spine in axim.spines.values():
            spine.set_color(colors[j])
            spine.set_linewidth(0.85)
        for color, lw in [("white", 1.7), (colors[j], 0.65)]:
            axim.add_patch(
                Rectangle(
                    (cc * 16 - left - 0.5, rr * 16 - up - 0.5),
                    16,
                    16,
                    fill=False,
                    edgecolor=color,
                    linewidth=lw,
                )
            )
        axim.text(
            3,
            3,
            str(j + 1),
            fontsize=5.7,
            fontweight="bold",
            color="#243b48",
            va="top",
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.9, pad=0.3),
        )
        tiles.append((axim, colors[j]))
    cloudx = x0 + 2 * tile + 0.11
    cloudw = width - (cloudx - x0) + 0.015
    cloudh = rowheight - 0.37
    ax = fig.add_axes(
        box(cloudx, rowbottom + 0.12, cloudw, cloudh),
        projection="3d",
        computed_zorder=False,
    )
    points = np.vstack([cloud, marked])
    lo = points.min(0)
    hi = points.max(0)
    span = np.maximum(hi - lo, 0.001)
    bounds = np.column_stack([lo - 0.08 * span, hi + 0.08 * span])
    floor = bounds[2, 0]
    plane = [
        [
            (bounds[0, 0], bounds[1, 0], floor),
            (bounds[0, 1], bounds[1, 0], floor),
            (bounds[0, 1], bounds[1, 1], floor),
            (bounds[0, 0], bounds[1, 1], floor),
        ]
    ]
    ax.add_collection3d(
        Poly3DCollection(plane, facecolor="#f1f4f6", edgecolor="none", zorder=0)
    )
    ax.scatter(
        cloud[:, 0],
        cloud[:, 1],
        np.full(len(cloud), floor),
        s=0.45,
        c="#4f5b64",
        alpha=0.055,
        depthshade=False,
        linewidths=0,
        rasterized=True,
        zorder=1,
    )
    ax.scatter(
        *cloud.T,
        c=rgb(cloud),
        s=1.0,
        alpha=0.78,
        depthshade=False,
        linewidths=0,
        rasterized=True,
        zorder=4,
    )
    for j in range(6):
        q = np.array([m["pc"] for m in rec["marked"] if m["view"] == j + 1])
        ax.scatter(
            *q.T,
            c=rgb(q),
            s=10,
            edgecolors=colors[j],
            linewidths=0.55,
            depthshade=False,
            zorder=7,
        )
        ax.scatter(
            *q[0],
            c=rgb(q[:1]),
            s=19,
            edgecolors=colors[j],
            linewidths=0.9,
            depthshade=False,
            zorder=8,
        )
    ax.set(xlim=bounds[0], ylim=bounds[1], zlim=bounds[2])
    ax.view_init(24, -57)
    ax.set_box_aspect(span, zoom=1.12)
    ax.set_axis_off()
    fig.canvas.draw()
    for j, (axim, color) in enumerate(tiles):
        q = next(
            m["pc"] for m in rec["marked"] if m["view"] == j + 1 and m["rank"] == 1
        )
        screen = ax.transData.transform(proj3d.proj_transform(*q, ax.get_proj())[:2])
        end = fig.transFigure.inverted().transform(screen)
        # Two-column image grid, with curved arrows in the gap to the cloud.
        bb = axim.get_position()
        start = (bb.x1 + 0.001, bb.y0 + bb.height * 0.5)
        if j % 2 == 0:
            # Route left-column leaders through the empty gutters, never over another image.
            corridor = bb.x1 + 0.012 / W
            under = bb.y0 - 0.012 / H
            exitx = (x0 + 2 * tile + gap + 0.035) / W
            control = (exitx + (end[0] - exitx) * 0.30, under)
            path = MplPath(
                [
                    start,
                    (corridor, start[1]),
                    (corridor, under),
                    (exitx, under),
                    control,
                    end,
                ],
                [
                    MplPath.MOVETO,
                    MplPath.LINETO,
                    MplPath.LINETO,
                    MplPath.LINETO,
                    MplPath.CURVE3,
                    MplPath.CURVE3,
                ],
            )
            arrow = FancyArrowPatch(
                path=path,
                transform=fig.transFigure,
                arrowstyle="->",
                mutation_scale=4.1,
                color=color,
                alpha=0.74,
                linewidth=0.42,
                zorder=12,
            )
        else:
            arrow = FancyArrowPatch(
                start,
                end,
                transform=fig.transFigure,
                connectionstyle=f"arc3,rad={[0.18, 0.07, 0.09, -0.02, -0.09, -0.12][j]}",
                arrowstyle="->",
                mutation_scale=4.1,
                color=color,
                alpha=0.74,
                linewidth=0.42,
                shrinkA=1,
                shrinkB=3,
                zorder=12,
            )
        fig.add_artist(arrow)
    fig.text(
        (cloudx + cloudw * 0.5) / W,
        (ytop - 0.38) / H,
        "Measured contribution PCA",
        ha="center",
        fontsize=6.9,
        color="#3f5662",
    )
    fig.text(
        x0 / W,
        (rowbottom + 0.055) / H,
        "Theme matches: " + ", ".join(str(n) + "/6" for n in fits),
        fontsize=6.15,
        color="#4c606c",
    )
    fig.text(
        (x0 + width) / W,
        (rowbottom + 0.055) / H,
        f"3-PC variance: {rec['variance_retained']:.1%}",
        ha="right",
        fontsize=6.15,
        color="#4c606c",
    )
    records.append(
        dict(
            method=method,
            displayed_row_height_inches=rowheight,
            reference_cloud_point_count=len(cloud),
            RGB_coordinate_limits=[low.tolist(), high.tolist()],
            image_opacity_norm_limit=limit,
            color_rule="RGB from the same measured first three PCs in image overlays and clouds; colored borders, outlines and arrows denote the annotated marked content.",
            no_new_points_or_surfaces=True,
        )
    )
fig.add_artist(
    plt.Line2D(
        [0.009, 0.99],
        [(BOTTOM + 0.155) / H] * 2,
        transform=fig.transFigure,
        color="#dce3e6",
        linewidth=0.5,
    )
)
for ext in ["pdf", "svg", "png"]:
    kwargs = dict(dpi=500) if ext != "svg" else {}
    if ext == "pdf":
        kwargs["metadata"] = {"CreationDate": None, "ModDate": None}
    fig.savefig(OUT / ("ggf_coherence_group_geometry." + ext), **kwargs)
plt.close(fig)
print(
    "Regenerated the measured response-geometry figure without model inference or training."
)
