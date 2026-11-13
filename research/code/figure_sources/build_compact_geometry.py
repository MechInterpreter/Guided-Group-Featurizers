"""Compact measured image-to-contribution geometry preview. Paper untouched."""

from pathlib import Path
import json, hashlib
import subprocess
import sys
import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter, map_coordinates
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from mpl_toolkits.mplot3d import proj3d
import fitz

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "figures"
OUT.mkdir(exist_ok=True)
S = ROOT / "figure_sources/inputs/figure1_current"
SOURCE_IMAGES = ROOT / "figure_sources/inputs/compact_geometry"
A = np.load(S / "measured_responses.npz")
M = json.loads((S / "checkpoint_projection.json").read_text())
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
assert all(
    sha(S / k) == v
    for k, v in json.loads((S / "INPUT_SHA256.json").read_text()).items()
)
IDS = A["training_image_ids"].tolist()
plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 8,
        "pdf.fonttype": 42,
        "svg.fonttype": "none",
    }
)
W, H = 6.875, 1.80
fig = plt.figure(figsize=(W, H), facecolor="white")


def box(x, y, w, h):
    return [x / W, y / H, w / W, h / H]


def field(v, res=650):
    v = np.asarray(v).reshape(14, 14)
    t = (np.arange(res) + 0.5) * 14 / res - 0.5
    yy, xx = np.meshgrid(t, t, indexing="ij")
    return np.clip(
        map_coordinates(
            gaussian_filter(v, 0.35, mode="nearest"), [yy, xx], order=3, mode="nearest"
        ),
        v.min(),
        v.max(),
    )


def photo(i):
    path = S / "plant_vase.jpg" if i == 553000 else SOURCE_IMAGES / f"{i:012d}.jpg"
    im = Image.open(path).convert("RGB")
    ow, oh = im.size
    scale = 256 / min(ow, oh)
    rw, rh = int(ow * scale), int(oh * scale)
    left, top = int(round((rw - 224) / 2)), int(round((rh - 224) / 2))
    bounds = (
        left * ow / rw,
        top * oh / rh,
        (left + 224) * ow / rw,
        (top + 224) * oh / rh,
    )
    return np.asarray(
        im.transform(
            (650, 650),
            Image.Transform.EXTENT,
            bounds,
            resample=Image.Resampling.BICUBIC,
        ),
        float,
    ) / 255, path


def code(j, i):
    if i == 553000:
        return A[f"{j}_image_code"]
    q = IDS.index(i)
    z = np.zeros((196, 12), np.float32)
    flat = A[f"{j}_active_flat_indices"]
    sel = flat // 196 == q
    z[flat[sel] % 196] = A[f"{j}_all_active_code"][sel]
    return z


records = []
for j, (name, g, images) in enumerate(
    [
        ("Potted plant", 55, [256029, 57979, 261151, 553000]),
        ("Vase", 71, [128831, 256029, 423010, 553000]),
    ]
):
    start = 0.025 + 3.47 * j
    gridx = start
    gy = 0.035
    tile = 0.735
    gap = 0.023
    u = A[f"{j}_cloud_pc"]
    ix = A[f"{j}_plot_indices"]
    T = A[f"{j}_coefficient_to_pc"]
    center = A[f"{j}_pc_center"]
    assert np.allclose(
        A[f"{j}_all_active_code"].astype(float) @ T - center, u, rtol=1e-12, atol=1e-12
    )
    lo, hi = np.percentile(u, [2, 98], axis=0)

    def rgb(v):
        return 0.04 + 0.92 * np.clip((v - lo) / np.maximum(hi - lo, 1e-12), 0, 1)

    norms = np.linalg.norm(A[f"{j}_all_active_code"], axis=1)
    limit = float(np.percentile(norms, 99))
    fig.text(
        (start + 0.75) / W,
        1.675 / H,
        f"{name} ({g})",
        ha="center",
        fontsize=8.5,
        fontweight="bold",
    )
    fig.text(
        (start + 2.49) / W,
        1.675 / H,
        "Contribution geometry",
        ha="center",
        fontsize=8.2,
    )
    image_records = []
    image_axes = []
    for n, i in enumerate(images):
        z = code(j, i)
        pc = z.astype(float) @ T - center
        nn = np.linalg.norm(z, axis=1)
        strength = field(nn)
        colors = rgb(pc)
        hue = np.stack([field(colors[:, k] * nn) for k in range(3)], -1) / np.maximum(
            strength[..., None], 1e-12
        )
        alpha = 0.76 * np.clip(strength / limit, 0, 1) ** 0.65
        base, path = photo(i)
        overlay = base * (1 - alpha[..., None]) + np.clip(hue, 0, 1) * alpha[..., None]
        x = gridx + (n % 2) * (tile + gap)
        y = gy + (1 - n // 2) * (tile + gap)
        ax = fig.add_axes(box(x, y, tile, tile))
        ax.imshow(np.clip(overlay, 0, 1), interpolation="lanczos")
        ax.set_axis_off()
        image_axes.append(ax)
        image_records.append(
            dict(
                image_id=i, source_sha256=sha(path), active_patches=int((nn > 0).sum())
            )
        )
    cloudx = start + 1.64
    ax = fig.add_axes(
        box(cloudx, -0.02, 1.65, 1.65), projection="3d", computed_zorder=False
    )
    p = u[ix]
    cols = rgb(p)
    low = p.min(0)
    high = p.max(0)
    span = np.maximum(high - low, 0.001)
    bounds = np.column_stack([low - 0.08 * span, high + 0.08 * span])
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
        Poly3DCollection(plane, facecolor="#f2f4f6", edgecolor="none", zorder=0)
    )
    ax.scatter(
        p[:, 0],
        p[:, 1],
        np.full(len(p), floor),
        s=0.36,
        c="#60666c",
        alpha=0.065,
        depthshade=False,
        linewidths=0,
        zorder=1,
        rasterized=True,
    )
    ax.scatter(
        *p.T,
        c=cols,
        s=0.70,
        alpha=0.80,
        depthshade=False,
        linewidths=0,
        zorder=4,
        rasterized=True,
    )
    ax.set(xlim=bounds[0], ylim=bounds[1], zlim=bounds[2])
    ax.view_init(24, -57)
    ax.set_box_aspect(span, zoom=1.05)
    ax.set_axis_off()
    # Each arrow locates one measured patch, not an entire image at a single point.
    highlights = []
    for n, i in enumerate(images):
        z = code(j, i)
        q = int(np.argmax(np.linalg.norm(z, axis=1)))
        pc = z[q].astype(float) @ T - center
        ax.scatter(
            *pc,
            color=rgb(pc),
            s=11,
            edgecolor="#172b3b",
            linewidth=0.55,
            depthshade=False,
            zorder=8,
        )
        imax = image_axes[n]
        row, col = divmod(q, 14)
        imax.add_patch(
            Rectangle(
                (col * 650 / 14, row * 650 / 14),
                650 / 14,
                650 / 14,
                fill=False,
                edgecolor="white",
                linewidth=1.25,
            )
        )
        imax.add_patch(
            Rectangle(
                (col * 650 / 14, row * 650 / 14),
                650 / 14,
                650 / 14,
                fill=False,
                edgecolor="#172b3b",
                linewidth=0.55,
            )
        )
        fig.canvas.draw()
        screen = ax.transData.transform(
            np.array(proj3d.proj_transform(*pc, ax.get_proj())[:2])
        )
        end = fig.transFigure.inverted().transform(screen)
        begin = fig.transFigure.inverted().transform(
            imax.transData.transform(((col + 0.5) * 650 / 14, (row + 0.5) * 650 / 14))
        )
        fig.add_artist(
            FancyArrowPatch(
                begin,
                end,
                transform=fig.transFigure,
                connectionstyle=f"arc3,rad={[0.18, 0.08, -0.18, -0.08][n]}",
                arrowstyle="->",
                mutation_scale=5,
                color="#243f50",
                alpha=0.82,
                linewidth=0.50,
                shrinkA=4,
                shrinkB=3,
                zorder=12,
            )
        )
        highlights.append(
            dict(
                image_id=i,
                patch=q,
                pc=pc.tolist(),
                selection="Largest gated coefficient norm within this image",
            )
        )
    assert len(np.unique(np.array([h["pc"] for h in highlights]), axis=0)) == 4
    records.append(
        dict(
            category=name,
            group=g,
            images=image_records,
            point_count=len(ix),
            rgb_train_percentiles=[2, 98],
            rgb_limits=[lo.tolist(), hi.tolist()],
            opacity_norm_limit=limit,
            highlights=highlights,
            variance_retained_pct=100
            * sum(M["groups_displayed"][j]["pca_variance_fractions"]),
        )
    )

fig.canvas.draw()
for text in fig.texts:
    bb = text.get_window_extent(fig.canvas.get_renderer())
    assert (
        bb.x0 >= 0 and bb.x1 <= fig.bbox.x1 and bb.y0 >= 0 and bb.y1 <= fig.bbox.y1
    ), (text.get_text(), bb)
fig.savefig(OUT / "ggf_group_geometry_rgb.pdf", dpi=600)
fig.savefig(OUT / "ggf_group_geometry_rgb.png", dpi=600)
plt.close(fig)
caption = (
    "Image patches mapped to group geometry. The first three principal coordinates of decoded contributions "
    "determine the RGB colors shared by each overlay and its cloud. Each arrow locates one marked patch, not an entire image. "
    "Clouds contain active training patches. Overlays are smoothly interpolated."
)
src = fitz.open(OUT / "ggf_group_geometry_rgb.pdf")
d = fitz.open()
page = d.new_page(width=W * 72, height=H * 72 + 37)
page.show_pdf_page(fitz.Rect(0, 0, W * 72, H * 72), src, 0)
spare = page.insert_textbox(
    fitz.Rect(2, H * 72 + 3, W * 72 - 2, H * 72 + 36),
    caption,
    fontname="tiro",
    fontsize=8.5,
    lineheight=1.07,
)
assert spare >= 0, "Caption overflow"
d.save(OUT / "compact_geometry_with_caption.pdf", garbage=4, deflate=True)
page.get_pixmap(matrix=fitz.Matrix(600 / 72, 600 / 72), alpha=False).save(
    OUT / "compact_geometry_with_caption.png"
)
audit = dict(
    preview_only=True,
    paper_changed=False,
    source_sha256=sha(S / "measured_responses.npz"),
    weights_sha=M["weights_sha"],
    source_geometry="Saved centered PCA of 12D decoded group contributions from fixed 256-image training cohort",
    photograph_selection="Illustrative high-response training images plus the existing plant/vase photograph. Not a comparison of methods.",
    same_rgb_function_for_image_and_cloud=True,
    no_model_training=True,
    figure_inches=[W, H],
    captioned_inches=[W, (H * 72 + 37) / 72],
    groups=records,
)
(OUT / "geometry_preview_audit.json").write_text(json.dumps(audit, indent=2))
print(
    json.dumps(
        {
            "figure_inches": [W, H],
            "captioned_height_inches": (H * 72 + 37) / 72,
            "output": str(OUT / "compact_geometry_with_caption.pdf"),
            "records": [(r["category"], r["point_count"]) for r in records],
        }
    )
)

# Compose the two approved panels. The original panel is preserved as PDF content.
import pymupdf as pdf

if not (OUT / "figure1_current.pdf").is_file():
    subprocess.run(
        [sys.executable, str(Path(__file__).with_name("make_response_figure.py"))],
        check=True,
    )
first = pdf.open(OUT / "figure1_current.pdf")
second = pdf.open(OUT / "ggf_group_geometry_rgb.pdf")
w = first[0].rect.width
sa, sb = 0.82, 0.90
ha = first[0].rect.height * sa
hb = second[0].rect.height * w / second[0].rect.width * sb
combined = pdf.open()
page = combined.new_page(width=w, height=ha + hb + 20)
page.insert_text((1, 8), "(a)", fontname="hebo", fontsize=8)
page.show_pdf_page(pdf.Rect(w * (1 - sa) / 2, 10, w * (1 + sa) / 2, 10 + ha), first, 0)
page.insert_text((1, ha + 18), "(b)", fontname="hebo", fontsize=8)
page.show_pdf_page(
    pdf.Rect(w * (1 - sb) / 2, ha + 20, w * (1 + sb) / 2, ha + 20 + hb), second, 0
)
combined.save(OUT / "figure1_ab.pdf", garbage=4, deflate=True)
