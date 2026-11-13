"""Re-render the measured current GGF figure. Changes display only."""

from pathlib import Path
import hashlib, json
import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter, map_coordinates
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from mpl_toolkits.mplot3d import proj3d

H = Path(__file__).resolve().parent.parent
S = H / "figure_sources/inputs/figure1_current"
O = H / "figures"
O.mkdir(exist_ok=True)
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
manifest = json.loads((S / "INPUT_SHA256.json").read_text())
assert all(sha(S / n) == h for n, h in manifest.items())
meta = json.loads((S / "checkpoint_projection.json").read_text())
a = np.load(S / "measured_responses.npz")
image = S / "plant_vase.jpg"
assert sha(image) == meta["image_input"]["image_sha256"]
geom = meta["image_input"]["geometry"]
ow, oh = geom["original"]
rw, rh = geom["resized"]
l, t, r, b = geom["box"]
photo = (
    np.asarray(
        Image.open(image)
        .convert("RGB")
        .transform(
            (1200, 1200),
            Image.Transform.EXTENT,
            (l * ow / rw, t * oh / rh, r * ow / rw, b * oh / rh),
            resample=Image.Resampling.BICUBIC,
        ),
        float,
    )
    / 255
)


def field(x):
    x = np.asarray(x, float).reshape(14, 14)
    smooth = gaussian_filter(x, 0.35, mode="nearest")
    t = (np.arange(1200) + 0.5) * 14 / 1200 - 0.5
    yy, xx = np.meshgrid(t, t, indexing="ij")
    return np.clip(
        map_coordinates(smooth, [yy, xx], order=3, mode="nearest"), x.min(), x.max()
    )


norms = [np.linalg.norm(a[f"{j}_image_code"], axis=1) for j in range(2)]
scales = [float(np.percentile(n[n > 0], 99)) for n in norms]
palettes = [["#253bb8", "#00aee9", "#04efd0"], ["#6918b5", "#f01bc9", "#ff9e68"]]
cmaps = [
    LinearSegmentedColormap.from_list("group" + str(j), p, N=1024)
    for j, p in enumerate(palettes)
]
colors = np.array([[0.015, 0.94, 0.82], [0.96, 0.045, 0.78]])
fields = np.stack([np.clip(field(n) / v, 0, 1) for n, v in zip(norms, scales)], axis=-1)
mix = fields @ colors / np.maximum(fields.sum(-1, keepdims=True), 1e-12)
alpha = 0.86 * fields.max(-1) ** 1.10
joint = photo * (1 - alpha[..., None]) + mix * alpha[..., None]
plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 8,
        "pdf.fonttype": 42,
        "svg.fonttype": "none",
    }
)
fig = plt.figure(figsize=(6.875, 2.60), facecolor="white")


def img(im, box):
    ax = fig.add_axes(box)
    ax.imshow(im, interpolation="lanczos", aspect="auto")
    ax.axis("off")


def cloud(points, cols, box):
    ax = fig.add_axes(box, projection="3d", computed_zorder=False)
    half = np.maximum(np.ptp(points, axis=0) / 2, 0.001)
    center = (points.min(0) + points.max(0)) / 2
    bounds = np.stack([center - half * 1.07, center + half * 1.07], axis=1)
    floor = bounds[2, 0]
    v = [
        [
            (bounds[0, 0], bounds[1, 0], floor),
            (bounds[0, 1], bounds[1, 0], floor),
            (bounds[0, 1], bounds[1, 1], floor),
            (bounds[0, 0], bounds[1, 1], floor),
        ]
    ]
    ax.add_collection3d(
        Poly3DCollection(
            v,
            facecolor="#f2f5f6",
            edgecolor="#cbd5da",
            linewidth=0.25,
            alpha=0.9,
            zorder=0,
        )
    )
    hist, ex, ey = np.histogram2d(
        points[:, 0], points[:, 1], bins=32, range=[bounds[0], bounds[1]]
    )
    hist = gaussian_filter(hist, 0.8)
    xx, yy = np.meshgrid((ex[:-1] + ex[1:]) / 2, (ey[:-1] + ey[1:]) / 2, indexing="ij")
    rgba = np.ones((32, 32, 4)) * 0.2
    rgba[..., 3] = 0.25 * hist / max(hist.max(), 1e-12)
    ax.plot_surface(
        xx,
        yy,
        np.full_like(xx, floor + 0.001),
        facecolors=rgba,
        rstride=1,
        cstride=1,
        shade=False,
        antialiased=False,
        zorder=1,
    )
    ax.scatter(
        *points.T,
        c=cols,
        s=0.64,
        alpha=0.87,
        linewidths=0,
        depthshade=False,
        zorder=5,
        rasterized=True,
    )
    ax.set(xlim=bounds[0], ylim=bounds[1], zlim=bounds[2])
    ax.set_axis_off()
    ax.view_init(28, -57)
    aspect = np.ptp(bounds, axis=1)
    zoom = 1.12
    for _ in range(20):
        ax.set_box_aspect(aspect, zoom=zoom)
        fig.canvas.draw()
        x, y, _ = proj3d.proj_transform(*points.T, ax.get_proj())
        screen = ax.transData.transform(np.column_stack([x, y]))
        bb = ax.bbox
        pad = 1.5
        if (
            screen[:, 0].min() >= bb.x0 + pad
            and screen[:, 0].max() <= bb.x1 - pad
            and screen[:, 1].min() >= bb.y0 + pad
            and screen[:, 1].max() <= bb.y1 - pad
        ):
            break
        zoom *= 0.94
    else:
        raise ValueError("Measured point falls outside cloud panel")
    return zoom


records = []
saved = {}
overlays = []
for j, name in enumerate(["Potted plant", "Vase"]):
    u = a[f"{j}_cloud_pc"]
    iu = a[f"{j}_image_pc"]
    ix = a[f"{j}_plot_indices"]
    lo, hi = np.percentile(u[:, 0], [2, 98])

    def color(v):
        return cmaps[j](np.clip((v[:, 0] - lo) / max(hi - lo, 1e-12), 0, 1))[..., :3]

    col = color(u)
    icol = color(iu)
    code = a[f"{j}_image_code"]
    recomputed = code.astype(float) @ a[f"{j}_coefficient_to_pc"] - a[f"{j}_pc_center"]
    assert np.allclose(recomputed, iu, rtol=1e-12, atol=1e-12)
    assert np.allclose(color(recomputed), icol, rtol=1e-12, atol=1e-12)
    hue = np.stack(
        [field(icol[:, k] * norms[j]) for k in range(3)], axis=-1
    ) / np.maximum(field(norms[j])[..., None], 1e-12)
    alpha = 0.88 * fields[..., j] ** 0.90
    overlay = photo * (1 - alpha[..., None]) + np.clip(hue, 0, 1) * alpha[..., None]
    overlays.append(overlay)
    top = 0.54 if j == 0 else 0.075
    z = cloud(u[ix], col[ix], [0.005, top - 0.03, 0.19, 0.37])
    fig.text(
        0.10, top + 0.35, name, ha="center", fontsize=8, color=["#087a85", "#ae148e"][j]
    )
    fraction = 100 * sum(meta["groups_displayed"][j]["pca_variance_fractions"])
    fig.text(
        0.10,
        top - 0.03,
        f"3 PCs: {fraction:.1f}% variance",
        ha="center",
        fontsize=8,
        color="#4b5b65",
    )
    val = np.abs(code[:, 0])
    vscale = float(np.percentile(val[val > 0], 99))
    strength = np.clip(field(val) / max(vscale, 1e-12), 0, 1)
    rgb = plt.get_cmap("Reds")(0.1 + 0.9 * strength)[..., :3]
    alpha = 0.94 * strength**0.65
    bg = 0.19 * photo + 0.81
    img(bg * (1 - alpha[..., None]) + rgb * alpha[..., None], [0.215, top, 0.135, 0.34])
    fig.text(
        0.2825, top + 0.35, "Direction 1", ha="center", fontsize=8, color="#304e60"
    )
    fig.add_artist(
        FancyArrowPatch(
            (0.212, top + 0.20),
            (0.191, top + 0.20),
            transform=fig.transFigure,
            connectionstyle="arc3,rad=.14",
            arrowstyle="->",
            mutation_scale=7,
            color=["#087a85", "#ae148e"][j],
            linewidth=0.8,
        )
    )
    records.append(
        dict(
            category=name,
            group=meta["groups_displayed"][j]["group"],
            activity_display_scale=scales[j],
            direction1_display_scale=vscale,
            PC1_color_limits=[float(lo), float(hi)],
            palette=palettes[j],
            point_count=len(ix),
            display_zoom=z,
            variance_retained_pct=fraction,
        )
    )
    saved.update({f"{j}_cloud_rgb": col, f"{j}_image_rgb": icol})
img(joint, [0.370, 0.075, 0.325, 0.805])
fig.text(0.5325, 0.895, "GGF group activity", ha="center", fontsize=8.5)
img(photo, [0.705, 0.540, 0.140, 0.34])
img(overlays[0], [0.855, 0.540, 0.140, 0.34])
img(overlays[1], [0.705, 0.075, 0.140, 0.34])
difference = fields[..., 0] - fields[..., 1]
diffcolors = np.where((difference >= 0)[..., None], colors[0], colors[1])
alpha = 0.86 * np.abs(difference) ** 0.65
img(
    photo * (1 - alpha[..., None]) + diffcolors * alpha[..., None],
    [0.855, 0.075, 0.140, 0.34],
)
for x, y, label in [
    (0.775, 0.895, "Input crop"),
    (0.925, 0.895, "Plant: PC1"),
    (0.775, 0.43, "Vase: PC1"),
    (0.925, 0.43, "Plant minus vase"),
]:
    fig.text(x, y, label, ha="center", fontsize=8)
for artist in fig.texts:
    if artist.get_fontsize() < 9.2:
        artist.set_fontsize(9.2)
for ext in ["pdf", "png", "svg"]:
    fig.savefig(
        O / ("figure1_current." + ext), dpi=600, bbox_inches="tight", pad_inches=0.02
    )
plt.close(fig)
np.savez_compressed(O / "figure1_colors.npz", **saved)
audit = dict(
    source_weights_sha=meta["weights_sha"],
    source_run=meta["pilot_run"],
    source_measured_data_sha=sha(S / "measured_responses.npz"),
    measured_codes_and_coordinates_unchanged=True,
    points_not_added_removed_or_shifted=True,
    plotted_indices_unchanged=True,
    color_mapping="First centered contribution principal coordinate, 2nd–98th active TRAIN percentile limits, clipped for display. Identical function for cloud and image within each group.",
    activity_scale="99th percentile of nonzero group norms in the fixed displayed image, separately for each group. Joint and individual panels share these scales.",
    opacity="0.88 * normalized activity^0.90 in individual overlays; 0.86 * max normalized activity^1.10 in categorical cyan/magenta joint panel.",
    source_image_sha=sha(image),
    groups=records,
    display_interpolation="Gaussian sigma0.35 patches, cubic interpolation at patch centers, clipped to measured scalar range.",
    scope="Display colors and scales only; no retraining, readout fitting, test selection, geometry smoothing or changed quantitative results.",
)
(O / "display_audit.json").write_text(json.dumps(audit, indent=2))
print(json.dumps(audit, indent=2))
