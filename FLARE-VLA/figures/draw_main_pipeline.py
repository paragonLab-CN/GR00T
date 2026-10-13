#!/usr/bin/env python3
"""Draw the FLARE-VLA paper overview as editable vector art."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Polygon, Rectangle


OUT = Path(__file__).resolve().parent
OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
})

fig, ax = plt.subplots(figsize=(18, 9), dpi=180)
fig.patch.set_facecolor("white")
ax.set_xlim(0, 1800)
ax.set_ylim(900, 0)
ax.axis("off")
fig.subplots_adjust(left=0, right=1, top=1, bottom=0)

C = {
    "navy": "#17263F",
    "ink": "#243651",
    "muted": "#65748A",
    "line": "#B7C3D2",
    "panel": "#F7F9FC",
    "teal": "#0D9A91",
    "teal_bg": "#EAF8F5",
    "violet": "#7255C6",
    "violet_bg": "#F2EEFC",
    "blue": "#3B75C5",
    "blue_bg": "#EBF3FC",
    "amber": "#C8851D",
    "amber_bg": "#FFF5E4",
    "green": "#448A5D",
    "green_bg": "#EAF5EE",
    "white": "#FFFFFF",
}


def box(x, y, w, h, *, fc="white", ec="#D5DDE7", lw=1.5, r=15, z=1):
    p = FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f"round,pad=0,rounding_size={r}",
        linewidth=lw, edgecolor=C.get(ec, ec), facecolor=C.get(fc, fc), zorder=z,
    )
    ax.add_patch(p)
    return p


def label(x, y, s, *, size=16, color="ink", weight="regular", ha="left", va="center", z=8, **kw):
    return ax.text(x, y, s, fontsize=size, color=C.get(color, color),
                   fontweight=weight, ha=ha, va=va, zorder=z, **kw)


def arrow(x1, y1, x2, y2, *, color="line", lw=2.5, head=13, z=5, style="-"):
    a = FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                        mutation_scale=head, linewidth=lw,
                        linestyle=style, color=C.get(color, color), zorder=z,
                        shrinkA=0, shrinkB=0, connectionstyle="arc3")
    ax.add_patch(a)


def route(points, *, color="line", lw=2, head=12, style="-", z=3):
    xs, ys = zip(*points)
    ax.plot(xs, ys, color=C.get(color, color), linewidth=lw,
            linestyle=style, solid_capstyle="round", zorder=z)
    arrow(*points[-2], *points[-1], color=color, lw=lw, head=head, z=z + 0.1, style=style)


def pill(x, y, w, h, s, *, fc, color, size=12):
    box(x, y, w, h, fc=C.get(fc, fc), ec="none", lw=0, r=h / 2, z=5)
    label(x + w / 2, y + h / 2, s, size=size, color=color, weight="bold", ha="center", z=6)


def heading(x, y, n, title, subtitle, accent):
    pill(x, y, 34, 34, n, fc=C[accent + "_bg"], color=accent, size=13)
    label(x + 46, y + 8, title, size=17, weight="bold", color="navy")
    label(x + 46, y + 32, subtitle, size=10.8, color="muted")


def latent_bar(x, y, colors, *, w=15, h=21, gap=4, alpha=1):
    for i, col in enumerate(colors):
        box(x + i * (w + gap), y, w, h, fc=col, ec="none", lw=0, r=3, z=5)


# Two stacked, softly bounded phases.
box(36, 55, 1728, 534, fc="panel", ec="#E5EAF1", lw=1.4, r=23, z=0)
box(36, 628, 1728, 224, fc="panel", ec="#E5EAF1", lw=1.4, r=23, z=0)
pill(65, 73, 293, 35, "A  ·  OFFLINE TEACHER DISCOVERY", fc="blue_bg", color="blue", size=10)
pill(65, 646, 226, 35, "B  ·  POLICY REFINEMENT", fc="green_bg", color="green", size=10)

# A1 — source demonstration, with a tiny original vector scene.
box(68, 204, 228, 257, fc="white", ec="#C9D6E5", r=17)
heading(85, 223, "1", "Offline data", "task trajectories", "blue")
box(89, 289, 185, 104, fc="#F0F5F8", ec="#DCE5EB", lw=1, r=8)
ax.add_patch(Rectangle((92, 357), 179, 34, facecolor="#D4E3E8", edgecolor="none", zorder=2))
ax.add_patch(Rectangle((117, 341), 105, 13, facecolor="#A9C8CC", edgecolor="none", zorder=3))
ax.add_patch(Rectangle((202, 317), 25, 23, facecolor="#E5AB4C", edgecolor="#A96D1A", linewidth=1, zorder=4))
ax.add_patch(Circle((139, 324), 17, facecolor="#596B82", edgecolor="#31455B", linewidth=1, zorder=4))
ax.plot([245, 221, 207, 196], [300, 309, 325, 328], color="#58728A", linewidth=8, solid_capstyle="round", zorder=4)
ax.plot([195, 195], [327, 340], color="#344E65", linewidth=5, zorder=4)
ax.plot([188, 202], [341, 341], color="#344E65", linewidth=4, zorder=4)
label(92, 411, "image · language · state", size=11.5, color="ink", weight="bold")
label(92, 435, "GT actions · future frames", size=10.2, color="muted")

# A2 — candidate policy and actual K=4 sampled chunks.
box(364, 204, 253, 257, fc="white", ec="#A9DCD6", r=17)
heading(381, 223, "2", "GR00T", "sample action chunks", "teal")
for j, yy in enumerate([302, 331, 360, 389]):
    pill(388, yy - 9, 35, 20, f"a{j+1}", fc="teal_bg", color="teal", size=10)
    xx = [443, 469, 495, 523, 552, 580]
    offsets = ([0, -6, -1, -10, -5, -13], [0, 5, -2, 4, -6, -4],
               [0, -5, 7, 1, 9, 3], [0, 3, 9, -1, 7, -7])[j]
    ax.plot(xx, [yy + v for v in offsets], color=C["teal"], linewidth=2.2,
            marker="o", markersize=3.5, zorder=5, alpha=0.95)
label(489, 431, "K = 4 candidates · H = 8", size=10.4, color="muted", ha="center")

# A3 — predictive rollout through LeWM.
box(701, 204, 278, 257, fc="white", ec="#CFC1ED", r=17)
heading(718, 223, "3", "LeWM", "latent future rollout", "violet")
pill(724, 298, 232, 31, "history + candidate action", fc="violet_bg", color="violet", size=10.3)
for i, yy in enumerate([341, 363, 385, 407]):
    latent_bar(744, yy, ["#D9CFF2", "#BFADE6", "#A58AD9", "#8C6BCC", "#7354B7", "#5F469D"], w=24, h=14, gap=9)
    label(726, yy + 7, str(i + 1), size=10, color="violet", weight="bold")
label(840, 442, "predicted future latents", size=10.8, color="muted", ha="center")

# A4 — future-latent comparison.
box(1071, 204, 271, 257, fc="white", ec="#B8CEF0", r=17)
heading(1088, 223, "4", "Future score", "compare with GT future", "blue")
label(1093, 304, "predicted", size=11.5, color="muted")
latent_bar(1181, 292, ["#D6C8F1", "#AB95DD", "#8366C9", "#624BA7", "#A58ED8"], w=24, h=21, gap=5)
label(1093, 348, "observed", size=11.5, color="muted")
latent_bar(1181, 336, ["#D0DEF4", "#9FBDE8", "#6F9BD9", "#4E7FC7", "#8FB0E0"], w=24, h=21, gap=5)
pill(1090, 390, 231, 32, "mean latent MSE over H", fc="blue_bg", color="blue", size=10.2)
label(1206, 439, "rank by future match", size=10.4, color="muted", ha="center")

# A5 — quality gate and output teacher action.
box(1430, 204, 300, 257, fc="white", ec="#EACD98", r=17)
heading(1447, 223, "5", "Select teacher", "quality + budget filtering", "amber")
for i, (yy, width) in enumerate([(303, 145), (331, 68), (359, 126), (387, 165)]):
    good = i == 1
    label(1454, yy + 8, f"a{i+1}", size=11.5, color="amber" if good else "muted",
          weight="bold" if good else "regular")
    box(1485, yy, 187, 16, fc="#EFF2F6", ec="none", lw=0, r=8)
    box(1485, yy, width, 16, fc=C["amber"] if good else "#AFC0D0", ec="none", lw=0, r=8)
    if good:
        label(1696, yy + 8, "✓", size=20, color="amber", weight="bold", ha="center")
label(1580, 433, "agreement · oracle · action checks", size=10.0, color="muted", ha="center")

# Main directional connections.
arrow(296, 331, 364, 331, color="teal", lw=2.7, head=16)
label(330, 311, "oₜ, ℓ", size=11.0, color="teal", ha="center")
arrow(617, 331, 701, 331, color="violet", lw=2.7, head=16)
label(659, 311, "K actions", size=11.4, color="violet", ha="center")
arrow(979, 331, 1071, 331, color="violet", lw=2.7, head=16)
label(1025, 310, "K futures", size=11.4, color="violet", ha="center")
arrow(1342, 331, 1430, 331, color="amber", lw=2.7, head=16)
label(1386, 310, "latent error", size=10.5, color="amber", ha="center")

# The two data branches that make this an offline method.
route([(181, 204), (181, 151), (700, 151)], color="blue", lw=2.3, head=13, z=2)
pill(705, 133, 165, 36, "Encode GT future", fc="blue_bg", color="blue", size=10.4)
route([(870, 151), (1204, 151), (1204, 204)], color="blue", lw=2.3, head=13, z=2)
label(457, 132, "observed future frames", size=10.5, color="blue", ha="center")
label(1036, 132, "GT future latents  zᴳᵀ", size=10.5, color="blue", ha="center")

route([(182, 461), (182, 513), (838, 513), (838, 461)],
      color="violet", lw=2, head=12, style=(0, (5, 4)), z=2)
label(513, 495, "encoded history + recent actions", size=10.2, color="violet", ha="center")

# B — keep original supervision and add selected teachers.
box(83, 710, 302, 109, fc="white", ec="#C9D6E5", r=15)
pill(101, 727, 49, 25, "GT", fc="blue_bg", color="blue", size=11)
label(161, 740, "GT demonstrations", size=14.0, color="navy", weight="bold")
label(101, 780, "observations + expert actions", size=10.8, color="muted")

box(493, 710, 326, 109, fc="white", ec="#EACD98", r=15)
pill(511, 727, 77, 25, "FLARE", fc="amber_bg", color="amber", size=11)
label(601, 740, "FLARE teachers", size=14.0, color="navy", weight="bold")
label(511, 780, "same observations + new actions", size=10.8, color="muted")

box(925, 710, 270, 109, fc="white", ec="#B7DEC2", r=15)
pill(943, 727, 58, 25, "MIX", fc="green_bg", color="green", size=11)
label(1015, 740, "GT + teacher", size=14.0, color="navy", weight="bold")
label(943, 780, "GT retained; teacher added", size=10.8, color="muted")

box(1300, 710, 422, 109, fc="white", ec="#A9DCD6", r=15)
pill(1318, 727, 78, 25, "TRAIN", fc="teal_bg", color="teal", size=11)
label(1410, 740, "Fine-tune GR00T", size=14.0, color="navy", weight="bold")
label(1318, 780, "updated vision-language-action policy", size=10.8, color="muted")

route([(236, 819), (236, 839), (1059, 839), (1059, 819)],
      color="blue", lw=2.2, head=12, z=2)
arrow(819, 765, 925, 765, color="amber", lw=2.5, head=14)
arrow(1195, 765, 1300, 765, color="green", lw=2.5, head=14)

# The teacher hand-off joins the two panels; source color identifies GT.
route([(1578, 461), (1578, 608), (656, 608), (656, 710)],
      color="amber", lw=2.4, head=13, z=3)
label(1150, 598, "accepted teacher actions", size=10.4, color="amber", ha="center")

for ext in ("svg", "pdf", "png"):
    fig.savefig(OUT / f"flare_vla_main_pipeline.{ext}", dpi=220,
                facecolor="white", bbox_inches="tight", pad_inches=0.03)
plt.close(fig)
