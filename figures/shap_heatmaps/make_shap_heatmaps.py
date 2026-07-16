"""
Build directional SHAP interaction heatmaps for three genomic contexts:
N315 core genome (capsule region), the N315 plasmid, and the mecA + ACME
cassette from JE2. Each input workbook holds a square gene x gene matrix of
directional SHAP interaction values plus a "Group" column that tags genes
belonging to a named functional module. Genes belonging to the same module
are contiguous in the row/column order, so each module's intra-module
interaction block is a single square sitting on the diagonal.

Usage: python make_shap_heatmaps.py
"""
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).parent
DATA = HERE / "data"
OUT = HERE / "output"
OUT.mkdir(exist_ok=True)

VLIM = 0.5  # heatmap color scale: -0.5 to 0.5
CMAP = plt.get_cmap("RdBu").copy()  # RdBu (not reversed): red=negative, blue=positive
CMAP.set_bad("#e4e3dc")  # self-pairs / missing values

# Categorical outline colors: dark, highly-saturated hues chosen to stand out
# hard against the pale red/blue heatmap fill. Blue and red are avoided
# entirely so an outline is never mistaken for the value encoding.
GROUP_HUES = [
    "#1b7837",  # dark green
    "#762a83",  # dark purple
    "#e08214",  # vivid orange
    "#000000",  # black
    "#c51b7d",  # vivid magenta
    "#01665e",  # dark teal
    "#8c510a",  # brown
    "#b8860b",  # dark goldenrod
]
LINESTYLES = ["-", "--", ":"]
OUTLINE_WIDTH = 3.2

PANELS = [
    dict(
        key="N315_capsules",
        file=DATA / "N315_capsules.xlsx",
        sheet="selected_gene_directional_shap",
        title="N315 core genome — capsule region",
        label="A",
    ),
    dict(
        key="N315_plasmid",
        file=DATA / "N315_plasmid.xlsx",
        sheet="selected_gene_directional_shap_",
        title="N315 plasmid",
        label="B",
    ),
    dict(
        key="JE2_mecA_ACME",
        file=DATA / "JE2_mecA_ACME.xlsx",
        sheet="selected_gene_directional_shap",
        title="mecA + ACME cassette (JE2)",
        label="C",
    ),
]

STOP_SYMBOLS = {
    "protein", "transporter", "family", "system", "subunit", "domain",
    "component", "module", "kinase", "reductase", "transposase",
    "dehydrogenase", "synthase", "ATPase", "putative", "repressor",
    "regulator", "B", "A",
}


def short_label(row_gene_name: str) -> str:
    """Locus-tag suffix, plus a trailing gene symbol when the annotation ends
    with one (e.g. 'PhnE', 'Cap8B', 'MecA')."""
    locus, _, desc = row_gene_name.partition(" ")
    desc = desc.strip().rstrip(".")
    last = desc.split()[-1] if desc else ""
    suffix = locus.split("_")[-1]
    if (
        re.fullmatch(r"[A-Z][A-Za-z0-9]{1,9}", last)
        and last not in STOP_SYMBOLS
        and any(c.islower() for c in last)
    ):
        return f"{suffix} {last}"
    return suffix


def group_blocks(groups: list) -> list:
    """Contiguous (start, end, name) index ranges for each named group."""
    blocks = []
    start = None
    current = None
    for i, g in enumerate(groups + [None]):
        if g != current:
            if current is not None:
                blocks.append((start, i - 1, current))
            if pd.notna(g):
                start = i
                current = g
            else:
                current = None
    return blocks


def load_panel(spec):
    df = pd.read_excel(spec["file"], sheet_name=spec["sheet"])
    labels = [short_label(n) for n in df["row_gene_name"]]
    matrix = df.iloc[:, 3:].to_numpy(dtype=float)
    blocks = group_blocks(df["Group"].tolist())
    return labels, matrix, blocks


def draw_panel(ax, matrix, blocks, title, panel_label, color_map):
    n = matrix.shape[0]
    im = ax.imshow(
        np.ma.masked_invalid(matrix),
        cmap=CMAP,
        vmin=-VLIM,
        vmax=VLIM,
        interpolation="nearest",
    )

    # Gene-level tick labels aren't legible at this scale; drop them and
    # just frame the axes so the matrix extent is still clear.
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("#c3c2b7")
        spine.set_linewidth(1)

    ax.set_xlim(-0.5, n - 0.5)
    ax.set_ylim(n - 0.5, -0.5)

    # Outline each module's self-interaction block on the diagonal.
    for depth, (start, end, name) in enumerate(blocks):
        color, ls = color_map[name]
        size = end - start + 1
        rect = Rectangle(
            (start - 0.5, start - 0.5), size, size,
            fill=False, edgecolor=color, linestyle=ls,
            linewidth=OUTLINE_WIDTH, zorder=6 + depth,
        )
        ax.add_patch(rect)

    ax.set_title(f"{panel_label}.  {title}", fontsize=22, fontweight="bold",
                 loc="left", pad=14)
    ax.set_xlabel("genes (genomic order) →", fontsize=14, color="#52514e")
    ax.set_ylabel("genes (genomic order) →", fontsize=14, color="#52514e")
    return im


def draw_legend(ax, blocks, color_map, ncol=1):
    ax.axis("off")
    handles = [
        Line2D([0], [0], color=color_map[name][0], linestyle=color_map[name][1],
               linewidth=OUTLINE_WIDTH)
        for _, _, name in blocks
    ]
    names = [name for _, _, name in blocks]
    if not handles:
        return
    ax.legend(
        handles, names,
        loc="upper center",
        frameon=False,
        fontsize=15,
        title="Functional module (diagonal block outline)",
        title_fontsize=16,
        handlelength=2.6,
        labelspacing=1.0,
        columnspacing=2.2,
        ncol=ncol,
        borderaxespad=0,
    )


def assign_colors(blocks):
    color_map = {}
    for i, (_, _, name) in enumerate(blocks):
        hue = GROUP_HUES[i % len(GROUP_HUES)]
        ls = LINESTYLES[i // len(GROUP_HUES)]
        color_map[name] = (hue, ls)
    return color_map


def main():
    panel_data = []
    for spec in PANELS:
        labels, matrix, blocks = load_panel(spec)
        color_map = assign_colors(blocks)
        panel_data.append((spec, labels, matrix, blocks, color_map))

    # Render each panel individually at full resolution, then stitch them
    # into one combined figure. Doing the composite in pixel space (rather
    # than one shared matplotlib gridspec) avoids the empty-space blowup
    # that aspect-equal panels of very different gene counts produce when
    # forced to share row-height ratios.
    panel_pngs = []
    for spec, labels, matrix, blocks, color_map in panel_data:
        n_groups = len(blocks)
        ncol = 2 if n_groups > 4 else 1
        legend_rows = -(-n_groups // ncol)
        legend_h = 0.5 * legend_rows + 0.5
        cbar_h = 1.3

        pf = plt.figure(figsize=(10, 10 + legend_h + cbar_h))
        pgs = pf.add_gridspec(
            nrows=3, ncols=1, height_ratios=[10, legend_h, cbar_h], hspace=0.05,
            left=0.06, right=0.97, top=0.95, bottom=0.02,
        )
        pax = pf.add_subplot(pgs[0, 0])
        pax.set_aspect("equal")
        im = draw_panel(pax, matrix, blocks, spec["title"], spec["label"], color_map)

        lax = pf.add_subplot(pgs[1, 0])
        draw_legend(lax, blocks, color_map, ncol=ncol)

        bax = pf.add_subplot(pgs[2, 0])
        bax.axis("off")
        cax = bax.inset_axes([0.32, 0.55, 0.36, 0.28])
        pcbar = pf.colorbar(im, cax=cax, orientation="horizontal", extend="both")
        pcbar.set_label("Directional SHAP interaction value", fontsize=15)
        pcbar.ax.tick_params(labelsize=13)

        png_path = OUT / f"shap_heatmap_{spec['key']}.png"
        pf.savefig(png_path, dpi=300, bbox_inches="tight", pad_inches=0.15)
        pf.savefig(OUT / f"shap_heatmap_{spec['key']}.pdf", bbox_inches="tight", pad_inches=0.15)
        plt.close(pf)
        panel_pngs.append(png_path)

    stitch_panels(panel_pngs, OUT / "shap_interaction_heatmaps.png")
    print("Saved figures to", OUT)


def stitch_panels(png_paths, out_path):
    """Stack the individually-rendered panels into one combined figure."""
    images = [Image.open(p) for p in png_paths]
    target_w = max(im.width for im in images)
    gap = 40
    title_h = 220

    resized = []
    for im in images:
        if im.width != target_w:
            scale = target_w / im.width
            im = im.resize((target_w, round(im.height * scale)), Image.LANCZOS)
        resized.append(im)

    total_h = title_h + sum(im.height for im in resized) + gap * (len(resized) - 1) + 20
    canvas = Image.new("RGB", (target_w, total_h), "white")
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 130
        )
    except OSError:
        font = ImageFont.load_default()
    draw.text((30, 40), "Directional SHAP gene–gene interactions", fill="black", font=font)

    y = title_h
    for im in resized:
        canvas.paste(im, (0, y))
        y += im.height + gap

    canvas.save(out_path)


if __name__ == "__main__":
    main()
