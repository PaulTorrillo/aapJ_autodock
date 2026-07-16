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

VLIM = 0.3  # heatmap color scale: -0.3 to 0.3
CMAP = plt.get_cmap("RdBu_r").copy()
CMAP.set_bad("#e4e3dc")  # self-pairs / missing values

# Categorical outline colors (validated CVD-safe set), reserving blue/red
# for the heatmap's own diverging fill so outlines never fight the data.
GROUP_HUES = ["#008300", "#e87ba4", "#eda100", "#1baf7a", "#eb6834", "#4a3aa7"]
LINESTYLES = ["-", "--", ":"]

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


def draw_panel(ax, labels, matrix, blocks, title, panel_label, color_map):
    n = len(labels)
    im = ax.imshow(
        np.ma.masked_invalid(matrix),
        cmap=CMAP,
        vmin=-VLIM,
        vmax=VLIM,
        interpolation="nearest",
    )

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(labels, rotation=90, fontsize=5.2, family="monospace")
    ax.set_yticklabels(labels, fontsize=5.2, family="monospace")
    ax.tick_params(length=2, pad=1.5)
    for spine in ax.spines.values():
        spine.set_visible(False)

    ax.set_xlim(-0.5, n - 0.5)
    ax.set_ylim(n - 0.5, -0.5)

    for start, end, name in blocks:
        color, ls = color_map[name]
        size = end - start + 1
        rect = Rectangle(
            (start - 0.5, start - 0.5),
            size,
            size,
            fill=False,
            edgecolor=color,
            linestyle=ls,
            linewidth=1.6,
            zorder=5,
        )
        ax.add_patch(rect)

    ax.set_title(f"{panel_label}.  {title}", fontsize=11, fontweight="bold",
                 loc="left", pad=8)
    ax.set_xlabel("gene (SHAP feature)", fontsize=7, color="#52514e")
    ax.set_ylabel("gene (SHAP feature)", fontsize=7, color="#52514e")
    return im


def draw_legend(ax, blocks, color_map):
    ax.axis("off")
    handles = [
        Line2D([0], [0], color=color_map[name][0], linestyle=color_map[name][1],
               linewidth=2.2)
        for _, _, name in blocks
    ]
    names = [name for _, _, name in blocks]
    if not handles:
        return
    ax.legend(
        handles, names,
        loc="upper left",
        frameon=False,
        fontsize=7.5,
        title="Functional module\n(diagonal block outline)",
        title_fontsize=7.5,
        handlelength=2.2,
        labelspacing=0.9,
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
        n = len(labels)
        pf = plt.figure(figsize=(0.24 * n + 3.2, 0.24 * n + 1.8))
        pgs = pf.add_gridspec(1, 2, width_ratios=[4, 1.1], wspace=0.05,
                               left=0.10, right=0.86, top=0.93, bottom=0.16)
        pax = pf.add_subplot(pgs[0, 0])
        pax.set_aspect("equal")
        im = draw_panel(pax, labels, matrix, blocks, spec["title"], spec["label"], color_map)
        plax = pf.add_subplot(pgs[0, 1])
        draw_legend(plax, blocks, color_map)
        pcax = pf.add_axes([0.89, 0.25, 0.02, 0.5])
        pcbar = pf.colorbar(im, cax=pcax, extend="both")
        pcbar.set_label("Directional SHAP\ninteraction value", fontsize=8)
        pcbar.ax.tick_params(labelsize=7)
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
    title_h = 110

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
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 46
        )
    except OSError:
        font = ImageFont.load_default()
    draw.text((20, 30), "Directional SHAP gene–gene interactions", fill="black", font=font)

    y = title_h
    for im in resized:
        canvas.paste(im, (0, y))
        y += im.height + gap

    canvas.save(out_path)


if __name__ == "__main__":
    main()
