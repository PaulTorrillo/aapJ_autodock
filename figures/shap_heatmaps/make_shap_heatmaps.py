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

import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle
from PIL import Image, ImageDraw, ImageFont
from scipy import stats

HERE = Path(__file__).parent
DATA = HERE / "data"
OUT = HERE / "output"
OUT.mkdir(exist_ok=True)

VLIM = 0.5  # heatmap color scale: -0.5 to 0.5
CMAP = plt.get_cmap("RdBu").copy()  # RdBu (not reversed): red=negative, blue=positive
CMAP.set_bad("#e4e3dc")  # self-pairs / missing values

OUTLINE_COLOR = "black"
OUTLINE_WIDTH = 2.6

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

COOCCURRENCE_FILE = DATA / "mecA_cooccurrence.xlsx"
MIN_GENOMES_PER_CELL = 100  # min count required in each of the 4 present/absent combos
COLOR_FIT = "#2a6fb0"        # points used in the regression
COLOR_EXCLUDED = "#a8a6a0"   # points below the genome-count threshold
COLOR_LINE = "black"

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
    # Excel rows are the input gene, columns the output gene; transpose so
    # the plotted x-axis is input and y-axis is output.
    matrix = df.iloc[:, 3:].to_numpy(dtype=float).T
    blocks = group_blocks(df["Group"].tolist())
    return labels, matrix, blocks


def draw_panel(ax, matrix, blocks, title, panel_label):
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

    # Outline each module's self-interaction block on the diagonal, numbered
    # in the order the modules appear along the genome.
    for depth, (start, end, name) in enumerate(blocks):
        number = depth + 1
        size = end - start + 1
        rect = Rectangle(
            (start - 0.5, start - 0.5), size, size,
            fill=False, edgecolor=OUTLINE_COLOR,
            linewidth=OUTLINE_WIDTH, zorder=6 + depth,
        )
        ax.add_patch(rect)
        center = (start + end) / 2
        ax.text(
            center, center, str(number),
            ha="center", va="center", fontsize=13, fontweight="bold",
            color="black", zorder=20,
            path_effects=[pe.withStroke(linewidth=2.5, foreground="white")],
        )

    ax.set_title(f"{panel_label}.  {title}", fontsize=22, fontweight="bold",
                 loc="left", pad=14)
    ax.set_xlabel("gene (input)", fontsize=14, color="#52514e")
    ax.set_ylabel("gene (output)", fontsize=14, color="#52514e")
    return im


def draw_legend(ax, blocks, ncol=1):
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    n = len(blocks)
    if n == 0:
        return
    rows = -(-n // ncol)
    ax.text(0.0, 0.94, "Functional module (number = diagonal block outline)",
             fontsize=16, fontweight="bold", ha="left", va="top",
             transform=ax.transAxes)
    col_x = [0.0] if ncol == 1 else [0.0, 0.60]
    top = 0.66
    for idx, (start, end, name) in enumerate(blocks):
        row = idx % rows
        col = idx // rows
        x = col_x[col]
        y = top - row * (0.62 / max(rows, 1))
        ax.text(x, y, f"{idx + 1}.", fontsize=15, fontweight="bold",
                 ha="left", va="top", transform=ax.transAxes)
        ax.text(x + 0.045, y, name, fontsize=15,
                 ha="left", va="top", transform=ax.transAxes)


N_HVR_EXCLUDED = 3  # the lowest-SHAP genes are the HVR region; drop entirely


def load_cooccurrence():
    df = pd.read_excel(COOCCURRENCE_FILE, sheet_name="meca_all_gene_cooccurrence")

    # The HVR region genes have by far the most negative mecA SHAP values
    # (an artifact of that region's near-total linkage with mecA) and are
    # excluded from the plot and the fit entirely, not just re-colored.
    hvr_idx = df.nsmallest(N_HVR_EXCLUDED, "Shap Value").index
    df = df.drop(index=hvr_idx).copy()

    count_cols = ["meca_pos_gene_pos", "meca_pos_gene_neg",
                  "meca_neg_gene_pos", "meca_neg_gene_neg"]
    min_count = df[count_cols].min(axis=1)

    # log10 is undefined for a zero or infinite odds ratio (one combo count
    # of 0); those rows are dropped outright rather than force-plotted.
    finite = np.isfinite(df["cooccurrence_odds_ratio"]) & (df["cooccurrence_odds_ratio"] > 0)
    df = df.loc[finite].copy()
    min_count = min_count.loc[finite]
    df["log_odds"] = np.log10(df["cooccurrence_odds_ratio"])
    df["enough_genomes"] = min_count >= MIN_GENOMES_PER_CELL
    return df


def fit_line(x, y):
    result = stats.linregress(x, y)
    return result.slope, result.intercept, result.rvalue ** 2, result.pvalue


def draw_cooccurrence_panel(ax, df):
    hi = df[df["enough_genomes"]]
    lo = df[~df["enough_genomes"]]

    # x = mecA SHAP value, y = log10(co-occurrence odds ratio).
    ax.scatter(lo["Shap Value"], lo["log_odds"], s=10, color=COLOR_EXCLUDED,
               alpha=0.45, linewidths=0, zorder=2,
               label=f"< {MIN_GENOMES_PER_CELL} genomes in ≥1 combination "
                     f"(n={len(lo):,}, excluded from fit)")
    ax.scatter(hi["Shap Value"], hi["log_odds"], s=10, color=COLOR_FIT,
               alpha=0.45, linewidths=0, zorder=3,
               label=f"≥ {MIN_GENOMES_PER_CELL} genomes in all 4 combinations "
                     f"(n={len(hi):,}, used for fit)")

    slope, intercept, r2, pvalue = fit_line(
        hi["Shap Value"].to_numpy(), hi["log_odds"].to_numpy()
    )
    x_line = np.array([df["Shap Value"].min(), df["Shap Value"].max()])
    sign = "+" if intercept >= 0 else "−"
    p_str = f"{pvalue:.2e}" if pvalue < 0.001 else f"{pvalue:.3f}"
    eq = (f"y = {slope:.4f}x {sign} {abs(intercept):.4f}   "
          f"(R² = {r2:.3f}, p = {p_str})")
    ax.plot(x_line, slope * x_line + intercept, color=COLOR_LINE, linewidth=2.6,
            zorder=4, label=f"Line of best fit — {eq}")

    ax.axhline(0, color="#c3c2b7", linewidth=1, zorder=1)
    ax.axvline(0, color="#c3c2b7", linewidth=1, zorder=1)

    ax.set_title("D.  mecA SHAP value vs. gene co-occurrence", fontsize=22,
                 fontweight="bold", loc="left", pad=14)
    ax.set_xlabel("mecA SHAP value", fontsize=14, color="#52514e")
    ax.set_ylabel("log10(co-occurrence odds ratio)", fontsize=14, color="#52514e")
    ax.tick_params(labelsize=12)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color("#c3c2b7")


def draw_scatter_legend(ax, handles):
    ax.axis("off")
    ax.legend(
        handles=handles, loc="center left", frameon=False, fontsize=14,
        markerscale=2.2, labelspacing=1.0, borderaxespad=0,
        handletextpad=0.8,
    )


def make_cooccurrence_panel():
    df = load_cooccurrence()

    pf = plt.figure(figsize=(10.8, 10 + 1.7))
    pgs = pf.add_gridspec(
        nrows=2, ncols=1, height_ratios=[10, 1.7], hspace=0.22,
        left=0.08, right=0.97, top=0.95, bottom=0.03,
    )
    pax = pf.add_subplot(pgs[0, 0])
    draw_cooccurrence_panel(pax, df)

    lax = pf.add_subplot(pgs[1, 0])
    handles, labels = pax.get_legend_handles_labels()
    draw_scatter_legend(lax, handles)

    png_path = OUT / "shap_scatter_mecA_cooccurrence.png"
    pf.savefig(png_path, dpi=300, bbox_inches="tight", pad_inches=0.15)
    pf.savefig(OUT / "shap_scatter_mecA_cooccurrence.pdf", bbox_inches="tight", pad_inches=0.15)
    plt.close(pf)
    return png_path


def main():
    panel_data = []
    for spec in PANELS:
        labels, matrix, blocks = load_panel(spec)
        panel_data.append((spec, labels, matrix, blocks))

    # Render each panel individually at full resolution, then stitch them
    # into one combined figure. Doing the composite in pixel space (rather
    # than one shared matplotlib gridspec) avoids the empty-space blowup
    # that aspect-equal panels of very different gene counts produce when
    # forced to share row-height ratios.
    panel_pngs = []
    for spec, labels, matrix, blocks in panel_data:
        n_groups = len(blocks)
        ncol = 2 if n_groups > 4 else 1
        legend_rows = -(-n_groups // ncol)
        legend_h = 0.5 * legend_rows + 0.5

        pf = plt.figure(figsize=(10.8, 10 + legend_h))
        pgs = pf.add_gridspec(
            nrows=2, ncols=2, width_ratios=[10, 0.35], height_ratios=[10, legend_h],
            hspace=0.14, wspace=0.12,
            left=0.06, right=0.93, top=0.95, bottom=0.02,
        )
        pax = pf.add_subplot(pgs[0, 0])
        pax.set_aspect("equal")
        im = draw_panel(pax, matrix, blocks, spec["title"], spec["label"])

        cax = pf.add_subplot(pgs[0, 1])
        pcbar = pf.colorbar(im, cax=cax, orientation="vertical", extend="both")
        pcbar.set_label("Directional SHAP interaction value", fontsize=15)
        pcbar.ax.tick_params(labelsize=13)

        lax = pf.add_subplot(pgs[1, :])
        draw_legend(lax, blocks, ncol=ncol)

        png_path = OUT / f"shap_heatmap_{spec['key']}.png"
        pf.savefig(png_path, dpi=300, bbox_inches="tight", pad_inches=0.15)
        pf.savefig(OUT / f"shap_heatmap_{spec['key']}.pdf", bbox_inches="tight", pad_inches=0.15)
        plt.close(pf)
        panel_pngs.append(png_path)

    panel_pngs.append(make_cooccurrence_panel())

    stitch_panels(panel_pngs, OUT / "shap_interaction_heatmaps.png")
    print("Saved figures to", OUT)


def stitch_panels(png_paths, out_path, ncols=2):
    """Arrange the individually-rendered panels into an ncols-wide grid."""
    images = [Image.open(p) for p in png_paths]
    target_w = max(im.width for im in images)
    gap = 60
    title_h = 220

    resized = []
    for im in images:
        if im.width != target_w:
            scale = target_w / im.width
            im = im.resize((target_w, round(im.height * scale)), Image.LANCZOS)
        resized.append(im)

    nrows = -(-len(resized) // ncols)
    grid = [resized[i * ncols:(i + 1) * ncols] for i in range(nrows)]
    row_heights = [max(im.height for im in row) for row in grid]

    total_w = ncols * target_w + gap * (ncols - 1)
    total_h = title_h + sum(row_heights) + gap * (nrows - 1) + 20
    canvas = Image.new("RGB", (total_w, total_h), "white")
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 130
        )
    except OSError:
        font = ImageFont.load_default()
    draw.text((30, 40), "Directional SHAP gene–gene interactions", fill="black", font=font)

    y = title_h
    for row, row_h in zip(grid, row_heights):
        x = 0
        for im in row:
            y_off = y + (row_h - im.height) // 2
            canvas.paste(im, (x, y_off))
            x += im.width + gap
        y += row_h + gap

    canvas.save(out_path)


if __name__ == "__main__":
    main()
