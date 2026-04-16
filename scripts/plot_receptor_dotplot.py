"""Dotplot of receptor % expression and copy number across immune subsets.

For the investor deck: targets as rows, immune subsets as columns.
- Dot size  = mean % Receptor Expression (across donors)
- Dot color = mean Receptor Density / copy # per cell (across donors)

Only a small disclosed set of targets is labeled by name; all others are
relabeled 'Undisclosed N' to protect IP.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
REPO = Path(__file__).resolve().parents[1]
XLSX = REPO / "data" / "source_xlsx" / "compiled_data_vFinal.xlsx"
OUT_DIR = REPO / "figures"
OUT_STEM = "receptor_dotplot_investor"

DISCLOSED = ["CD44", "CD25", "CD69", "KLRG1", "PD1", "TNFRSF4"]

SUBSETS = [
    ("B Cells", "B Cells"),
    ("NK Cells", "NK Cells"),
    ("CD4+ Cells", "CD4+ T Cells"),
    ("CD8+ Cells", "CD8+ T Cells"),
]

SIZE_LEVELS = [25, 50, 75, 100]  # %
SIZE_SCALE = 0.55  # multiplier: dot area = pct * SIZE_SCALE

FIG_W = 2.8
FIG_H = 3.0

# Drop targets whose max % expression across the 4 subsets is below this.
MIN_MAX_PCT = 5.0

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 8,
        "axes.linewidth": 0.6,
        "axes.edgecolor": "#333333",
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.major.size": 2,
        "ytick.major.size": 2,
        "savefig.facecolor": "white",
        "figure.facecolor": "white",
    }
)


# ---------------------------------------------------------------------------
# Load & reshape
# ---------------------------------------------------------------------------
def load_long(xlsx: Path) -> pd.DataFrame:
    df = pd.read_excel(xlsx, sheet_name="Sheet1")
    rows = []
    for _, r in df.iterrows():
        target = r["Target"]
        tclass = r.get("Target Classification", "")
        donor = r.get("Donor ID", None)
        for src_name, display in SUBSETS:
            pct = r.get(f"{src_name} % Receptor Expression", np.nan)
            dens = r.get(f"{src_name} Receptor Density", np.nan)
            rows.append(
                {
                    "target": target,
                    "target_class": tclass,
                    "donor": donor,
                    "subset": display,
                    "pct": pct,
                    "density": dens,
                }
            )
    return pd.DataFrame(rows)


def aggregate(long: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    agg = (
        long.groupby(["target", "target_class", "subset"], as_index=False)
        .agg(pct=("pct", "mean"), density=("density", "mean"))
    )
    pct_wide = agg.pivot(index="target", columns="subset", values="pct")
    dens_wide = agg.pivot(index="target", columns="subset", values="density")
    class_map = (
        agg[["target", "target_class"]].drop_duplicates().set_index("target")
    )
    # reorder subset columns
    col_order = [d for _, d in SUBSETS]
    pct_wide = pct_wide.reindex(columns=col_order)
    dens_wide = dens_wide.reindex(columns=col_order)
    return pct_wide, dens_wide, class_map


def build_display_labels(targets: list[str], class_map: pd.DataFrame) -> dict[str, str]:
    """Return {target: display_label}. Disclosed targets keep their name;
    everything else becomes 'Undisclosed N' numbered stably."""
    labels: dict[str, str] = {}
    disclosed_set = set(DISCLOSED)
    # Undisclosed ordering: by target_class then alphabetical by name
    undisclosed = [t for t in targets if t not in disclosed_set]
    undisclosed_sorted = sorted(
        undisclosed,
        key=lambda t: (str(class_map.loc[t, "target_class"]) if t in class_map.index else "", t),
    )
    for t in disclosed_set:
        if t in targets:
            labels[t] = t
    for i, t in enumerate(undisclosed_sorted, start=1):
        labels[t] = f"Undisclosed {i}"
    return labels


def row_order(targets: list[str], class_map: pd.DataFrame) -> list[str]:
    """Distribute disclosed targets roughly evenly across the rows so their
    labels don't overlap in a compressed figure; undisclosed fill the gaps,
    ordered by class then name."""
    present_disclosed = [t for t in DISCLOSED if t in targets]
    disclosed_set = set(DISCLOSED)
    undisclosed = [t for t in targets if t not in disclosed_set]
    undisclosed_sorted = sorted(
        undisclosed,
        key=lambda t: (str(class_map.loc[t, "target_class"]) if t in class_map.index else "", t),
    )
    n_total = len(present_disclosed) + len(undisclosed_sorted)
    n_d = len(present_disclosed)
    # Even positions across [0, n_total-1]
    positions = [round(k * (n_total - 1) / max(n_d - 1, 1)) for k in range(n_d)]
    order: list[str | None] = [None] * n_total
    for pos, t in zip(positions, present_disclosed):
        order[pos] = t
    und_iter = iter(undisclosed_sorted)
    for i in range(n_total):
        if order[i] is None:
            order[i] = next(und_iter)
    return order  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------
def plot_dotplot(
    pct: pd.DataFrame, dens: pd.DataFrame, labels: dict[str, str], out_stem: Path
) -> None:
    targets = list(pct.index)
    subsets = list(pct.columns)
    n_rows = len(targets)
    n_cols = len(subsets)

    # Build coordinate arrays (row 0 at top)
    xs, ys, sizes, colors = [], [], [], []
    for i, t in enumerate(targets):
        for j, s in enumerate(subsets):
            p = pct.loc[t, s]
            d = dens.loc[t, s]
            if pd.isna(p) or pd.isna(d):
                continue
            xs.append(j)
            ys.append(n_rows - 1 - i)  # flip so first target appears at top
            sizes.append(max(p, 0) * SIZE_SCALE)
            colors.append(d)

    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))

    vmax = np.nanpercentile(dens.values, 98)
    sc = ax.scatter(
        xs,
        ys,
        s=sizes,
        c=colors,
        cmap="Blues",
        vmin=0,
        vmax=vmax,
        edgecolor="black",
        linewidth=0.3,
    )

    # Axes — tight xlim packs columns closer together
    ax.set_xlim(-0.35, n_cols - 0.65)
    ax.set_ylim(-0.5, n_rows - 0.5)

    # Very light grid at every row and column
    ax.set_xticks(np.arange(n_cols), minor=False)
    ax.set_yticks(np.arange(n_rows), minor=True)
    ax.grid(which="major", axis="x", color="#e8e8e8", linewidth=0.3, zorder=0)
    ax.grid(which="minor", axis="y", color="#f0f0f0", linewidth=0.25, zorder=0)
    ax.set_axisbelow(True)
    ax.set_xticks(range(n_cols))
    ax.set_xticklabels(subsets, rotation=35, ha="left", fontsize=6)
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position("top")

    # Only show ticks/labels for disclosed targets; hide undisclosed rows
    disclosed_set = set(DISCLOSED)
    disclosed_positions = []
    disclosed_labels = []
    for i, t in enumerate(targets):
        if t in disclosed_set:
            disclosed_positions.append(n_rows - 1 - i)
            disclosed_labels.append(t)
    ax.set_yticks(disclosed_positions)
    ax.set_yticklabels(disclosed_labels, fontsize=6)
    for tick_label in ax.get_yticklabels():
        tick_label.set_fontweight("bold")
        tick_label.set_color("#1f3a93")

    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.5)
        spine.set_edgecolor("#333333")
    ax.tick_params(axis="x", length=0)
    ax.tick_params(axis="y", length=0)

    # Size legend
    size_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor="#4a78c2",
            markeredgecolor="black",
            markeredgewidth=0.3,
            markersize=np.sqrt(lvl * SIZE_SCALE),
            label=f"{lvl}%",
        )
        for lvl in SIZE_LEVELS
    ]
    leg = ax.legend(
        handles=size_handles,
        title="% expr",
        loc="upper left",
        bbox_to_anchor=(1.08, 1.0),
        frameon=False,
        labelspacing=0.55,
        handletextpad=0.4,
        borderpad=0.2,
        fontsize=5.5,
        title_fontsize=6,
    )
    leg._legend_box.align = "left"
    ax.add_artist(leg)

    # Colorbar
    cax = fig.add_axes([0.68, 0.08, 0.025, 0.22])
    cbar = fig.colorbar(sc, cax=cax)
    cbar.set_label("Copy #/cell", fontsize=6, labelpad=2)
    cbar.ax.tick_params(labelsize=5, width=0.4, length=1.5, pad=1)
    cbar.outline.set_linewidth(0.3)

    fig.subplots_adjust(left=0.22, right=0.58, top=0.82, bottom=0.04)

    out_stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_stem.with_suffix(".png"), dpi=600, bbox_inches="tight")
    fig.savefig(out_stem.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_stem.with_suffix('.png')}")
    print(f"Wrote {out_stem.with_suffix('.svg')}")


# ---------------------------------------------------------------------------
def main() -> None:
    long = load_long(XLSX)
    pct, dens, class_map = aggregate(long)
    # Drop isotype controls and ubiquitously low / null targets
    def keep(t: str) -> bool:
        tl = t.lower()
        if "isotype" in tl or "ctrl" in tl:
            return False
        row = pct.loc[t]
        if row.isna().all():
            return False
        if np.nanmax(row.values) < MIN_MAX_PCT:
            return False
        return True

    kept = [t for t in pct.index if keep(t)]
    dropped = [t for t in pct.index if t not in kept]
    print(f"Dropped {len(dropped)} targets (isotype/low): {dropped}")
    pct = pct.loc[kept]
    dens = dens.loc[kept]

    targets = list(pct.index)
    order = row_order(targets, class_map)
    pct = pct.loc[order]
    dens = dens.loc[order]
    labels = build_display_labels(order, class_map)

    # Sanity print
    print(f"Targets: {len(order)}  Subsets: {list(pct.columns)}")
    print("Disclosed present:", [t for t in DISCLOSED if t in order])

    plot_dotplot(pct, dens, labels, OUT_DIR / OUT_STEM)


if __name__ == "__main__":
    main()
