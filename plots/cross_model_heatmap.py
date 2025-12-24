import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import typer
import matplotlib.axes as axes
from matplotlib.colors import Normalize

plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42
cmap = sns.diverging_palette(230, 20, as_cmap=True)


def _draw_heatmap(ax: axes.Axes, df: pd.DataFrame, show_cbar: bool = True):
    analyzers = df.index.tolist()
    mask = np.eye(len(analyzers), dtype=bool)

    return sns.heatmap(
        df,
        mask=mask,
        cmap=cmap,
        vmin=0.0,
        vmax=1.0,
        square=True,
        linewidths=0.5,
        annot=True,
        fmt=".2f",
        cbar=show_cbar,
        cbar_kws={"label": "Escape Success Rate"} if show_cbar else None,
        ax=ax,
    )


def plot_directional_heatmap(
    df: pd.DataFrame,
    output_path: str,
    cell_scale: float,
):
    sns.set_theme(style="white", font_scale=0.7)

    base_w, base_h = 6.0, 5.0
    fig, ax = plt.subplots(figsize=(base_w * cell_scale, base_h * cell_scale))

    _draw_heatmap(ax, df, show_cbar=True)
    ax.set_xlabel("Target Analyzer")
    ax.set_ylabel("Source Analyzer")

    plt.tight_layout()
    plt.savefig(output_path, dpi=500, bbox_inches="tight")


def plot_heatmaps_from_directory(
    input_dir: Path,
    output_path: str,
    scale: float,
):
    csv_files = sorted(input_dir.glob("*.csv"))
    if not csv_files:
        raise ValueError(f"No CSV files found in {input_dir}")

    sns.set_theme(style="white", font_scale=0.7)

    cols = 3
    rows = math.ceil(len(csv_files) / cols)

    base_w, base_h = 6.0, 5.0

    # +1 column reserved for colorbar
    fig = plt.figure(figsize=(base_w * cols * scale + 0.6, base_h * rows * scale))
    gs = fig.add_gridspec(
        rows,
        cols + 1,
        width_ratios=[1] * cols + [0.05],
        wspace=0.25,
        hspace=0.15,
    )

    heatmap_axes = []
    for r in range(rows):
        for c in range(cols):
            idx = r * cols + c
            if idx >= len(csv_files):
                continue
            ax = fig.add_subplot(gs[r, c])
            heatmap_axes.append(ax)

            df = pd.read_csv(csv_files[idx], index_col=0)
            _draw_heatmap(ax, df, show_cbar=False)
            ax.set_title(csv_files[idx].stem)
            ax.set_xlabel("Target Analyzer")
            ax.set_ylabel("Source Analyzer")

    # Colorbar axis (single, shared)
    cax = fig.add_subplot(gs[:, -1])
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=Normalize(0, 1))
    sm.set_array([])
    fig.colorbar(sm, cax=cax, label="Escape Success Rate")

    cax.set_frame_on(False)
    for spine in cax.spines.values():
        spine.set_visible(False)

    plt.savefig(output_path, dpi=500, bbox_inches="tight")


def main(
    input_path: str,
    output_path: str = "out.png",
    scale: float = 0.7,
):
    path = Path(input_path)
    if path.is_dir():
        plot_heatmaps_from_directory(path, output_path, scale)
    else:
        df = pd.read_csv(path, index_col=0)
        plot_directional_heatmap(df, output_path, scale)


if __name__ == "__main__":
    typer.run(main)
