import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import typer

plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42
cmap = sns.diverging_palette(230, 20, as_cmap=True)


def plot_directional_heatmap(
    df: pd.DataFrame,
    output_path: str,
    cell_scale: float,
):
    analyzers = df.index.tolist()

    # Mask diagonal (self-transfer)
    mask = np.eye(len(analyzers), dtype=bool)

    sns.set_theme(style="white", font_scale=0.7)

    base_w, base_h = 6.0, 5.0
    fig, ax = plt.subplots(figsize=(base_w * cell_scale, base_h * cell_scale))

    sns.heatmap(
        df,
        mask=mask,
        cmap=cmap,
        vmin=0.0,
        vmax=1.0,
        square=True,
        linewidths=0.5,
        annot=True,
        fmt=".2f",
        cbar_kws={"label": "Escape Success Rate"},
        ax=ax,
    )

    ax.set_xlabel("Target Analyzer")
    ax.set_ylabel("Source Analyzer")

    plt.tight_layout()
    plt.savefig(output_path, dpi=500, bbox_inches="tight")


def main(input_path: str, output_path: str = "out.png", scale: float = 0.7):
    # Load CSV
    df = pd.read_csv(input_path, index_col=0)
    plot_directional_heatmap(df, output_path, scale)


if __name__ == "__main__":
    typer.run(main)
