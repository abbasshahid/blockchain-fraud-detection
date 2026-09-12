from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from blockchain_fraud.analysis.make_tables import main_runs
from blockchain_fraud.utils.io import ensure_dir, read_json


COLORS = {
    "xgboost": "#0072B2",
    "gcn": "#009E73",
    "graphsage": "#D55E00",
    "gat": "#CC79A7",
}


def build_plots(output_dir: str | Path = "outputs") -> dict[str, str]:
    output_dir = Path(output_dir)
    fig_dir = ensure_dir(output_dir / "figures")
    paths: dict[str, str] = {}
    df = main_runs(output_dir)
    if not df.empty:
        summary = df.groupby("model")[["f1_illicit", "pr_auc", "mcc"]].mean().reset_index()
        summary.to_csv(fig_dir / "fig03_model_comparison_data.csv", index=False)
        ax = summary.set_index("model").plot(
            kind="bar",
            color=["#0072B2", "#009E73", "#D55E00"],
            figsize=(7, 4),
            ylim=(0, 1),
        )
        ax.set_ylabel("Score (higher is better)")
        ax.set_xlabel("Model")
        ax.legend(loc="best")
        plt.tight_layout()
        plt.savefig(fig_dir / "fig03_model_comparison.png", dpi=300)
        plt.savefig(fig_dir / "fig03_model_comparison.pdf")
        plt.close()
        paths["fig03"] = str(fig_dir / "fig03_model_comparison.png")

    manifests = sorted((output_dir / "metrics").glob("*_split_manifest.json"))
    if manifests:
        manifest = read_json(manifests[-1])
        rows = []
        for split, counts in manifest["counts"].items():
            rows.extend(
                [
                    {"split": split, "class": "illicit", "count": counts["illicit"]},
                    {"split": split, "class": "licit", "count": counts["licit"]},
                    {"split": split, "class": "unknown", "count": counts["unknown"]},
                ]
            )
        split_df = pd.DataFrame(rows)
        split_df.to_csv(fig_dir / "fig02_temporal_class_distribution_data.csv", index=False)
        pivot = split_df.pivot(index="split", columns="class", values="count").loc[["train", "val", "test"]]
        ax = pivot.plot(kind="bar", stacked=True, color=["#D55E00", "#009E73", "#999999"], figsize=(7, 4))
        ax.set_ylabel("Nodes")
        ax.set_xlabel("Temporal split")
        plt.tight_layout()
        plt.savefig(fig_dir / "fig02_temporal_class_distribution.png", dpi=300)
        plt.savefig(fig_dir / "fig02_temporal_class_distribution.pdf")
        plt.close()
        paths["fig02"] = str(fig_dir / "fig02_temporal_class_distribution.png")
    return paths

