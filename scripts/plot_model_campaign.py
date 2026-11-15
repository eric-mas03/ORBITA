"""Plot matched individual-model evaluations without fitting or selection."""
import argparse
import importlib.util
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("campaign_report", ROOT / "scripts/report_model_campaign.py")
report = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(report)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/model_campaign")
    parser.add_argument("--split", choices=["dev", "confirmation"], default="dev")
    parser.add_argument("--paths", nargs="+", type=Path, required=True)
    args = parser.parse_args()
    if args.split == "dev":
        fig, ax = plt.subplots(figsize=(10, 5))
        for directory in ("hourly_architecture_trials", "hourly_residual_control"):
            path = ROOT / "outputs" / directory / "learning_curves.csv"
            if not path.exists():
                continue
            curves = pd.read_csv(path)
            if "method" not in curves:
                curves["method"] = "own_summary_mlp_residual"
            for method, curve in curves.groupby("method"):
                curve = curve.sort_values("epoch")
                ax.plot(curve.epoch, curve.dev_macro_pinball, "o-", markersize=3, label=method)
                best = curve.loc[curve.dev_macro_pinball.idxmin()]
                ax.scatter([best.epoch], [best.dev_macro_pinball], s=70, facecolors="none", edgecolors="black")
        ax.axhline(.156520754, color="gray", linestyle="--", label="Lineal congelada · época 0")
        ax.set_xlabel("Época")
        ax.set_ylabel("Pinball macro normalizada en desarrollo ↓")
        ax.set_title("Ajuste residual · círculo negro = mínimo observado")
        ax.grid(alpha=.2)
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(args.output / "dev_learning_curves.png", dpi=150)
        plt.close(fig)
    cells = pd.read_csv(args.output / f"{args.split}_by_variable_horizon.csv")
    valid = cells[cells.n_labels.gt(0)]
    horizons = valid.groupby(["method", "horizon_hours"]).agg(
        pinball=("normalized_pinball", "mean"), coverage=("coverage", "mean"),
        mae=("normalized_mae", "mean"), width=("normalized_width", "mean")).reset_index()
    horizons.to_csv(args.output / f"{args.split}_by_horizon.csv", index=False)
    methods = horizons.groupby("method").pinball.mean().sort_values().index.tolist()
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for method in methods:
        sub = horizons[horizons.method.eq(method)].sort_values("horizon_hours")
        for ax, metric in zip(axes, ["pinball", "mae", "coverage"]):
            ax.plot(sub.horizon_hours, sub[metric], "o-", label=method, linewidth=1.3, markersize=4)
    for ax, title in zip(axes, ["Pinball normalizada ↓", "MAE normalizado ↓", "Cobertura q10–q90"]):
        ax.set_title(title)
        ax.set_xticks([1, 4, 8, 24])
        ax.set_xlabel("Horizonte (horas)")
        ax.grid(alpha=.2)
    axes[2].axhline(.8, color="gray", linestyle="--")
    axes[2].set_ylim(0, 1)
    axes[0].legend(fontsize=7)
    fig.suptitle(f"{args.split} · mismo peso por variable, mismas etiquetas observadas")
    fig.tight_layout()
    fig.savefig(args.output / f"{args.split}_horizons.png", dpi=150)
    plt.close(fig)
    champion = methods[0]
    if args.split == "confirmation":
        selection = json.loads((args.output / "selection_frozen.json").read_text(encoding="utf-8"))
        champion = selection["selected_new_methods"][0]
    heads = valid[valid.method.eq(champion)].pivot_table(index="head", columns="horizon_hours", values="normalized_pinball")
    fig, ax = plt.subplots(figsize=(8, 5))
    im = ax.imshow(heads, aspect="auto", cmap="YlOrRd")
    ax.set_yticks(range(len(heads)), heads.index)
    ax.set_xticks(range(len(heads.columns)), [f"{h} h" for h in heads.columns])
    ax.set_title(f"{args.split} · {champion}\nPinball normalizada por cabeza")
    for row in range(len(heads)):
        for col in range(len(heads.columns)):
            ax.text(col, row, f"{heads.iloc[row, col]:.3f}", ha="center", va="center", fontsize=9)
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(args.output / f"{args.split}_heads.png", dpi=150)
    plt.close(fig)
    by_head = valid.groupby(["method", "head", "horizon_hours"]).normalized_pinball.mean().reset_index()
    head_names = sorted(by_head["head"].unique())
    fig, axes = plt.subplots(int(np.ceil(len(head_names)/2)), 2,
                             figsize=(13, 3 * int(np.ceil(len(head_names)/2))), squeeze=False)
    for ax, head in zip(axes.flat, head_names):
        for method in methods:
            series = by_head[by_head.method.eq(method) & by_head["head"].eq(head)].sort_values("horizon_hours")
            ax.plot(series.horizon_hours, series.normalized_pinball, "o-", label=method, linewidth=1, markersize=3)
        ax.set_title(head)
        ax.set_xticks([1, 4, 8, 24])
        ax.set_xlabel("Horizonte (h)")
        ax.set_ylabel("Pinball normalizada ↓")
        ax.grid(alpha=.2)
    for ax in list(axes.flat)[len(head_names):]:
        ax.set_visible(False)
    axes.flat[0].legend(fontsize=6)
    fig.suptitle(f"{args.split} · comparación por cabeza biomédica y horizonte")
    fig.tight_layout(rect=(0, 0, 1, .97))
    fig.savefig(args.output / f"{args.split}_head_comparison.png", dpi=150)
    plt.close(fig)
    # A reproducible illustration, selected by ID and observation count, never
    # by prediction error. Each point is a forecast issued 4 h earlier.
    frame = report.read_split(args.paths, args.split)
    sub = frame[frame.method.eq(champion) & frame.horizon_hours.eq(4) & frame.variable.isin(["HR", "O2Sat", "Resp", "Creatinine"])]
    counts = sub.groupby("episode_id").size().sort_index()
    eligible = counts[counts.ge(30)]
    if eligible.empty:
        eligible = counts
    if not len(eligible):
        return
    episode = eligible.index[0]
    fig, axes = plt.subplots(4, 1, figsize=(12, 10))
    for ax, variable in zip(axes, ["HR", "O2Sat", "Resp", "Creatinine"]):
        trace = sub[sub.episode_id.eq(episode) & sub.variable.eq(variable)].sort_values("target_hour")
        if trace.empty:
            ax.text(.5, .5, "Sin etiquetas reales para esta variable", ha="center", transform=ax.transAxes)
        else:
            x = trace.target_hour.to_numpy()
            # Scatter avoids implying unobserved laboratory measurements.
            ax.scatter(x, trace.observed, s=12, color="black", label="Medición real")
            ax.plot(x, trace.q50, color="#176b87", label="q50 (emisión t−4 h)")
            ax.fill_between(x, trace.q10, trace.q90, color="#66aec2", alpha=.3, label="q10–q90")
        ax.set_ylabel(variable)
        ax.grid(alpha=.2)
    axes[0].legend(fontsize=8)
    axes[-1].set_xlabel("Hora desde el ingreso; solo pares con etiqueta real")
    fig.suptitle(f"{args.split} · {champion} · {episode}\nEjemplo elegido por ID, no por calidad del ajuste")
    fig.tight_layout()
    fig.savefig(args.output / f"{args.split}_forecast_example.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
