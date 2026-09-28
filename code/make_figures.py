"""Create publication figures from the audited #15 result tables."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import t

from panel_alignment_audit import require_matplotlib_panel_alignment


ROOT = Path(__file__).resolve().parents[1]
MAIN = ROOT / "code" / "outputs" / "main_fixed_iid_20260914" / "raw_results.csv"
SHIFT = ROOT / "code" / "outputs" / "shift_stress_20260915" / "raw_results.csv"
OUT = ROOT / "figures"

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans"],
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "font.size": 7,
        "axes.spines.right": False,
        "axes.spines.top": False,
        "axes.linewidth": 0.7,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "legend.frameon": False,
    }
)

COLORS = {"dt": "#0F4D92", "cat": "#42949E"}
LABELS = {"dt": "DT", "cat": "CatBoost"}


def add_panel_label(ax, label):
    ax.text(
        -0.14,
        1.04,
        label,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=8,
        fontweight="bold",
    )


def bootstrap_ci(values):
    values = np.asarray(values, dtype=float)
    mean = float(values.mean())
    half = float(t.ppf(0.975, len(values) - 1) * values.std(ddof=1) / np.sqrt(len(values)))
    return mean, mean - half, mean + half


def paired_task_values(raw, learner, metric, method="smote"):
    wide = raw.pivot_table(
        index=["dataset", "learner", "method", "seed", "scenario", "severity"],
        columns="metric",
        values="value",
    ).reset_index()
    subset = wide[
        (wide["learner"] == learner)
        & (wide["method"].isin(["none", method]))
        & (wide["scenario"] == "iid")
    ]
    table = subset.pivot_table(index=["dataset", "seed"], columns="method", values=metric)
    return (table[method] - table["none"]).groupby(level="dataset").mean()


def make_figure1():
    raw = pd.read_csv(MAIN)
    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(7.1, 3.0), gridspec_kw={"width_ratios": [0.9, 1.25]})
    add_panel_label(ax_a, "a")
    add_panel_label(ax_b, "b")

    source_rows = []

    for xpos, learner in enumerate(["dt", "cat"]):
        auc = paired_task_values(raw, learner, "auc")
        # Deterministic visual jitter only; no observations are sampled or excluded.
        jitter = np.linspace(-0.11, 0.11, len(auc))
        ax_a.scatter(
            np.full(len(auc), xpos) + jitter,
            auc.to_numpy(),
            s=13,
            color=COLORS[learner],
            alpha=0.48,
            linewidth=0,
            zorder=2,
        )
        mean, low, high = bootstrap_ci(auc.to_numpy())
        ax_a.errorbar(
            xpos,
            mean,
            yerr=[[mean - low], [high - mean]],
            fmt="o",
            color=COLORS[learner],
            ecolor="#272727",
            elinewidth=1.3,
            capsize=3,
            markersize=5,
            markeredgecolor="#272727",
            zorder=3,
        )
    ax_a.axhline(0, color="#767676", linestyle="--", linewidth=0.8)
    ax_a.set_xticks([0, 1], ["DT", "CatBoost"])
    ax_a.set_ylabel("SMOTE − none ΔAUC")
    ax_a.set_title("Task-clustered ranking effect", loc="left", fontsize=8, pad=7)
    ax_a.set_xlim(-0.4, 1.4)
    ax_a.grid(axis="y", color="#D9D9D9", linewidth=0.5, alpha=0.6)

    for learner in ["dt", "cat"]:
        auc = paired_task_values(raw, learner, "auc")
        metric_wide = raw.pivot_table(
            index=["dataset", "learner", "method", "seed", "scenario", "severity"],
            columns="metric",
            values="value",
        ).reset_index()
        base = metric_wide[
            (metric_wide["learner"] == learner)
            & (metric_wide["method"].isin(["none", "smote"]))
            & (metric_wide["scenario"] == "iid")
        ].pivot_table(index=["dataset", "seed"], columns="method", values="loss_minority")
        relative = ((base["none"] - base["smote"]) / base["none"].abs()).groupby(level="dataset").mean()
        majority = paired_task_values(raw, learner, "loss_majority")
        for dataset in auc.index:
            source_rows.append(
                {
                    "dataset": dataset,
                    "learner": LABELS[learner],
                    "delta_auc": float(auc.loc[dataset]),
                    "minority_logloss_relative_improvement": float(relative.loc[dataset]),
                    "delta_majority_logloss": float(majority.loc[dataset]),
                }
            )
        edge = np.where(majority.reindex(auc.index).to_numpy() > 0, "#B64342", "white")
        ax_b.scatter(
            auc.to_numpy(),
            relative.to_numpy(),
            s=23,
            color=COLORS[learner],
            edgecolor=edge,
            linewidth=0.7,
            alpha=0.78,
            label=LABELS[learner],
        )
    ax_b.axvline(0, color="#767676", linestyle="--", linewidth=0.8)
    ax_b.axhline(0.05, color="#767676", linestyle=":", linewidth=0.8)
    ax_b.set_xlabel("SMOTE − none ΔAUC")
    ax_b.set_ylabel("Minority logloss relative improvement")
    ax_b.set_title("Ranking–risk separation across tasks", loc="left", fontsize=8, pad=7)
    ax_b.grid(color="#D9D9D9", linewidth=0.5, alpha=0.55)
    handles, labels = ax_b.get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.03),
        ncol=2,
        fontsize=7,
        handletextpad=0.4,
    )

    fig.tight_layout(pad=1.0, w_pad=2.0, rect=[0, 0, 1, 0.94])
    pd.DataFrame(source_rows).to_csv(OUT / "source_data_figure1.csv", index=False)
    base = OUT / "figure1_primary_effects"
    require_matplotlib_panel_alignment(
        fig,
        json_out=str(base) + ".alignment.json",
        overlay_svg=str(base) + ".alignment.svg",
        tolerance_pt=1.5,
        gutter_tolerance_pt=1.5,
        strict=True,
    )
    fig.savefig(str(base) + ".svg", bbox_inches="tight")
    fig.savefig(str(base) + ".pdf", bbox_inches="tight")
    fig.savefig(str(base) + ".png", dpi=600, bbox_inches="tight")
    fig.savefig(str(base) + ".tiff", dpi=600, bbox_inches="tight")
    plt.close(fig)


def shift_delta(raw, learner, scenario, severity, metric):
    subset = raw[
        (raw["learner"] == learner)
        & (raw["method"].isin(["none", "smote_q05_k5"]))
        & (raw["scenario"] == scenario)
        & (raw["severity"] == severity)
        & (raw["metric"] == metric)
    ]
    wide = subset.pivot_table(
        index=["dataset", "seed"], columns="method", values="value"
    )
    return (wide["smote_q05_k5"] - wide["none"]).groupby(level="dataset").mean()


def make_figure2():
    raw = pd.read_csv(SHIFT)
    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(7.1, 3.0))
    add_panel_label(ax_a, "a")
    add_panel_label(ax_b, "b")

    source_rows = []

    priors = [0.05, 0.20, 0.50]
    for learner in ["dt_leaf01", "cat_iter500"]:
        means, lows, highs = [], [], []
        for severity in [0, 1, 2]:
            values = shift_delta(raw, learner, "label_shift", severity, "brier")
            mean, low, high = bootstrap_ci(values.to_numpy())
            means.append(mean)
            lows.append(low)
            highs.append(high)
            source_rows.append(
                {
                    "learner": LABELS["dt"] if learner.startswith("dt") else LABELS["cat"],
                    "scenario": "label_shift",
                    "severity": severity,
                    "target_positive_prior": priors[severity],
                    "covariate_shift_strength_sd": np.nan,
                    "delta_brier": mean,
                    "delta_majority_logloss": np.nan,
                }
            )
        color = COLORS["dt"] if learner.startswith("dt") else COLORS["cat"]
        label = LABELS["dt"] if learner.startswith("dt") else LABELS["cat"]
        ax_a.errorbar(
            priors,
            means,
            yerr=[np.array(means) - np.array(lows), np.array(highs) - np.array(means)],
            marker="o",
            markersize=4,
            linewidth=1.5,
            capsize=2.5,
            color=color,
            label=label,
        )
    ax_a.axhline(0, color="#767676", linestyle="--", linewidth=0.8)
    ax_a.set_xlabel("Target positive-class prior")
    ax_a.set_ylabel("SMOTE − none ΔBrier")
    ax_a.set_title("Label shift changes aggregate risk", loc="left", fontsize=8, pad=7)
    ax_a.set_xticks(priors, ["0.05", "0.20", "0.50"])
    ax_a.grid(axis="y", color="#D9D9D9", linewidth=0.5, alpha=0.6)

    for learner in ["dt_leaf01", "cat_iter500"]:
        means, lows, highs = [], [], []
        for severity in [0, 1, 2]:
            values = shift_delta(raw, learner, "covariate_shift", severity, "loss_majority")
            mean, low, high = bootstrap_ci(values.to_numpy())
            means.append(mean)
            lows.append(low)
            highs.append(high)
            source_rows.append(
                {
                    "learner": LABELS["dt"] if learner.startswith("dt") else LABELS["cat"],
                    "scenario": "covariate_shift",
                    "severity": severity,
                    "target_positive_prior": np.nan,
                    "covariate_shift_strength_sd": severity,
                    "delta_brier": np.nan,
                    "delta_majority_logloss": mean,
                }
            )
        color = COLORS["dt"] if learner.startswith("dt") else COLORS["cat"]
        label = LABELS["dt"] if learner.startswith("dt") else LABELS["cat"]
        ax_b.errorbar(
            [0, 1, 2],
            means,
            yerr=[np.array(means) - np.array(lows), np.array(highs) - np.array(means)],
            marker="o",
            markersize=4,
            linewidth=1.5,
            capsize=2.5,
            color=color,
            label=label,
        )
    ax_b.axhline(0, color="#767676", linestyle="--", linewidth=0.8)
    ax_b.set_xlabel("Covariate shift strength (train SDs)")
    ax_b.set_ylabel("SMOTE − none Δmajority logloss")
    ax_b.set_title("Covariate stress amplifies majority risk", loc="left", fontsize=8, pad=7)
    ax_b.set_xticks([0, 1, 2])
    ax_b.grid(axis="y", color="#D9D9D9", linewidth=0.5, alpha=0.6)
    handles, labels = ax_a.get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.03),
        ncol=2,
        fontsize=7,
    )

    fig.tight_layout(pad=1.0, w_pad=2.0, rect=[0, 0, 1, 0.94])
    pd.DataFrame(source_rows).to_csv(OUT / "source_data_figure2.csv", index=False)
    base = OUT / "figure2_shift_boundary"
    require_matplotlib_panel_alignment(
        fig,
        json_out=str(base) + ".alignment.json",
        overlay_svg=str(base) + ".alignment.svg",
        tolerance_pt=1.5,
        gutter_tolerance_pt=1.5,
        strict=True,
    )
    fig.savefig(str(base) + ".svg", bbox_inches="tight")
    fig.savefig(str(base) + ".pdf", bbox_inches="tight")
    fig.savefig(str(base) + ".png", dpi=600, bbox_inches="tight")
    fig.savefig(str(base) + ".tiff", dpi=600, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    make_figure1()
    make_figure2()
    print("figures written to", OUT)
