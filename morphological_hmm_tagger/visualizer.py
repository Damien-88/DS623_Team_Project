"""Plotting helpers for benchmark results."""

from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns

try:
    from .evaluator import results_to_frame
except ImportError:
    from evaluator import results_to_frame


def plot_benchmark_results(results, output_path=None):
    """Create a compact accuracy/latency comparison chart."""
    frame = results_to_frame(results)
    plot_frame = frame.melt(
        id_vars=["model_name"],
        value_vars=["overall_accuracy", "known_token_accuracy", "oov_token_accuracy"],
        var_name="metric",
        value_name="score",
    )

    sns.set_theme(style="whitegrid", context="talk")
    figure, axes = plt.subplots(1, 2, figsize=(14, 6), constrained_layout=True)

    sns.barplot(data=plot_frame, x="model_name", y="score", hue="metric", ax=axes[0], palette="viridis")
    axes[0].set_ylim(0, 1)
    axes[0].set_title("Accuracy by model")
    axes[0].set_xlabel("")
    axes[0].set_ylabel("Accuracy")
    axes[0].tick_params(axis="x", rotation=20)

    latency_frame = frame[["model_name", "milliseconds_per_sentence", "milliseconds_per_token"]].melt(
        id_vars=["model_name"],
        var_name="metric",
        value_name="milliseconds",
    )
    sns.barplot(data=latency_frame, x="model_name", y="milliseconds", hue="metric", ax=axes[1], palette="magma")
    axes[1].set_title("Inference latency")
    axes[1].set_xlabel("")
    axes[1].set_ylabel("Milliseconds")
    axes[1].tick_params(axis="x", rotation=20)

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output_path, dpi=200, bbox_inches="tight")

    return figure


def results_to_latex(results):
    """Return a LaTeX table for inclusion in the paper."""
    frame = results_to_frame(results)
    return frame.to_latex(index=False, float_format=lambda value: f"{value:.4f}")