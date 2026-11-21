"""Plot the recorded validation histories without running training or inference."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    runs = json.loads((root / "reports/experiment_comparison.json").read_text())
    output = root / "reports/figures/validation_comparison.png"
    output.parent.mkdir(parents=True, exist_ok=True)
    colors = ("#174A73", "#BD681C", "#367C65")
    markers = ("o", "s", "^")
    with plt.rc_context({"font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False, "axes.titleweight": "bold"}):
        fig, axes = plt.subplots(1, 2, figsize=(10.6, 3.05), layout="constrained")
        for run, color, marker in zip(runs, colors, markers):
            for ax, metric in zip(axes, ("val_macro_f1", "val_loss")):
                points = run["history"][metric]
                ax.plot([p["step"] for p in points], [p["value"] for p in points],
                        label=f"Run {run['label']}", color=color, marker=marker,
                        linewidth=2, markersize=5)
        for ax in axes:
            ax.set_xlabel("Époque")
            ax.set_xticks([1, 2, 3, 4])
            ax.grid(axis="y", alpha=0.2)
            ax.legend(frameon=False, loc="best", ncol=3)
        axes[0].set(title="Macro-F1 de validation", ylabel="Score", ylim=(0.40, 0.70))
        axes[1].set(title="Loss de validation", ylabel="Entropie croisée", ylim=(0.80, 1.12))
        fig.savefig(output, dpi=200, metadata={"Software": "AI53 report renderer"})
        plt.close(fig)
    print(output.relative_to(root))


if __name__ == "__main__":
    main()
