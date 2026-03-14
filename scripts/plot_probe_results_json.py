"""Plot probe accuracy curves from probe_results.json files.

Expected JSON format:
{
  "n_samples": ...,
  "n_layers": ...,
  "classes": [...],
  "per_layer": {
    "0": {"accuracy": ...},
    "1": {"accuracy": ...},
    ...
  }
}

Usage:
    python scripts/plot_probe_results_json.py \
        results/vrd_spatial/qwen2/correct/probe_results.json \
        results/vrd_color/qwen2/correct/probe_results.json \
        results/vrd_shape/qwen2/correct/probe_results.json
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def load_probe_results(json_path: str) -> dict:
    json_path = Path(json_path)
    with open(json_path, "r") as f:
        data = json.load(f)

    n_layers = data["n_layers"]
    classes = data["classes"]
    per_layer = data["per_layer"]

    layer_accuracies = []
    for i in range(n_layers):
        key = str(i)
        if key not in per_layer or "accuracy" not in per_layer[key]:
            raise ValueError(f"Missing accuracy for layer {i} in {json_path}")
        layer_accuracies.append(float(per_layer[key]["accuracy"]))

    best_layer = max(range(n_layers), key=lambda i: layer_accuracies[i])
    best_accuracy = layer_accuracies[best_layer]

    return {
        "json_path": str(json_path),
        "n_samples": data.get("n_samples"),
        "n_layers": n_layers,
        "classes": classes,
        "layer_accuracies": layer_accuracies,
        "best_layer": best_layer,
        "best_accuracy": best_accuracy,
    }


def plot_single_result(results: dict, output_path: str | None = None):
    accs = results["layer_accuracies"]
    n_layers = results["n_layers"]
    classes = results["classes"]

    layers = list(range(n_layers))
    n_classes = len(classes)
    chance = 1.0 / n_classes if n_classes > 0 else 0.0

    fig, ax = plt.subplots(figsize=(10, 5))

    ax.plot(layers, accs, marker="o", markersize=3, label="accuracy")

    best_l = results["best_layer"]
    best_a = results["best_accuracy"]
    ax.annotate(
        f"L{best_l} ({best_a:.3f})",
        xy=(best_l, best_a),
        xytext=(best_l + 0.8, max(best_a - 0.05, 0.02)),
        fontsize=8,
        color="gray",
        arrowprops=dict(arrowstyle="->", color="gray", lw=0.8),
    )

    ax.axhline(
        y=chance,
        color="red",
        linestyle="--",
        alpha=0.5,
        label=f"chance ({chance:.2f}, {n_classes} classes)",
    )

    title = Path(results["json_path"]).parent.as_posix()
    ax.set_title(title)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0, 1.05)
    ax.grid(alpha=0.3)
    ax.legend()

    plt.tight_layout()

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150)
        print(f"Saved plot → {output_path}")
    else:
        plt.show()

    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Plot probe_results.json files")
    parser.add_argument(
        "json_files",
        nargs="+",
        help="Path(s) to probe_results.json",
    )
    parser.add_argument(
        "--output_name",
        type=str,
        default="probe_results.png",
        help="Output filename saved in the same folder as each JSON",
    )
    args = parser.parse_args()

    for json_file in args.json_files:
        results = load_probe_results(json_file)
        output_path = Path(json_file).parent / args.output_name
        plot_single_result(results, output_path)


if __name__ == "__main__":
    main()