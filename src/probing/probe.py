"""Train and evaluate linear probes on extracted representations.

Following the TalkTuner methodology:
- One-vs-rest logistic regression with L2 regularization
- Trained per-layer on 80/20 train/val split
- Reports accuracy per layer to identify where spatial information is encoded
"""

import json
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report


def train_probes(
    representations_path: str,
    output_dir: str = "results",
    C: float = 1.0,
    max_iter: int = 1000,
    train_split_path: Optional[str] = None,
    val_split_path: Optional[str] = None,
    train_ratio: float = 0.8,
    seed: int = 42,
) -> dict:
    """Train a linear probe per layer and evaluate.

    Args:
        representations_path: Path to .npz from extraction step.
        output_dir: Where to save results.
        C: Inverse L2 regularization strength.
        max_iter: Max iterations for logistic regression.
        train_split_path: Optional path to train.json (for consistent splits).
        val_split_path: Optional path to val.json.
        train_ratio: Fallback train ratio if no split files provided.
        seed: Random seed.

    Returns:
        Dict mapping layer_idx → {"accuracy": float, "report": str}
    """
    data = np.load(representations_path, allow_pickle=True)
    representations = data["representations"]  # (n_samples, n_layers, hidden_dim)
    labels_raw = data["labels"]                # (n_samples,)
    image_ids = data["image_ids"]              # (n_samples,)

    n_samples, n_layers, hidden_dim = representations.shape
    print(f"Data: {n_samples} samples, {n_layers} layers, {hidden_dim} hidden dim")

    # Encode labels
    le = LabelEncoder()
    labels = le.fit_transform(labels_raw)
    class_names = le.classes_
    print(f"Classes: {list(class_names)}")

    # Create train/val split
    if train_split_path and val_split_path:
        with open(train_split_path) as f:
            train_ids = set(s["image_id"] for s in json.load(f))
        with open(val_split_path) as f:
            val_ids = set(s["image_id"] for s in json.load(f))

        train_mask = np.array([iid in train_ids for iid in image_ids])
        val_mask = np.array([iid in val_ids for iid in image_ids])
    else:
        rng = np.random.RandomState(seed)
        indices = rng.permutation(n_samples)
        split = int(n_samples * train_ratio)
        train_mask = np.zeros(n_samples, dtype=bool)
        train_mask[indices[:split]] = True
        val_mask = ~train_mask

    y_train = labels[train_mask]
    y_val = labels[val_mask]
    print(f"Train: {train_mask.sum()}, Val: {val_mask.sum()}")

    # Train probe per layer
    results = {}
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    for layer_idx in range(n_layers):
        X_train = representations[train_mask, layer_idx, :]  # (n_train, hidden_dim)
        X_val = representations[val_mask, layer_idx, :]

        probe = LogisticRegression(
            C=C,
            penalty="l2",
            multi_class="ovr",  # one-vs-rest
            solver="lbfgs",
            max_iter=max_iter,
            random_state=seed,
        )
        probe.fit(X_train, y_train)

        y_pred = probe.predict(X_val)
        acc = accuracy_score(y_val, y_pred)
        report = classification_report(y_val, y_pred, target_names=class_names)

        results[layer_idx] = {
            "accuracy": float(acc),
            "report": report,
            "probe": probe,
        }
        print(f"Layer {layer_idx:2d}: accuracy = {acc:.4f}")

    # --- Save trained probes ---
    probes_dir = Path(output_dir) / "probes"
    probes_dir.mkdir(parents=True, exist_ok=True)

    # Save each layer's probe separately (easy to load just the layer you need)
    for layer_idx, res in results.items():
        probe_path = probes_dir / f"probe_layer_{layer_idx:03d}.joblib"
        joblib.dump(res["probe"], probe_path)

    # Save the label encoder (needed to map predictions back to class names)
    joblib.dump(le, probes_dir / "label_encoder.joblib")

    # Save a manifest so you know what's in this directory
    manifest = {
        "n_layers": n_layers,
        "hidden_dim": hidden_dim,
        "classes": list(class_names),
        "C": C,
        "best_layer": max(results, key=lambda k: results[k]["accuracy"]),
        "per_layer_accuracy": {
            str(k): v["accuracy"] for k, v in results.items()
        },
    }
    with open(probes_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"Saved {n_layers} probes + label encoder → {probes_dir}")

    # --- Save results summary ---
    results_summary = {
        "n_samples": n_samples,
        "n_layers": n_layers,
        "hidden_dim": hidden_dim,
        "classes": list(class_names),
        "C": C,
        "per_layer": {
            str(k): {"accuracy": v["accuracy"]} for k, v in results.items()
        },
    }
    results_path = Path(output_dir) / "probe_results.json"
    with open(results_path, "w") as f:
        json.dump(results_summary, f, indent=2)

    print(f"Results saved → {results_path}")
    return results


def load_probe(probes_dir: str, layer_idx: int) -> tuple[LogisticRegression, LabelEncoder]:
    """Load a saved probe and its label encoder.

    Args:
        probes_dir: Path to the probes/ directory created by train_probes.
        layer_idx: Which layer's probe to load.

    Returns:
        (probe, label_encoder) tuple. Use as:
            pred_idx = probe.predict(hidden_state)  # hidden_state: (1, hidden_dim)
            pred_label = le.inverse_transform(pred_idx)
    """
    probes_dir = Path(probes_dir)
    probe = joblib.load(probes_dir / f"probe_layer_{layer_idx:03d}.joblib")
    le = joblib.load(probes_dir / "label_encoder.joblib")
    return probe, le


def load_best_probe(probes_dir: str) -> tuple[LogisticRegression, LabelEncoder, int]:
    """Load the probe from the best-performing layer.

    Returns:
        (probe, label_encoder, layer_idx) tuple.
    """
    probes_dir = Path(probes_dir)
    with open(probes_dir / "manifest.json") as f:
        manifest = json.load(f)

    best_layer = manifest["best_layer"]
    probe, le = load_probe(str(probes_dir), best_layer)
    print(f"Loaded best probe: layer {best_layer} (acc={manifest['per_layer_accuracy'][str(best_layer)]:.4f})")
    return probe, le, best_layer


def predict_with_probe(
    probe: LogisticRegression,
    le: LabelEncoder,
    hidden_state: np.ndarray,
) -> dict:
    """Run a probe on a hidden state and return prediction + probabilities.

    Args:
        probe: Trained LogisticRegression probe.
        le: Label encoder from training.
        hidden_state: Shape (hidden_dim,) or (1, hidden_dim).

    Returns:
        {"prediction": str, "probabilities": {class_name: float}}
    """
    if hidden_state.ndim == 1:
        hidden_state = hidden_state.reshape(1, -1)

    pred_idx = probe.predict(hidden_state)[0]
    pred_label = le.inverse_transform([pred_idx])[0]
    probs = probe.predict_proba(hidden_state)[0]

    return {
        "prediction": pred_label,
        "probabilities": {
            le.inverse_transform([i])[0]: float(p)
            for i, p in enumerate(probs)
        },
    }