#!/usr/bin/env python3
"""Steer a VLM's spatial reasoning using trained probe directions.

Usage:
    # Steer toward "left_of" on a single image
    python scripts/steer.py \
        --probes_dir results/qwen2_spatial/probes \
        --image_path data/raw/spatial/images/spatial_00042.png \
        --prompt "Where is the red circle relative to the blue square?" \
        --model_tag qwen2 \
        --layer 20 \
        --target left_of \
        --alpha 10

    # Contrast steering: push toward left_of, away from right_of
    python scripts/steer.py \
        --probes_dir results/qwen2_spatial/probes \
        --image_path data/raw/spatial/images/spatial_00042.png \
        --prompt "Where is the red circle relative to the blue square?" \
        --model_tag qwen2 \
        --layer 20 \
        --target left_of \
        --source right_of \
        --alpha 10

    # Sweep across alpha values
    python scripts/steer.py \
        --probes_dir results/qwen2_spatial/probes \
        --image_path data/raw/spatial/images/spatial_00042.png \
        --prompt "Where is the red circle relative to the blue square?" \
        --model_tag qwen2 \
        --layer 20 \
        --target left_of \
        --sweep
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from PIL import Image
from src.extraction.extract import MODEL_REGISTRY
from src.steering.steer import steer_and_generate, sweep_alpha


def main():
    parser = argparse.ArgumentParser(description="Steer VLM spatial reasoning")

    # Model
    parser.add_argument("--model_tag", type=str, required=True)
    parser.add_argument("--model_id", type=str, default=None)

    # Input
    parser.add_argument("--image_path", type=str, required=True)
    parser.add_argument("--prompt", type=str, required=True)

    # Probe / steering
    parser.add_argument("--probes_dir", type=str, required=True)
    parser.add_argument("--layers", type=int, nargs="+", required=True,
                        help="Layer(s) to steer on. Single: --layers 20. Multi: --layers 15 18 20 22 25")
    parser.add_argument("--target", type=str, required=True, help="Target class to steer toward")
    parser.add_argument("--source", type=str, default=None, help="Source class to steer away from (contrast)")
    parser.add_argument("--alpha", type=float, default=10.0)
    parser.add_argument("--strategy", type=str, default="push", choices=["push", "contrast"])
    parser.add_argument("--token_position", type=str, default="all", choices=["all", "last"])
    parser.add_argument("--when", type=str, default="all", choices=["all", "prefill"],
                        help="'all': steer on every forward (prompt + generation). "
                             "'prefill': steer only during prompt processing.")

    # Sweep mode
    parser.add_argument("--sweep", action="store_true", help="Sweep alpha values")
    parser.add_argument("--alphas", type=float, nargs="*",
                        default=[0, 1, 2, 5, 10, 20, 50])

    parser.add_argument("--max_new_tokens", type=int, default=50)
    parser.add_argument("--output_json", type=str, default=None, help="Save results to JSON")

    args = parser.parse_args()

    # Auto-select strategy
    if args.source and args.strategy == "push":
        args.strategy = "contrast"

    # Load model
    registry_entry = MODEL_REGISTRY[args.model_tag]
    model_id = args.model_id or registry_entry["default_id"]
    print(f"Loading model: {model_id}")
    model, processor = registry_entry["loader"](model_id)

    image = Image.open(args.image_path).convert("RGB")

    if args.sweep:
        print(f"\nSweeping α values on layers {args.layers}, target={args.target}, when={args.when}")
        print("=" * 60)
        results = sweep_alpha(
            model, processor, args.model_tag,
            image, args.prompt,
            args.probes_dir, args.layers,
            target_class=args.target,
            source_class=args.source,
            alphas=args.alphas,
            strategy=args.strategy,
            when=args.when,
            max_new_tokens=args.max_new_tokens,
        )

        if args.output_json:
            with open(args.output_json, "w") as f:
                json.dump(results, f, indent=2)
            print(f"\nSaved → {args.output_json}")

    else:
        print(f"\nSteering: layers={args.layers}, target={args.target}, α={args.alpha}, when={args.when}")
        print("=" * 60)
        result = steer_and_generate(
            model, processor, args.model_tag,
            image, args.prompt,
            probes_dir=args.probes_dir,
            layers=args.layers,
            target_class=args.target,
            source_class=args.source,
            alpha=args.alpha,
            strategy=args.strategy,
            token_position=args.token_position,
            when=args.when,
            max_new_tokens=args.max_new_tokens,
        )

        print(f"\nBaseline:  {result['baseline_output']}")
        print(f"Steered:   {result['steered_output']}")
        print(f"\n  target={result['target_class']}, α={result['alpha']}, "
              f"layers={result['layers']}, strategy={result['strategy']}, when={result['when']}")

        if args.output_json:
            with open(args.output_json, "w") as f:
                json.dump(result, f, indent=2)


if __name__ == "__main__":
    main()