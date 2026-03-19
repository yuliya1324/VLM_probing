"""
Evaluate raw VLM response accuracy on VRD CSVs.

This script:
  1. loads image + prompt from VRD CSV
  2. generates a raw response from the VLM
  3. parses the response into a task label
  4. compares it with ground truth
  5. saves per-sample CSV and prints accuracy

Usage:
    python scripts/evaluate_vrd_raw.py --task color   --model_tag qwen2
    python scripts/evaluate_vrd_raw.py --task shape   --model_tag qwen2
    python scripts/evaluate_vrd_raw.py --task spatial --model_tag qwen2
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm
import traceback

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.extraction.extract import MODEL_REGISTRY, INPUT_BUILDERS
from scripts.extract_vrd import (
    TASK_TO_CSV,
    TASK_TO_RESULT_DIR,
    build_prompt,
    get_image_path,
    get_label,
    get_sample_id,
)

# ============================================================
# Label normalization / parsing
# ============================================================
ALIAS_TO_CANONICAL = {
    "spatial": {
        "left_of": "left of",
        "right_of": "right of",
        "left of": "left of",
        "right of": "right of",
        "above": "above",
        "below": "below",
        "left": "left of",
        "right": "right of",
        "to the left of": "left of",
        "to the right of": "right of",
        "on the left of": "left of",
        "on the right of": "right of",
        "under": "below",
        "beneath": "below",
        "underneath": "below",
        "over": "above",
        "on top of": "above",
    },
    "color": {
        "black": "black",
        "blue": "blue",
        "brown": "brown",
        "gray": "gray",
        "grey": "gray",
        "green": "green",
        "orange": "orange",
        "pink": "pink",
        "purple": "purple",
        "red": "red",
        "white": "white",
        "yellow": "yellow",
    },
    "shape": {
        "circular": "circular",
        "circle": "circular",
        "round": "circular",
        "oval": "oval",
        "rectangular": "rectangular",
        "rectangle": "rectangular",
        "square": "square",
        "triangular": "triangular",
        "triangle": "triangular",
    },
}


# ============================================================
# Helpers
# ============================================================
def normalize_text(x: str) -> str:
    x = str(x).strip().lower()
    x = re.sub(r"[^\w\s\-]", " ", x)
    x = re.sub(r"\s+", " ", x).strip()
    return x


def normalize_label(label: str, task: str) -> str:
    text = normalize_text(label)
    alias_map = ALIAS_TO_CANONICAL[task]

    if text in alias_map:
        return alias_map[text]

    for alias in sorted(alias_map.keys(), key=len, reverse=True):
        if re.search(rf"\b{re.escape(alias)}\b", text):
            return alias_map[alias]

    return text


def parse_answer(raw_response: str, task: str) -> str | None:
    text = normalize_text(raw_response)
    alias_map = ALIAS_TO_CANONICAL[task]

    for alias in sorted(alias_map.keys(), key=len, reverse=True):
        if re.search(rf"\b{re.escape(alias)}\b", text):
            return alias_map[alias]

    return None


def move_inputs_to_device(inputs, device):
    if hasattr(inputs, "to"):
        return inputs.to(device)
    return {
        k: (v.to(device) if torch.is_tensor(v) else v)
        for k, v in inputs.items()
    }


def flush_rows(rows: list[dict], output_csv: Path) -> None:
    if not rows:
        return

    out_df = pd.DataFrame(rows)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    if output_csv.exists():
        out_df.to_csv(output_csv, mode="a", header=False, index=False)
    else:
        out_df.to_csv(output_csv, index=False)
        
# ============================================================
# VILA raw generation helper
# Adapted from Iuliia's steering demo implementation
# (src/steering/steer.py::_generate_vila)
# ============================================================
@torch.inference_mode()
def generate_single_vila(
    model,
    processor_tuple,
    image: Image.Image,
    prompt: str,
    max_new_tokens: int,
) -> str:
    import inspect
    from llava.mm_utils import tokenizer_image_token
    from llava.constants import DEFAULT_IMAGE_TOKEN

    _, tokenizer, image_processor = processor_tuple

    # Use actual module devices instead of trusting processor_tuple[0]
    text_device = model.llm.model.embed_tokens.weight.device
    vision_device = next(model.get_vision_tower().parameters()).device

    image = image.convert("RGB")
    img_t = image_processor(image, return_tensors="pt")["pixel_values"][0].to(
        vision_device
    ).half()

    media = {"image": [img_t]}
    media_config = {"image": {}}

    chat_prompt = f"USER: <image>\n{prompt}\nASSISTANT:"

    image_token_id = tokenizer.convert_tokens_to_ids(DEFAULT_IMAGE_TOKEN)
    sig = inspect.signature(tokenizer_image_token)
    param_names = list(sig.parameters.keys())

    args = [chat_prompt, tokenizer]
    kwargs = {}
    if "return_tensors" in sig.parameters:
        kwargs["return_tensors"] = "pt"

    for name in ("image_token_index", "image_token", "image_token_id"):
        if name in sig.parameters:
            pos = param_names.index(name)
            while len(args) < pos:
                args.append(None)
            args.append(image_token_id)
            break

    input_ids = tokenizer_image_token(*args, **kwargs)
    if input_ids.dim() == 1:
        input_ids = input_ids.unsqueeze(0)
    input_ids = input_ids.to(text_device)

    attention_mask = torch.ones_like(input_ids, device=text_device)

    output_ids = model.generate(
        input_ids=input_ids,
        media=media,
        media_config=media_config,
        attention_mask=attention_mask,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

    input_len = input_ids.shape[1]
    text = tokenizer.decode(output_ids[0, input_len:], skip_special_tokens=True)
    return text.strip()
    
@torch.inference_mode()
def generate_single(
    model,
    processor,
    model_tag: str,
    image: Image.Image,
    prompt: str,
    max_new_tokens: int,
) -> str:
    if model_tag == "vila":
        return generate_single_vila(
            model=model,
            processor_tuple=processor,
            image=image,
            prompt=prompt,
            max_new_tokens=max_new_tokens,
        )

    if model_tag not in INPUT_BUILDERS:
        raise NotImplementedError(
            f"Raw generation is currently supported for: {list(INPUT_BUILDERS.keys()) + ['vila']}. "
            f"Got: {model_tag}"
        )

    try:
        device = model.get_input_embeddings().weight.device
    except Exception:
        device = next(model.parameters()).device

    prepare_fn = INPUT_BUILDERS[model_tag]
    inputs = prepare_fn(processor, prompt, image, device)
    inputs = move_inputs_to_device(inputs, device)

    input_len = inputs["input_ids"].shape[1]

    output_ids = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        use_cache=True,
    )

    generated_ids = output_ids[:, input_len:]
    raw_text = processor.batch_decode(
        generated_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=True,
    )[0].strip()

    del inputs, output_ids, generated_ids
    return raw_text


def print_summary(output_csv: Path) -> None:
    df = pd.read_csv(output_csv)

    n_rows = len(df)
    n_correct = int(df["correct"].sum()) if n_rows > 0 else 0
    accuracy = float(df["correct"].mean()) if n_rows > 0 else 0.0
    parse_rate = float(df["parsed_answer"].notna().mean()) if n_rows > 0 else 0.0

    print("\n===== Summary =====")
    print(f"Rows saved           : {n_rows}")
    print(f"Correct              : {n_correct}")
    print(f"Accuracy             : {accuracy:.4f}")
    print(f"Parse rate           : {parse_rate:.4f}")

    if n_rows > 0:
        print("\nPer-class accuracy:")
        print(
            df.groupby("ground_truth_eval")["correct"]
            .agg(["count", "mean"])
            .rename(columns={"mean": "accuracy"})
            .sort_values("count", ascending=False)
        )

    unparsable = int(df["parsed_answer"].isna().sum()) if n_rows > 0 else 0
    if unparsable > 0:
        print(f"\nUnparsable responses : {unparsable}")


# ============================================================
# Main
# ============================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, required=True, choices=["spatial", "color", "shape"])
    parser.add_argument("--model_tag", type=str, required=True, choices=["qwen2", "llava15", "vila"])
    parser.add_argument("--model_id", type=str, default=None)
    parser.add_argument("--csv_path", type=str, default=None)
    parser.add_argument("--output_csv", type=str, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max_new_tokens", type=int, default=6)
    parser.add_argument("--save_every", type=int, default=20)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    csv_path = Path(args.csv_path) if args.csv_path else TASK_TO_CSV[args.task]
    output_csv = (
        Path(args.output_csv)
        if args.output_csv
        else TASK_TO_RESULT_DIR[args.task] / args.model_tag / "raw_response_predictions.csv"
    )

    print(f"Task       : {args.task}")
    print(f"Model tag  : {args.model_tag}")
    print(f"CSV path   : {csv_path}")
    print(f"Output CSV : {output_csv}")

    df = pd.read_csv(csv_path)
    if args.limit is not None:
        df = df.iloc[:args.limit].copy()

    done_row_idxs = set()
    if args.resume and output_csv.exists():
        existing_df = pd.read_csv(output_csv)
        if "row_idx" in existing_df.columns:
            done_row_idxs = set(existing_df["row_idx"].tolist())
        print(f"Resume mode: found {len(done_row_idxs)} completed rows")

    registry_entry = MODEL_REGISTRY[args.model_tag]
    model_id = args.model_id or registry_entry["default_id"]

    print(f"Loading model: {model_id}")
    model, processor = registry_entry["loader"](model_id)
    model.eval()
    
    #
    if args.model_tag == "vila":
        print("processor type:", type(processor))
    if isinstance(processor, tuple):
        print("processor tuple len:", len(processor))
        for i, x in enumerate(processor):
            print(f"processor[{i}] type:", type(x))

    rows_to_save = []

    for row_idx, (_, row) in enumerate(tqdm(df.iterrows(), total=len(df), desc="Generating")):
        if args.resume and row_idx in done_row_idxs:
            continue

        try:
            image_path = get_image_path(row)
            image = Image.open(image_path).convert("RGB")

            prompt = build_prompt(row, args.task)
            gt_raw = str(get_label(row, args.task))
            gt_eval = normalize_label(gt_raw, args.task)

            raw_response = generate_single(
                model=model,
                processor=processor,
                model_tag=args.model_tag,
                image=image,
                prompt=prompt,
                max_new_tokens=args.max_new_tokens,
            )

            parsed_answer = parse_answer(raw_response, args.task)
            correct = parsed_answer == gt_eval if parsed_answer is not None else False

            rows_to_save.append(
                {
                    "row_idx": row_idx,
                    "sample_id": get_sample_id(row, args.task),
                    "image_path": str(image_path),
                    "prompt": prompt,
                    "ground_truth": gt_raw,
                    "ground_truth_eval": gt_eval,
                    "raw_response": raw_response,
                    "parsed_answer": parsed_answer,
                    "correct": bool(correct),
                }
            )

            if len(rows_to_save) >= args.save_every:
                flush_rows(rows_to_save, output_csv)
                rows_to_save = []

        except torch.cuda.OutOfMemoryError:
            print(f"\n[{row_idx}] OOM — skipping")
            torch.cuda.empty_cache()
        except Exception as e:
            print(f"\n[{row_idx}] error: {e} — skipping")
            traceback.print_exc()

    if rows_to_save:
        flush_rows(rows_to_save, output_csv)

    print_summary(output_csv)


if __name__ == "__main__":
    main()