from __future__ import annotations

from pathlib import Path
import csv
import re
from typing import Union, Dict, Any, Optional

import torch
from PIL import Image
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor


# =========================
# prompt (same spec)
# =========================
LLaVA_SPATIAL_PROMPT = (
    "Determine the spatial relationship of '{obj1}' relative to '{obj2}'.\n"
    "Choose ONE label from:\n"
    "[left of, right of, above, below, in front of, behind]\n"
    "Respond with ONLY the label. No explanation."
)
LLaVA_CHAT_WRAPPER = "USER: <image>\n{instruction}\nASSISTANT:"


def build_prompt(obj1: str, obj2: str) -> str:
    instruction = LLaVA_SPATIAL_PROMPT.format(obj1=obj1, obj2=obj2)
    return LLaVA_CHAT_WRAPPER.format(instruction=instruction)


def _safe(s: str) -> str:
    s = str(s).strip().lower()
    s = re.sub(r"\s+", "_", s)
    s = re.sub(r"[^a-zA-Z0-9_\-\.]+", "", s)
    return s[:80] if len(s) > 80 else s


# =========================
# Model loading (ONCE)
# =========================
def load_qwen2vl(model_id: str):
    """
    Load model + processor ONCE.
    Note: device_map="auto" may shard across devices.
    """
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_id,
        device_map="auto",
        torch_dtype="auto",
    )
    model.eval()
    processor = AutoProcessor.from_pretrained(model_id)
    return model, processor


# =========================
# Single example extraction (REUSE model/processor)
# =========================
@torch.no_grad()
def extract_qwen2vl_lasttoken_layers_from_ex(
    ex: Dict[str, Any],
    save_dir: Union[str, Path],
    model,
    processor,
    model_id: str,
    prompt_builder=build_prompt,
) -> Path:
    """
    ex: {"img_path": Path/str, "subj": str, "obj": str, "relationship": str} (or "rel")
    Saves: <image_stem>__<subj>__<obj>.pt
    Extracts: last-token hidden state for ALL layers (embeddings + each layer).
    """

    img_path = Path(ex["img_path"])
    obj1 = ex["subj"]
    obj2 = ex["obj"]
    rel = ex.get("rel", ex.get("relationship", None))

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    prompt = prompt_builder(obj1, obj2)

    # Qwen2-VL: messages format
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": str(img_path)},
                {"type": "text", "text": prompt},
            ],
        }
    ]

    # Convert messages to text with chat template
    text = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=False,
    )

    image = Image.open(img_path).convert("RGB")
    inputs = processor(
        text=[text],
        images=[image],
        return_tensors="pt",
    )

    # With device_map="auto", best effort: put inputs on embedding device
    # (works for many setups; if your model is heavily sharded, keeping inputs on cpu may be safer,
    # but this usually works for Qwen2-VL on a single GPU.)
    try:
        target_device = model.get_input_embeddings().weight.device
    except Exception:
        target_device = next(model.parameters()).device

    inputs = {k: (v.to(target_device) if torch.is_tensor(v) else v) for k, v in inputs.items()}

    out = model(
        **inputs,
        output_hidden_states=True,
        return_dict=True,
        use_cache=False,
    )

    hs = out.hidden_states
    layers = {l: hs[l][0, -1].detach().cpu() for l in range(len(hs))}

    fname = f"{img_path.stem}__{_safe(obj1)}__{_safe(obj2)}.pt"
    save_path = save_dir / fname

    payload = {
        "layers": layers,
        "meta": {
            "img_path": str(img_path),
            "obj1": obj1,
            "obj2": obj2,
            "rel": rel,
            "prompt": prompt,
            "model_id": model_id,
        },
    }
    torch.save(payload, save_path)
    return save_path


# =========================
# CSV runner (same as LLaVA style)
# =========================
def iter_relationships_csv(csv_path: Union[str, Path]):
    """
    CSV header is handled by DictReader.
    Expected columns: img_path, subj, obj, relationship
    """
    csv_path = Path(csv_path)
    with csv_path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if not row:
                continue
            yield {
                "img_path": (row.get("img_path") or "").strip(),
                "subj": (row.get("subj") or "").strip(),
                "obj": (row.get("obj") or "").strip(),
                "relationship": (row.get("relationship") or "").strip(),
            }


def run_csv(
    csv_path: Union[str, Path],
    save_dir: Union[str, Path],
    model_id: str = "Qwen/Qwen2-VL-7B-Instruct",
    limit: Optional[int] = None,   # None => all rows
    skip_existing: bool = True,
):
    """
    Iterate CSV and save features.
    - limit=10 => head(10)
    - limit=None => all rows
    """

    model, processor = load_qwen2vl(model_id)

    ok = 0
    skipped = 0
    errored = 0

    for i, ex in enumerate(iter_relationships_csv(csv_path), start=1):
        if limit is not None and i > limit:
            break

        if not ex["img_path"] or not ex["subj"] or not ex["obj"]:
            #print(f"[skip] row {i}: missing fields -> {ex}")
            skipped += 1
            continue

        img_path = Path(ex["img_path"])
        if not img_path.exists():
            #print(f"[skip] row {i}: image not found -> {img_path}")
            skipped += 1
            continue

        fname = f"{img_path.stem}__{_safe(ex['subj'])}__{_safe(ex['obj'])}.pt"
        save_path = Path(save_dir) / fname
        if skip_existing and save_path.exists():
            #print(f"[skip] row {i}: exists -> {save_path}")
            skipped += 1
            continue

        try:
            out_path = extract_qwen2vl_lasttoken_layers_from_ex(
                ex=ex,
                save_dir=save_dir,
                model=model,
                processor=processor,
                model_id=model_id,
            )
            #print(f"[ok] row {i} saved: {out_path}")
            ok += 1
        except torch.cuda.OutOfMemoryError as e:
            # robust mode for full CSV runs
            print(f"[oom] row {i}: {e} -> clearing cache and continuing")
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass
            errored += 1
        except Exception as e:
            print(f"[error] row {i}: {e}")
            errored += 1

    #print(f"done. ok={ok} skipped={skipped} errored={errored}")
    # ---- final summary ----
    with csv_path.open("r") as f:
        total_rows = sum(1 for _ in f) - 1  # exclude header

    n_report = total_rows if limit is None else min(limit, total_rows)

    print(f"Successfully saved {n_report} hidden states under {save_dir}!")

if __name__ == "__main__":
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    csv_path = PROJECT_ROOT / "data" / "vrd_relationships.csv"
    save_dir = PROJECT_ROOT / "features" / "Qwen2-VL"

    run_csv(
        csv_path=csv_path,
        save_dir=save_dir,
        model_id="Qwen/Qwen2-VL-7B-Instruct",
        limit=10,          # None => all rows
        skip_existing=True,
    )