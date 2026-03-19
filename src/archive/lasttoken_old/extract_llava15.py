from __future__ import annotations

from pathlib import Path
import csv
import re
from typing import Union, Dict, Any, Optional, Tuple

import torch
from transformers import LlavaForConditionalGeneration, LlavaProcessor, BitsAndBytesConfig
from PIL import Image


# =========================
# Prompt
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
def load_llava_4bit(model_id: str):
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )

    model = LlavaForConditionalGeneration.from_pretrained(
        model_id,
        quantization_config=bnb,
        device_map="auto",
        low_cpu_mem_usage=True,
    )
    model.eval()

    processor = LlavaProcessor.from_pretrained(model_id)
    return model, processor


# =========================
# Single example extraction
# =========================
@torch.no_grad()
def extract_llava_lasttoken_layers_from_ex(
    ex: Dict[str, Any],
    save_dir: Union[str, Path],
    model,
    processor,
    model_id: str,
    prompt_builder=build_prompt,
) -> Path:
    """
    ex: {"img_path": Path/str, "subj": str, "obj": str, "relationship": str}  (or "rel")
    Saves: <image_stem>__<subj>__<obj>.pt
    """

    img_path = Path(ex["img_path"])
    obj1 = ex["subj"]
    obj2 = ex["obj"]
    rel = ex.get("rel", ex.get("relationship", None))

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    image = Image.open(img_path).convert("RGB")
    prompt = prompt_builder(obj1, obj2)
    inputs = processor(text=prompt, images=image, return_tensors="pt")

    device = model.get_input_embeddings().weight.device
    inputs = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in inputs.items()}

    out = model(
        **inputs,
        output_hidden_states=True,
        output_attentions=False,
        use_cache=False,
        return_dict=True,
    )

    hs = out.hidden_states  # tuple: (embeddings + each layer)
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
# CSV runner
# =========================
def iter_relationships_csv(csv_path: Union[str, Path]):
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
    model_id: str = "llava-hf/llava-1.5-7b-hf",
    limit: Optional[int] = None,
    skip_existing: bool = True,
):
    """
    Iterate CSV and save features.
    - limit=10 => head(10)
    - limit=None => all rows
    """
    
    model, processor = load_llava_4bit(model_id)

    ok = 0
    skipped = 0
    errored = 0

    for i, ex in enumerate(iter_relationships_csv(csv_path), start=1):
        if limit is not None and i > limit:
            break

        # validation
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
            out_path = extract_llava_lasttoken_layers_from_ex(
                ex=ex,
                save_dir=save_dir,
                model=model,
                processor=processor,
                model_id=model_id,
            )
            #print(f"[ok] row {i} saved: {out_path}")
            ok += 1
        except Exception as e:
            print(f"[error] row {i}: {e}")
            errored += 1

    # ---- final summary ----
    with csv_path.open("r") as f:
        total_rows = sum(1 for _ in f) - 1  # exclude header

    n_report = total_rows if limit is None else min(limit, total_rows)

    print(f"Successfully saved {n_report} hidden states under {save_dir}!")

# =========================
# CLI-ish usage
# =========================
if __name__ == "__main__":
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    csv_path = PROJECT_ROOT / "data" / "vrd_csv" / "vrd_spatial.csv"
    save_dir = PROJECT_ROOT / "features" / "LLaVA"

    run_csv(
        csv_path=csv_path,
        save_dir=save_dir,
        model_id="llava-hf/llava-1.5-7b-hf",
        limit=10,          # None => all rows
        skip_existing=True,
    )