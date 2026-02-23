from __future__ import annotations

from pathlib import Path
import csv
import re
import sys
import inspect
from typing import Union, Dict, Any, Optional

import torch
from PIL import Image
import os

import sys
try:
    import psutil as ps3
    sys.modules["ps3"] = ps3
except Exception:
    pass

# =========================
# import path (VILA)
# =========================
REPO_ROOT = Path(__file__).resolve().parents[2]  # src/lasttoken -> src -> VLM_probing
VILA_ROOT = REPO_ROOT / "VILA"
sys.path.insert(0, str(VILA_ROOT))

from llava.model.builder import load_pretrained_model
from llava.mm_utils import tokenizer_image_token
from llava.constants import DEFAULT_IMAGE_TOKEN


# =========================
# prompt
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
# load model (once)
# =========================
import os

def load_specialrgbt(model_id: str):
    # Force-disable FlashAttention2 in Transformers
    os.environ["FLASH_ATTENTION_2"] = "0"
    os.environ["TRANSFORMERS_ATTENTION_IMPLEMENTATION"] = "sdpa"  # or "eager"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path=model_id,
        model_name=model_id,
        model_base=None,
        device=device,
        device_map="auto",
    )
    model.eval()
    return device, tokenizer, model, image_processor

# =========================
# single example
# =========================
@torch.no_grad()
def extract_specialrgbt_lasttoken_layers_from_ex(
    ex: Dict[str, Any],
    save_dir: Union[str, Path],
    device: str,
    tokenizer,
    model,
    image_processor,
    model_id: str,
    prompt_builder=build_prompt,
) -> Path:
    img_path = Path(ex["img_path"])
    obj1 = ex["subj"]
    obj2 = ex["obj"]
    rel = ex.get("rel", ex.get("relationship", None))

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    img = Image.open(img_path).convert("RGB")
    img_t = image_processor(img, return_tensors="pt")["pixel_values"][0].to(device)
    media = {"image": [img_t]}
    media_config = {"image": {}}

    prompt = prompt_builder(obj1, obj2)

    image_token_id = tokenizer.convert_tokens_to_ids(DEFAULT_IMAGE_TOKEN)
    assert isinstance(image_token_id, int) and image_token_id >= 0

    sig = inspect.signature(tokenizer_image_token)
    param_names = list(sig.parameters.keys())

    args = [prompt, tokenizer]
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
    input_ids = input_ids.to(device)

    attention_mask = torch.ones_like(input_ids, device=device)

    inputs_embeds, _, attention_mask = model._embed(
        input_ids=input_ids,
        media=media,
        media_config=media_config,
        labels=None,
        attention_mask=attention_mask,
    )

    out = model.llm.model(
        inputs_embeds=inputs_embeds,
        attention_mask=attention_mask,
        output_hidden_states=True,
        return_dict=True,
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
# CSV
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
    model_id: str = "a8cheng/SpatialRGPT-VILA1.5-8B",
    limit: Optional[int] = 10,          # set None for all rows
    skip_existing: bool = True,
):
    csv_path = Path(csv_path)
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    device, tokenizer, model, image_processor = load_specialrgbt(model_id)

    ok = 0
    skipped = 0
    errored = 0

    for i, ex in enumerate(iter_relationships_csv(csv_path), start=1):
        if limit is not None and i > limit:
            break

        if not ex["img_path"] or not ex["subj"] or not ex["obj"]:
            skipped += 1
            continue

        img_path = Path(ex["img_path"])
        if not img_path.exists():
            skipped += 1
            continue

        fname = f"{img_path.stem}__{_safe(ex['subj'])}__{_safe(ex['obj'])}.pt"
        out_path = save_dir / fname
        if skip_existing and out_path.exists():
            skipped += 1
            continue

        try:
            extract_specialrgbt_lasttoken_layers_from_ex(
                ex=ex,
                save_dir=save_dir,
                device=device,
                tokenizer=tokenizer,
                model=model,
                image_processor=image_processor,
                model_id=model_id,
            )
            ok += 1
        except torch.cuda.OutOfMemoryError as e:
            #print(f"[oom] row {i}: {e}")
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass
            errored += 1
        except Exception as e:
            #print(f"[error] row {i}: {e}")
            errored += 1

    with csv_path.open("r") as f:
        total_rows = sum(1 for _ in f) - 1  # exclude header
        
    n_report = total_rows if limit is None else min(limit, total_rows)

    print(f"Successfully saved {n_report} hidden states under {save_dir}!")
    #print(f"(ok={ok}, skipped={skipped}, errored={errored})")


# =========================
# main
# =========================
if __name__ == "__main__":
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    csv_path = PROJECT_ROOT / "data" / "vrd_relationships.csv"
    save_dir = PROJECT_ROOT / "features" / "SpecialRGBT"

    run_csv(
        csv_path=csv_path,
        save_dir=save_dir,
        model_id="a8cheng/SpatialRGPT-VILA1.5-8B",
        limit=10,           # None => all rows
        skip_existing=True,
    )