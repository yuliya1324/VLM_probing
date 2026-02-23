import torch
from transformers import LlavaForConditionalGeneration, LlavaProcessor, BitsAndBytesConfig
from PIL import Image

model_id = "llava-hf/llava-1.5-7b-hf"

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

# --- image ---
img_path = "/Data/masayo.tomita/VLM_probing/data/raw/vrd/sg_train_images/5516535415_0bae63e88d_b.jpg"
image = Image.open(img_path).convert("RGB")

# --- prompt ---
prompt = (
    "Determine the spatial relationship of 'truck' relative to 'bus'.\n"
    "Choose ONE label from:\n"
    "[left of, right of, above, below, in front of, behind]\n"
    "Respond with ONLY the label. No explanation."
)

prompt = f"USER: <image>\n{prompt}\nASSISTANT:"

inputs = processor(text=prompt, images=image, return_tensors="pt")

# move tensors to model device
device = model.get_input_embeddings().weight.device
inputs = {k: v.to(device) if torch.is_tensor(v) else v for k, v in inputs.items()}

# --- generation ---
with torch.no_grad():
    output_ids = model.generate(
        **inputs,
        max_new_tokens=30,
        do_sample=False,   # deterministic (good for probing)
        temperature=0.0,
    )

# --- decode ---
response = processor.batch_decode(
    output_ids,
    skip_special_tokens=True
)[0]

print(response)