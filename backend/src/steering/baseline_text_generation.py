import os
import json
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

from huggingface_hub import login


from dotenv import load_dotenv

load_dotenv()
login(token=os.environ["HF_TOKEN"])


base_dir = Path("data/steering/tests")
input_filename = "prompts_test_data.json"
output_filename = "generated_texts__baseline.json"
input_file = base_dir / input_filename
output_file = base_dir / output_filename

def load_text_generation_model(model_name: str, device: torch.device):
    """
    Load the HuggingFace model for text generation.
    
    Args:
        model_name: Model name/path
        device: Torch device
    """
    hf_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
    )
    hf_model = hf_model.to(device)
    hf_model.eval()
    hf_tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Ensure tokenizer has pad token
    if hf_tokenizer.pad_token is None:
        hf_tokenizer.pad_token = hf_tokenizer.eos_token

    return hf_model, hf_tokenizer



def read_test_data():
    with open(input_file, "r") as f:
        test_data = json.load(f)
    return test_data

def generate_text(prompt, hf_model, hf_tokenizer, device, max_new_tokens=500, temperature=0.7):
    prompt_ids = hf_tokenizer.encode(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        generated_ids = hf_model.generate(
            prompt_ids,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature
        )
    generated_text = hf_tokenizer.decode(generated_ids[0, prompt_ids.shape[1]:], skip_special_tokens=True)

    print(generated_text)
    print("-"*20)

    return generated_text

output_data = []

test_data = read_test_data()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
hf_model, hf_tokenizer = load_text_generation_model("google/gemma-2-9b-it", device)

for prompt in tqdm(test_data):
    prompt_text = prompt["prompt"]
    generated_text = generate_text(prompt_text, hf_model, hf_tokenizer, device)

    output_data.append({
        "prompt": prompt,
        "generated_text": generated_text,
        "author": prompt["author"],
        "original_article": prompt["article"],
    })
with open(output_file, "w") as f:
    json.dump(output_data, f, indent=2, ensure_ascii=False)
