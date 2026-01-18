import os
import json
from pathlib import Path
import torch
import torch.multiprocessing as mp
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
from huggingface_hub import login
from dotenv import load_dotenv
import tempfile

# Disable torch.compile/dynamo to avoid recompilation issues with varying input lengths
# This is necessary because Gemma-2 models trigger dynamo and varying prompt lengths
# cause excessive recompilations
torch._dynamo.config.cache_size_limit = 256  # Increase cache size
torch._dynamo.config.suppress_errors = True


def setup_environment():
    """Load environment variables and login to HuggingFace."""
    load_dotenv()
    login(token=os.environ["HF_TOKEN"])


base_dir = Path("data/steering/tests")
input_filename = "prompts_train_data__detailed.json"
output_filename = "generated_training_texts__baseline__detailed.json"
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
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        attn_implementation="eager",  # Use eager attention to avoid dynamo recompilations
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


def load_existing_output() -> tuple[list, set]:
    """
    Load existing output file if it exists.
    
    Returns:
        Tuple of (existing_data, existing_ids_set)
    """
    if output_file.exists():
        with open(output_file, "r") as f:
            existing_data = json.load(f)
        existing_ids = {item["id"] for item in existing_data}
        print(f"Loaded {len(existing_data)} existing results from {output_file}")
        return existing_data, existing_ids
    return [], set()


def generate_text(prompt, hf_model, hf_tokenizer, device, max_new_tokens=500, temperature=0.7, model_id="google/gemma-2-9b-it"):
    """Generate text from a prompt using the loaded model."""
    
    if model_id == "Qwen/Qwen3-14B":  
        messages = [
            {"role": "user", "content": prompt}
        ]
        text = hf_tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False  # Switches between thinking and non-thinking modes. Default is True.
        )
        model_inputs = hf_tokenizer([text], return_tensors="pt").to(device)
    else:
        model_inputs = hf_tokenizer(prompt, return_tensors="pt").to(device)
    
    with torch.no_grad():
        generated_ids = hf_model.generate(
            **model_inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature
        )

    output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist() 
    generated_text = hf_tokenizer.decode(output_ids, skip_special_tokens=True)

    return generated_text


def split_data_into_chunks(data: list, num_chunks: int) -> list:
    """
    Split data into contiguous chunks for multi-GPU processing.
    
    Strategy: Contiguous chunk splitting
    - Each GPU gets a contiguous block of data
    - Preserves original indices for easy merging
    - No inter-process communication needed
    - Optimal for independent text generation tasks
    
    Args:
        data: List of items to split
        num_chunks: Number of chunks (typically number of GPUs)
    
    Returns:
        List of (chunk_data, original_indices) tuples
    """
    total_items = len(data)
    chunk_size = total_items // num_chunks
    remainder = total_items % num_chunks
    
    chunks = []
    start_idx = 0
    
    for i in range(num_chunks):
        # Distribute remainder items across first 'remainder' chunks
        extra = 1 if i < remainder else 0
        end_idx = start_idx + chunk_size + extra
        
        chunk_data = data[start_idx:end_idx]
        original_indices = list(range(start_idx, end_idx))
        chunks.append((chunk_data, original_indices))
        
        start_idx = end_idx
    
    return chunks


def worker_process(rank: int, world_size: int, model_id: str, test_data: list, 
                   data_indices: list, temp_dir: str, existing_ids: set = None,
                   checkpoint_interval: int = 100):
    """
    Worker function for each GPU process.
    
    Args:
        rank: GPU rank (0 to world_size-1)
        world_size: Total number of GPUs
        model_id: HuggingFace model identifier
        test_data: List of prompts for this GPU
        data_indices: Original indices corresponding to test_data
        temp_dir: Directory to save temporary results
        existing_ids: Set of IDs already processed (for continue_generation)
        checkpoint_interval: How often to save intermediate results
    """
    if existing_ids is None:
        existing_ids = set()
    
    # Set the device for this process
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)
    
    # Filter out already processed items
    items_to_process = [
        (prompt, global_idx) 
        for prompt, global_idx in zip(test_data, data_indices)
        if prompt.get("id", global_idx) not in existing_ids
    ]
    
    total_items = len(test_data)
    skipped_items = total_items - len(items_to_process)
    
    print(f"[GPU {rank}] Starting worker with {len(items_to_process)} prompts on {device} "
          f"(skipped {skipped_items} already processed)")
    
    if len(items_to_process) == 0:
        print(f"[GPU {rank}] All items already processed, exiting")
        # Save empty results file
        temp_output_file = Path(temp_dir) / f"results_gpu_{rank}.json"
        with open(temp_output_file, "w") as f:
            json.dump([], f)
        return
    
    # Load model on this specific GPU
    hf_model, hf_tokenizer = load_text_generation_model(model_id, device)
    print(f"[GPU {rank}] Model loaded successfully")
    
    output_data = []
    temp_output_file = Path(temp_dir) / f"results_gpu_{rank}.json"
    
    for local_idx, (prompt, global_idx) in enumerate(tqdm(
        items_to_process, 
        total=len(items_to_process),
        desc=f"GPU {rank}",
        position=rank
    )):
        prompt_text = prompt["prompt"]
        prompt_id = prompt.get("id", global_idx)
        
        try:
            generated_text = generate_text(
                prompt_text, hf_model, hf_tokenizer, device, model_id=model_id
            )
        except Exception as e:
            print(f"[GPU {rank}] Error generating text for index {global_idx}: {e}")
            generated_text = f"ERROR: {str(e)}"
        
        output_data.append({
            "id": prompt_id,
            "prompt": prompt_text,
            "generated_text": generated_text,
            "author": prompt["author"],
            "original_article": prompt["article"],
        })
        
        # Checkpoint: save intermediate results
        if (local_idx + 1) % checkpoint_interval == 0:
            with open(temp_output_file, "w") as f:
                json.dump(output_data, f, indent=2, ensure_ascii=False)
            print(f"[GPU {rank}] Checkpoint saved at {local_idx + 1}/{len(items_to_process)}")
    
    # Final save
    with open(temp_output_file, "w") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    print(f"[GPU {rank}] Completed processing {len(output_data)} prompts")
    
    # Clean up GPU memory
    del hf_model
    del hf_tokenizer
    torch.cuda.empty_cache()


def merge_results(temp_dir: str, num_gpus: int) -> list:
    """
    Merge results from all GPU workers.
    
    Args:
        temp_dir: Directory containing temporary result files
        num_gpus: Number of GPU workers
    
    Returns:
        Merged and sorted results list
    """
    all_results = []
    
    for rank in range(num_gpus):
        temp_file = Path(temp_dir) / f"results_gpu_{rank}.json"
        if temp_file.exists():
            with open(temp_file, "r") as f:
                results = json.load(f)
                all_results.extend(results)
            print(f"Merged {len(results)} results from GPU {rank}")
        else:
            print(f"Warning: Missing results file for GPU {rank}")
    
    # Sort by original index to maintain order
    all_results.sort(key=lambda x: x["id"])
    
    return all_results


def run_multiprocessing(model_id: str, num_gpus: int = None, continue_generation: bool = False):
    """
    Main function to run multi-GPU text generation.
    
    Args:
        model_id: HuggingFace model identifier
        num_gpus: Number of GPUs to use (default: all available)
        continue_generation: If True, load existing output and skip already processed items
    """
    # Determine number of GPUs
    if num_gpus is None:
        num_gpus = torch.cuda.device_count()
    
    if num_gpus == 0:
        raise RuntimeError("No CUDA GPUs available. Please check your setup.")
    
    print(f"Using {num_gpus} GPU(s) for text generation")
    
    # Load existing results if continuing
    existing_data = []
    existing_ids = set()
    if continue_generation:
        existing_data, existing_ids = load_existing_output()
        if existing_ids:
            print(f"Continue mode: will skip {len(existing_ids)} already processed items")
    
    # Load test data
    test_data = read_test_data()
    print(f"Loaded {len(test_data)} prompts")
    
    # Split data into chunks for each GPU
    chunks = split_data_into_chunks(test_data, num_gpus)
    
    for i, (chunk, indices) in enumerate(chunks):
        print(f"GPU {i}: {len(chunk)} prompts (indices {indices[0]} to {indices[-1]})")
    
    # Create temporary directory for intermediate results
    with tempfile.TemporaryDirectory() as temp_dir:
        print(f"Temporary directory: {temp_dir}")
        
        # Spawn processes for each GPU
        # Using spawn context for CUDA compatibility
        # mp.spawn(
        #     worker_process,
        #     args=(
        #         num_gpus,
        #         model_id,
        #         chunks[0][0],  # Will be overridden in actual call
        #         chunks[0][1],
        #         temp_dir,
        #     ),
        #     nprocs=num_gpus,
        #     join=False  # Don't join yet
        # )
        
        # Actually, we need a different approach - spawn expects same args for all
        # Let's use Process directly for more control
        processes = []
        
        for rank in range(num_gpus):
            chunk_data, chunk_indices = chunks[rank]
            p = mp.Process(
                target=worker_process,
                args=(rank, num_gpus, model_id, chunk_data, chunk_indices, temp_dir, existing_ids)
            )
            p.start()
            processes.append(p)
        
        # Wait for all processes to complete
        for p in processes:
            p.join()
        
        # Check for failures
        for rank, p in enumerate(processes):
            if p.exitcode != 0:
                print(f"Warning: GPU {rank} process exited with code {p.exitcode}")
        
        # Merge results from all GPUs
        print("\nMerging results from all GPUs...")
        new_results = merge_results(temp_dir, num_gpus)
    
    # Combine existing and new results
    if continue_generation and existing_data:
        # Create a dict for fast lookup and merge
        results_dict = {item["id"]: item for item in existing_data}
        for item in new_results:
            results_dict[item["id"]] = item
        final_results = sorted(results_dict.values(), key=lambda x: x["id"])
        print(f"Merged {len(new_results)} new results with {len(existing_data)} existing results")
    else:
        final_results = new_results
    
    # Save final merged results
    with open(output_file, "w") as f:
        json.dump(final_results, f, indent=2, ensure_ascii=False)
    
    print(f"\nFinal results saved to {output_file}")
    print(f"Total prompts in output: {len(final_results)}")
    
    return final_results


def run_single_gpu(model_id: str, continue_generation: bool = False):
    """
    Fallback function to run on a single GPU (original behavior).
    
    Args:
        model_id: HuggingFace model identifier
        continue_generation: If True, load existing output and skip already processed items
    """
    # Load existing results if continuing
    existing_data = []
    existing_ids = set()
    if continue_generation:
        existing_data, existing_ids = load_existing_output()
        if existing_ids:
            print(f"Continue mode: will skip {len(existing_ids)} already processed items")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    test_data = read_test_data()
    
    # Filter out already processed items
    items_to_process = [
        (index, prompt) for index, prompt in enumerate(test_data)
        if prompt.get("id", index) not in existing_ids
    ]
    
    skipped_count = len(test_data) - len(items_to_process)
    print(f"Loaded {len(test_data)} prompts, processing {len(items_to_process)} (skipping {skipped_count} already done)")
    
    if len(items_to_process) == 0:
        print("All items already processed!")
        return existing_data
    
    # Only load model if we have items to process
    hf_model, hf_tokenizer = load_text_generation_model(model_id, device)
    
    output_data = list(existing_data)  # Start with existing data
    processed_count = 0
    
    for index, prompt in tqdm(items_to_process, total=len(items_to_process)):
        prompt_text = prompt["prompt"]
        prompt_id = prompt.get("id", index)
        
        generated_text = generate_text(prompt_text, hf_model, hf_tokenizer, device, model_id=model_id)

        output_data.append({
            "id": prompt_id,
            "prompt": prompt_text,
            "generated_text": generated_text,
            "author": prompt["author"],
            "original_article": prompt["article"],
        })

        print(generated_text[:100])
        processed_count += 1

        if processed_count % 100 == 0:
            # Sort by id before saving to maintain order
            output_data_sorted = sorted(output_data, key=lambda x: x["id"])
            with open(output_file, "w") as f:
                json.dump(output_data_sorted, f, indent=2, ensure_ascii=False)

    # Final save, sorted by id
    output_data_sorted = sorted(output_data, key=lambda x: x["id"])
    with open(output_file, "w") as f:
        json.dump(output_data_sorted, f, indent=2, ensure_ascii=False)
    
    print(f"Processed {processed_count} new items, total in output: {len(output_data_sorted)}")
    
    return output_data_sorted


if __name__ == "__main__":
    # IMPORTANT: Set spawn method for CUDA multiprocessing
    # Must be called before any CUDA operations
    mp.set_start_method("spawn", force=True)
    
    # Setup environment (HuggingFace login, etc.)
    setup_environment()
    
    # Configuration
    model_id = "google/gemma-2-9b-it" #"Qwen/Qwen3-14B"
    num_gpus = torch.cuda.device_count()
    continue_generation = True  # Set to True to resume from existing output
    
    print(f"Available GPUs: {num_gpus}")
    for i in range(num_gpus):
        print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
    
    print(f"Continue generation: {continue_generation}")
    
    if num_gpus > 1:
        # Multi-GPU mode
        print("\n" + "="*50)
        print("Running in MULTI-GPU mode")
        print("="*50 + "\n")
        run_multiprocessing(model_id, num_gpus=num_gpus, continue_generation=continue_generation)
    else:
        # Single GPU fallback
        print("\n" + "="*50)
        print("Running in SINGLE-GPU mode")
        print("="*50 + "\n")
        run_single_gpu(model_id, continue_generation=continue_generation)
