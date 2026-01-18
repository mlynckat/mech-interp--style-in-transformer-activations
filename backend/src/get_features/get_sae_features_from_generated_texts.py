"""
Extract SAE features from generated texts and original articles.

This script processes texts from generated_training_texts__baseline.json,
groups them by author, and extracts SAE features for both generated_text 
and original_article fields.

Features are extracted with the prompt prepended, but only activations
for the text (without prompt) are saved.
"""
import os
import argparse
import psutil
import gc
import traceback
import json
import logging
import time
from typing import List, Dict, Tuple
from pathlib import Path
from collections import defaultdict

from tqdm.auto import tqdm
import numpy as np
import scipy.sparse as sp

from dotenv import load_dotenv
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from huggingface_hub import login
from sae_lens import SAE, HookedSAETransformer

from backend.src.utils.run_configuration import ModelConfig, SAELayerConfig
from backend.src.utils.shared_utilities import ActivationMetadata, StorageFormat
from backend.src.get_features.get_sae_features import (
    ActivationStorage, 
    load_model, 
    load_saes, 
    log_gpu_memory_usage,
    lm_cross_entropy_loss,
    compute_entropy
)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

load_dotenv()
login(token=os.environ["HF_TOKEN"])


# Default paths
DEFAULT_INPUT_FILE = Path("data/steering/tests/generated_training_texts__baseline.json")
DEFAULT_OUTPUT_DIR = Path("data/raw_features/generated_texts")


def load_generated_texts(input_file: Path) -> Dict[str, List[Dict]]:
    """
    Load generated texts from JSON file and group by author.
    
    Args:
        input_file: Path to the generated_training_texts__baseline.json file
        
    Returns:
        Dictionary mapping author names to list of entries (preserving order)
    """
    logger.info(f"Loading generated texts from {input_file}")
    
    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    # Group by author while preserving order
    author_to_entries = defaultdict(list)
    for entry in data:
        author = entry["author"]
        author_to_entries[author].append(entry)
    
    # Log statistics
    logger.info(f"Loaded {len(data)} total entries")
    for author, entries in author_to_entries.items():
        logger.info(f"  {author}: {len(entries)} entries")
    
    return dict(author_to_entries)


class GeneratedTextDataset(Dataset):
    """
    Dataset for processing generated texts or original articles with prompts.
    
    The prompt is prepended to each text, but we track the prompt length
    so that only the text activations (excluding prompt) are saved.
    """
    
    def __init__(
        self, 
        entries: List[Dict], 
        model: HookedSAETransformer, 
        text_field: str,  # "generated_text" or "original_article"
        max_length: int = 500
    ):
        self.entries = entries
        self.tokenizer = model.tokenizer
        self.text_field = text_field
        self.max_length = max_length
        
    def __len__(self):
        return len(self.entries)
    
    def __getitem__(self, idx):
        entry = self.entries[idx]
        entry_id = entry.get("id", idx)
        prompt = entry["prompt"]
        text = entry[self.text_field]
        author = entry["author"]
        
        # Full text with prompt prepended
        full_text = prompt + "\n\n" + text
        
        # Tokenize prompt to get its length
        prompt_tokens = self.tokenizer(
            prompt + "\n\n",
            return_tensors="pt",
            add_special_tokens=True
        )
        prompt_length = prompt_tokens['input_ids'].shape[1]
        
        # Tokenize full text with padding
        encoded_inputs = self.tokenizer(
            full_text,
            return_tensors="pt",
            add_special_tokens=True,
            max_length=self.max_length + prompt_length,
            truncation=True,
            padding="max_length"
        )
        
        input_tokens = self.tokenizer.convert_ids_to_tokens(
            encoded_inputs['input_ids'].squeeze(0).tolist()
        )
        
        return {
            "entry_id": entry_id,
            "text": text,
            "prompt": prompt,
            "author": author,
            "encoded_inputs": encoded_inputs,
            "input_tokens": input_tokens,
            "prompt_length": prompt_length
        }


def custom_collate_fn(batch):
    """
    Custom collate function for GeneratedTextDataset.
    """
    entry_ids = [item["entry_id"] for item in batch]
    texts = [item["text"] for item in batch]
    prompts = [item["prompt"] for item in batch]
    authors = [item["author"] for item in batch]
    input_tokens_list = [item["input_tokens"] for item in batch]
    prompt_lengths = [item["prompt_length"] for item in batch]
    
    # Stack encoded inputs
    input_ids_batch = {
        'input_ids': torch.cat([item["encoded_inputs"]['input_ids'] for item in batch], dim=0),
        'attention_mask': torch.cat([item["encoded_inputs"]['attention_mask'] for item in batch], dim=0)
    }
    
    return {
        "entry_ids": entry_ids,
        "texts": texts,
        "prompts": prompts,
        "authors": authors,
        "input_ids_batch": input_ids_batch,
        "input_tokens_list": input_tokens_list,
        "prompt_lengths": prompt_lengths
    }


def process_author_texts(
    author: str,
    entries: List[Dict],
    text_field: str,  # "generated_text" or "original_article"
    model: HookedSAETransformer,
    saes: Dict,
    saes_ids: Dict,
    model_config: ModelConfig,
    save_dir: Path,
    max_seq_len: int = 500,
    batch_size: int = 2,
    storage_format: StorageFormat = StorageFormat.SPARSE,
    device: torch.device = None
):
    """
    Process all texts for a single author and text field.
    
    Args:
        author: Author name
        entries: List of entry dictionaries for this author
        text_field: Which field to process ("generated_text" or "original_article")
        model: The HookedSAETransformer model
        saes: Dictionary of loaded SAEs
        saes_ids: Dictionary of SAE IDs
        model_config: Model configuration
        save_dir: Directory to save activations
        max_seq_len: Maximum sequence length for activations
        batch_size: Batch size for processing
        storage_format: Storage format (dense or sparse)
        device: Torch device
    """
    logger.info(f"Processing {text_field} for author {author} ({len(entries)} entries)")
    
    # Create dataset and dataloader
    dataset = GeneratedTextDataset(
        entries=entries,
        model=model,
        text_field=text_field,
        max_length=max_seq_len
    )
    
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
        collate_fn=custom_collate_fn
    )
    
    n_docs = len(entries)
    
    # Initialize storage for each layer and SAE
    activations = {}
    for layer_type in saes_ids:
        activations[layer_type] = {}
        for sae_name, sae in zip(saes_ids[layer_type], saes[layer_type]):
            activations[layer_type][sae_name] = ActivationStorage(
                storage_format=storage_format,
                n_docs=n_docs,
                max_seq_len=max_seq_len,
                n_features=sae.cfg.d_sae
            )
    
    # Initialize storage for cross-entropy loss and entropy
    cross_entropy_loss_author = np.zeros((n_docs, max_seq_len), dtype=np.float32)
    entropy_author = np.zeros((n_docs, max_seq_len), dtype=np.float32)
    
    # Track document lengths
    doc_lengths = np.zeros(n_docs, dtype=np.int32)
    
    tokens_per_author = []
    full_texts_per_author = []
    
    with torch.no_grad():
        try:
            for batch_idx, batch in enumerate(tqdm(loader, desc=f"Processing {author}/{text_field}")):
                start_idx = batch_idx * batch_size
                end_idx = start_idx + len(batch["texts"])
                
                # Get prompt length (assume same for all in batch for simplicity)
                prompt_length = batch["prompt_lengths"][0]
                
                # Move tensors to device
                input_ids = batch["input_ids_batch"]['input_ids'].to(device)
                attention_mask = batch["input_ids_batch"]['attention_mask'].to(device)
                
                # Calculate actual text lengths (excluding padding and prompt)
                actual_lengths = []
                for i, pl in enumerate(batch["prompt_lengths"]):
                    total_len = attention_mask[i].sum().item()
                    text_len = min(total_len - pl, max_seq_len)  # Exclude prompt, cap at max_seq_len
                    actual_lengths.append(max(0, text_len))
                
                # Store tokens (text portion only)
                for i, input_tokens_doc in enumerate(batch["input_tokens_list"]):
                    pl = batch["prompt_lengths"][i]
                    tokens_per_author.append(input_tokens_doc[pl:])
                full_texts_per_author.extend(batch["texts"])
                
                # Forward pass
                full_list_saes = []
                for layer_type in model_config.layer_types:
                    full_list_saes.extend(saes[layer_type])
                
                logits, cache = model.run_with_cache_with_saes(input_ids, saes=full_list_saes)
                
                # Get cross-entropy loss
                ce_loss = lm_cross_entropy_loss(
                    logits, input_ids,
                    attention_mask=attention_mask,
                    per_token=True
                )
                mask_ce = attention_mask[:, 1:].bool()
                ce_loss = ce_loss.masked_fill(~mask_ce, 0.0)
                
                entropy = compute_entropy(logits)
                entropy = entropy.masked_fill(~attention_mask.bool(), 0.0)
                
                # Store cross-entropy and entropy (text portion only)
                for i in range(len(batch["texts"])):
                    pl = batch["prompt_lengths"][i]
                    actual_len = actual_lengths[i]
                    
                    # Extract text portion (skip prompt tokens)
                    ce_text = ce_loss[i, pl - 1:].cpu().numpy()[:max_seq_len]
                    entropy_text = entropy[i, pl:].cpu().numpy()[:max_seq_len]
                    
                    cross_entropy_loss_author[start_idx + i, :len(ce_text)] = ce_text
                    entropy_author[start_idx + i, :len(entropy_text)] = entropy_text
                    doc_lengths[start_idx + i] = actual_len
                
                # Store activations (text portion only)
                batch_doc_indices = list(range(start_idx, end_idx))
                
                for layer_type in model_config.layer_types:
                    for layeri, sae_id, sae in zip(model_config.layer_indices, saes_ids[layer_type], saes[layer_type]):
                        hook_name = f'{sae.cfg.metadata.hook_name}.hook_sae_acts_post'
                        
                        # Get activations for text portion only (skip prompt)
                        sae_acts_full = cache[hook_name].to(device)
                        
                        # Process each item in batch
                        for i in range(len(batch["texts"])):
                            pl = batch["prompt_lengths"][i]
                            doc_idx = batch_doc_indices[i]
                            actual_len = actual_lengths[i]
                            
                            # Extract text activations (skip prompt)
                            sae_acts_text = sae_acts_full[i, pl:][:max_seq_len]
                            
                            # Create single-item batch for add_batch
                            sae_acts_single = sae_acts_text.unsqueeze(0)
                            
                            activations[layer_type][sae_id].add_batch(
                                sae_acts_single,
                                [doc_idx],
                                [actual_len]
                            )
                
                # Clean up memory periodically
                if batch_idx % 5 == 0:
                    torch.cuda.empty_cache()
                    gc.collect()
                
                del logits, cache, ce_loss, entropy
                
        except Exception as e:
            logger.error(f"Error processing author {author}, field {text_field}: {e}")
            traceback.print_exc()
            torch.cuda.empty_cache()
            gc.collect()
            raise
    
    # Save results
    model_name_safe = model_config.model_name.replace('/', '_')
    field_suffix = "generated" if text_field == "generated_text" else "original"
    
    logger.info("Saving entropy and cross-entropy loss...")
    np.save(
        save_dir / f"sae_generated_texts__{model_name_safe}__entropy__{author}__{field_suffix}.npy",
        entropy_author
    )
    np.save(
        save_dir / f"sae_generated_texts__{model_name_safe}__cross_entropy_loss__{author}__{field_suffix}.npy",
        cross_entropy_loss_author
    )
    
    # Save tokens and full texts
    logger.info("Saving tokens and full texts...")
    with open(
        save_dir / f"sae_generated_texts__{model_name_safe}__tokens__{author}__{field_suffix}.json",
        "w", encoding="utf-8"
    ) as f:
        json.dump(tokens_per_author, f, ensure_ascii=False, indent=4)
    
    with open(
        save_dir / f"sae_generated_texts__{model_name_safe}__full_texts__{author}__{field_suffix}.json",
        "w", encoding="utf-8"
    ) as f:
        json.dump(full_texts_per_author, f, ensure_ascii=False, indent=4)
    
    # Save activations
    logger.info("Saving activations...")
    save_times = []
    file_sizes = []
    
    for layer_type in model_config.layer_types:
        for layeri, sae_id in zip(model_config.layer_indices, saes_ids[layer_type]):
            base_filename = (f"sae_generated_texts__{model_name_safe}__{layer_type}"
                           f"__activations__{author}__{field_suffix}__layer_{layeri}")
            filepath = save_dir / base_filename
            
            save_time, file_size = activations[layer_type][sae_id].save(
                filepath=Path(filepath),
                author_id=author,
                doc_lengths=doc_lengths,
                layer_type=layer_type,
                layer_index=layeri,
                sae_id=sae_id,
                model_name=model_name_safe
            )
            save_times.append(save_time)
            file_sizes.append(file_size)
    
    # Log summary
    if save_times:
        avg_save_time = np.mean(save_times)
        total_size = np.sum(file_sizes)
        
        def format_size(size_bytes):
            for unit in ['B', 'KB', 'MB', 'GB']:
                if size_bytes < 1024.0:
                    return f"{size_bytes:.2f} {unit}"
                size_bytes /= 1024.0
            return f"{size_bytes:.2f} TB"
        
        logger.info(f"=== Activation Files Summary for {author}/{field_suffix} ===")
        logger.info(f"Total files saved: {len(save_times)}")
        logger.info(f"Average time per file: {avg_save_time:.2f}s")
        logger.info(f"Total storage used: {format_size(total_size)}")
    
    # Cleanup
    del activations, cross_entropy_loss_author, entropy_author
    del doc_lengths, tokens_per_author, full_texts_per_author
    del dataset, loader
    torch.cuda.empty_cache()
    gc.collect()
    
    logger.info(f"Completed processing {text_field} for author: {author}")


def main(parsed_args):
    """Main function to extract SAE features from generated texts."""
    
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Create save directory
    save_dir = Path(parsed_args.output_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Data will be saved to: {save_dir}")
    
    # Load generated texts
    input_file = Path(parsed_args.input_file)
    author_to_entries = load_generated_texts(input_file)
    
    # Initialize model config
    model_config = ModelConfig(
        model_name=parsed_args.model,
        layer_indices=parsed_args.layers
    )
    
    # Load model
    model = load_model(model_config, device)
    
    # Load SAEs
    logger.info("Loading SAEs...")
    saes, saes_ids = load_saes(
        model_config, 
        parsed_args.layers, 
        parsed_args.sae_features_width, 
        device
    )
    
    # Determine storage format
    storage_format = StorageFormat(parsed_args.storage_format)
    
    # Process each author
    start_time = time.time()
    
    for author_idx, (author, entries) in enumerate(author_to_entries.items()):
        logger.info(f"\n=== Processing Author {author_idx + 1}/{len(author_to_entries)}: {author} ===")
        
        # First process generated_text
        logger.info(f"Processing generated_text for {author}...")
        process_author_texts(
            author=author,
            entries=entries,
            text_field="generated_text",
            model=model,
            saes=saes,
            saes_ids=saes_ids,
            model_config=model_config,
            save_dir=save_dir,
            max_seq_len=parsed_args.max_seq_len,
            batch_size=parsed_args.batch_size,
            storage_format=storage_format,
            device=device
        )
        
        # Then process original_article
        logger.info(f"Processing original_article for {author}...")
        process_author_texts(
            author=author,
            entries=entries,
            text_field="original_article",
            model=model,
            saes=saes,
            saes_ids=saes_ids,
            model_config=model_config,
            save_dir=save_dir,
            max_seq_len=parsed_args.max_seq_len,
            batch_size=parsed_args.batch_size,
            storage_format=storage_format,
            device=device
        )
        
        log_gpu_memory_usage()
    
    total_time = time.time() - start_time
    logger.info(f"\n=== Total runtime: {total_time:.2f}s ({total_time/60:.2f} minutes) ===")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract SAE features from generated texts and original articles."
    )
    
    # Model arguments
    parser.add_argument(
        "--model",
        type=str,
        default="google/gemma-2-9b-it",
        choices=["google/gemma-2-2b", "google/gemma-2-9b", "google/gemma-2-9b-it"],
        help="Model to use for feature extraction"
    )
    parser.add_argument(
        "--sae_features_width",
        type=str,
        default="16k",
        help="Number of SAE features (e.g., 16k, 32k, 65k, 131k)"
    )
    parser.add_argument(
        '--layers',
        nargs='+',
        type=int,
        default=[0, 5, 10, 15, 20, 25, 30, 35, 40, 41],
        help='Layer indices to process'
    )
    
    # Input/Output arguments
    parser.add_argument(
        "--input_file",
        type=str,
        default=str(DEFAULT_INPUT_FILE),
        help="Path to generated_training_texts__baseline.json"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory to save results"
    )
    
    # Processing arguments
    parser.add_argument(
        "--max_seq_len",
        type=int,
        default=500,
        help="Maximum sequence length for saved activations"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Batch size for processing"
    )
    parser.add_argument(
        "--storage_format",
        type=str,
        default="sparse",
        choices=["dense", "sparse"],
        help="Storage format: 'dense' for full arrays, 'sparse' for sparse matrices"
    )
    
    parsed_args = parser.parse_args()
    main(parsed_args)

