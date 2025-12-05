"""
Synthetic Data Generation and SAE Feature Extraction.

This module generates synthetic text data using language models and extracts SAE (Sparse Autoencoder) 
features from the generated texts. It supports multi-GPU processing for efficient large-scale analysis.

Key Features:
- Two-phase approach: text generation followed by feature extraction
- Support for multiple authors and topics
- Memory-efficient activation storage (dense and sparse formats)
- Cross-entropy loss and entropy computation
- Multi-GPU parallel processing
"""

import os
import argparse
import psutil
import gc
import json
import logging
import time
from typing import List
from pathlib import Path
from collections import defaultdict

from tqdm.auto import tqdm
import numpy as np
import scipy.sparse as sp
from dotenv import load_dotenv
import torch
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F
from huggingface_hub import login
from sae_lens import SAE, HookedSAETransformer
import transformer_lens.utils as utils
from transformers import AutoModelForCausalLM, AutoTokenizer

from backend.src.utils.run_configuration import ModelConfig, DatasetConfig, SAELayerConfig
from backend.src.utils.shared_utilities import ActivationMetadata, StorageFormat
from backend.src.get_features.activation_tracking import get_tracker

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ['TORCH_USE_CUDA_DSA'] = '1'

load_dotenv()
login(token=os.environ["HF_TOKEN"])



class ActivationStorage:
    """
    Manages activation storage in either dense or sparse format.
    Handles memory-efficient accumulation and saving.
    """
    
    def __init__(
        self, 
        storage_format: StorageFormat,
        n_docs: int,
        max_seq_len: int,
        n_features: int
    ):
        self.storage_format = storage_format
        self.n_docs = n_docs
        self.max_seq_len = max_seq_len
        self.n_features = n_features
        
        # Initialize storage based on format
        if storage_format == StorageFormat.DENSE:
            self.activations = np.zeros((n_docs, max_seq_len, n_features), dtype=np.float32)
            self.padding_mask = np.zeros((n_docs, max_seq_len), dtype=bool)
        else:
            # For sparse, accumulate as list of (doc_idx, tok_idx, feature_idx, value) tuples
            self.sparse_entries = []
            self.padding_mask = np.zeros((n_docs, max_seq_len), dtype=bool)
    
    def add_batch(
        self, 
        batch_activations: torch.Tensor,
        doc_indices: List[int],
        actual_lengths: List[int]
    ):
        """
        Add a batch of activations to storage.
        
        Args:
            batch_activations: Shape (batch_size, seq_len, n_features)
            doc_indices: Document indices for this batch
            actual_lengths: Actual token lengths (excluding padding)
        """
        batch_acts_np = batch_activations.cpu().numpy()
        
        for local_idx, (doc_idx, actual_len) in enumerate(zip(doc_indices, actual_lengths)):
            seq_len = min(batch_acts_np.shape[1], actual_len)
            
            # Update padding mask
            self.padding_mask[doc_idx, :seq_len] = True
            
            if self.storage_format == StorageFormat.DENSE:
                # Dense storage: directly assign
                self.activations[doc_idx, :seq_len, :] = batch_acts_np[local_idx, :seq_len, :]
            else:
                # Sparse storage: only store non-zero values
                acts = batch_acts_np[local_idx, :seq_len, :]
                non_zero_indices = np.nonzero(acts)
                
                for tok_offset, feat_idx in zip(non_zero_indices[0], non_zero_indices[1]):
                    value = acts[tok_offset, feat_idx]
                    self.sparse_entries.append((doc_idx, tok_offset, feat_idx, value))
    
    def save(
        self,
        filepath: Path,
        author_id: str,
        doc_lengths: np.ndarray,
        layer_type: str,
        layer_index: int,
        sae_id: str,
        model_name: str
    ):
        """
        Save activations and metadata to disk.
        
        Args:
            filepath: Base path for saving (without extension)
            author_id: Author identifier
            doc_lengths: Actual length of each document
            layer_type: Type of layer (res, mlp, att)
            layer_index: Layer number
            sae_id: SAE identifier,
            model_name: Model name
            
        Returns:
            tuple: (save_time, total_file_size) - Time taken in seconds and total size in bytes
        """
        start_time = time.time()
        
        # Create metadata
        n_docs = self.padding_mask.shape[0]
        max_seq = self.padding_mask.shape[1]
        doc_ids = np.repeat(np.arange(n_docs), max_seq)
        tok_ids = np.tile(np.arange(max_seq), n_docs)
        valid_mask = self.padding_mask.flatten()
        
        metadata = ActivationMetadata(
            doc_ids=doc_ids,
            tok_ids=tok_ids,
            author_id=author_id,
            doc_lengths=doc_lengths,
            valid_mask=valid_mask,
            original_shape=(n_docs, max_seq, self.n_features),
            n_features=self.n_features,
            storage_format=self.storage_format.value,
            layer_type=layer_type,
            layer_index=layer_index,
            sae_id=sae_id,
            model_name=model_name
        )
        
        # Save based on final format
        if self.storage_format == StorageFormat.DENSE:
            data_path, meta_path = self._save_dense(filepath, metadata)
        else:
            data_path, meta_path = self._save_sparse(filepath, metadata)
        
        save_time = time.time() - start_time
        
        # Calculate file sizes
        data_size = os.path.getsize(data_path)
        meta_size = os.path.getsize(meta_path)
        total_size = data_size + meta_size
        
        

        
        logger.info(f"Saved {self.storage_format.value} activations to {filepath} in {save_time:.2f}s")
        logger.info(f"  File sizes: data={format_size(data_size)}, metadata={format_size(meta_size)}, total={format_size(total_size)}")
        logger.info(f"  Valid tokens: {np.sum(valid_mask)}, Padding: {np.sum(~valid_mask)}")
        
        return save_time, total_size
    
    def _save_dense(self, filepath: Path, metadata: ActivationMetadata):
        """Save in dense format.
        
        Returns:
            tuple: (data_path, meta_path) - Paths to the saved files
        """
        data_path = Path(filepath).with_suffix('.npz')
        meta_path = Path(filepath).with_suffix('.meta.pkl')
        
        np.savez_compressed(
            data_path,
            activations=self.activations,
            padding_mask=self.padding_mask
        )
        
        metadata.save(meta_path)
        
        return data_path, meta_path
    
    def _save_sparse(self, filepath: Path, metadata: ActivationMetadata):
        """Save in sparse format.
        
        Returns:
            tuple: (data_path, meta_path) - Paths to the saved files
        """
        data_path = Path(filepath).with_suffix('.sparse.npz')
        meta_path = Path(filepath).with_suffix('.meta.pkl')
        
        if self.storage_format == StorageFormat.DENSE:
            # Convert dense to sparse
            flat_acts = self.activations.reshape(-1, self.n_features)
            sparse_matrix = sp.csr_matrix(flat_acts)
        else:
            # Build sparse matrix from entries
            n_positions = self.n_docs * self.max_seq_len
            
            if len(self.sparse_entries) == 0:
                # Handle empty case
                sparse_matrix = sp.csr_matrix((n_positions, self.n_features), dtype=np.float32)
            else:
                doc_indices, tok_indices, feat_indices, values = zip(*self.sparse_entries)
                row_indices = np.array(doc_indices) * self.max_seq_len + np.array(tok_indices)
                
                sparse_matrix = sp.csr_matrix(
                    (values, (row_indices, feat_indices)),
                    shape=(n_positions, self.n_features),
                    dtype=np.float32
                )
        
        sp.save_npz(data_path, sparse_matrix)
        metadata.save(meta_path)
        
        return data_path, meta_path

def format_size(size_bytes):
    """
    Convert bytes to human-readable format.
    
    Args:
        size_bytes (int): Size in bytes
        
    Returns:
        str: Human-readable size string (e.g., "1.23 MB", "4.56 GB")
    """
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.2f} TB"

def generate_text_with_temperature(
    model,
    prompt: str,
    max_new_tokens: int = 512,
    temperature: float = 1.2,
    top_p: float = 0.95,
    min_word_count: int = 100,
    max_attempts: int = 3
) -> str:
    """
    Generate text from a prompt with temperature sampling.
    
    Args:
        model: HookedSAETransformer model
        prompt: Input prompt string
        max_new_tokens: Maximum number of tokens to generate
        temperature: Sampling temperature for diversity (higher = more diverse)
        top_p: Nucleus sampling threshold
        min_word_count: Minimum word count for generated text
        max_attempts: Maximum number of generation attempts to meet min_word_count
        
    Returns:
        Generated text string
    """
    tokenizer = model.tokenizer
    
    for attempt in range(max_attempts):
        # Encode prompt
        prompt_tokens = tokenizer.encode(prompt, return_tensors="pt").to(model.cfg.device)
        
        # Generate tokens with temperature sampling
        generated_tokens = prompt_tokens.clone()
        
        with torch.no_grad():
            for _ in range(max_new_tokens):
                # Get logits for next token
                logits = model(generated_tokens)[:, -1, :]
                
                # Apply temperature
                logits = logits / temperature
                
                # Apply top-p (nucleus) sampling
                sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                
                # Remove tokens with cumulative probability above threshold
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                
                # Set logits to -inf for removed indices
                logits_processed = logits.clone()
                for batch_idx in range(logits.shape[0]):
                    indices_to_remove = sorted_indices[batch_idx][sorted_indices_to_remove[batch_idx]]
                    logits_processed[batch_idx, indices_to_remove] = float('-inf')
                
                # Sample from the filtered distribution
                probs = F.softmax(logits_processed, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                
                # Append to generated tokens
                generated_tokens = torch.cat([generated_tokens, next_token], dim=-1)
                
                # Check for end of sequence
                if next_token.item() == tokenizer.eos_token_id:
                    break
        
        # Decode generated text (excluding prompt)
        generated_text = tokenizer.decode(generated_tokens[0, prompt_tokens.shape[1]:], skip_special_tokens=True)
        
        # Check word count
        word_count = len(generated_text.split())
        if word_count >= min_word_count:
            return generated_text
        
        logger.debug(f"Attempt {attempt + 1}: Generated {word_count} words, need {min_word_count}. Retrying...")
    
    # Return the last attempt even if it doesn't meet minimum
    logger.warning(f"Could not generate text with {min_word_count} words after {max_attempts} attempts. Returning last attempt with {word_count} words.")
    return generated_text

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

def generate_text_with_hf_model(hf_model, hf_tokenizer, prompt_ids, max_new_tokens, temperature):
    """
    Generate text with the HuggingFace model.
    
    Args:
        hf_model: HuggingFace model
        hf_tokenizer: HuggingFace tokenizer
        prompt_ids: Input prompt tokens
        max_new_tokens: Maximum number of tokens to generate
        temperature: Sampling temperature
        
    Returns:
        Generated text string
        Word count
    """
    with torch.no_grad():
        generated_ids = hf_model.generate(
            prompt_ids,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=0.95,
            pad_token_id=hf_tokenizer.pad_token_id,
            eos_token_id=hf_tokenizer.eos_token_id,
        )
    generated_text = hf_tokenizer.decode(generated_ids[0, prompt_ids.shape[1]:], skip_special_tokens=True)
    del generated_ids
    word_count = len(generated_text.split())
    return generated_text, word_count

def generate_texts_only(
    model_name: str,
    topics: List[str],
    authors: List[str],
    author_full_names: dict,
    n_docs_per_author: int,
    min_length_doc: int,
    save_dir: Path,
    temperature: float = 1.2,
    max_new_tokens: int = 512,
    device: torch.device = None
):
    """
    Phase 1: Generate synthetic texts using HuggingFace model only.
    
    This function loads ONLY the HuggingFace model for generation, saves the texts,
    then deletes the model to free memory before the analysis phase.
    
    Args:
        model_name: Model name/path
        topics: List of topics to generate speeches about
        authors: List of author identifiers
        author_full_names: Dictionary mapping author IDs to full names
        n_docs_per_author: Number of documents to generate per author
        min_length_doc: Minimum length of documents in words
        save_dir: Directory to save generated texts
        temperature: Sampling temperature (higher = more diverse)
        max_new_tokens: Maximum tokens per generation
        device: Torch device
    
    Returns:
        Dictionary mapping (author, topic) to list of generated texts
    """
    
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load HuggingFace model for text generation
    logger.info(f"[PHASE 1: GENERATION] Loading HuggingFace model {model_name}...")
    
    hf_model, hf_tokenizer = load_text_generation_model(model_name, device)
    
    log_gpu_memory_usage()
    
    model_name_safe = model_name.replace('/', '_')
    generated_texts = defaultdict(dict)
    
    for author_idx, author in enumerate(authors):
        for topic in topics:
            logger.info(f"[GENERATION] Author {author_idx + 1}/{len(authors)}: {author} on topic: {topic}")
            
            author_full_name = author_full_names.get(author, author)
            texts_for_author_topic = []
            
            # Generate documents
            for doc_idx in tqdm(range(n_docs_per_author), desc=f"Generating for {author}"):
                # Create prompt
                prompt = f"Write a speech or a point of view on topic of {topic} the way and in style as president {author_full_name}."
                
                # Tokenize prompt
                prompt_ids = hf_tokenizer.encode(prompt, return_tensors="pt", add_special_tokens=True).to(device)

                retry_count = 0
                word_count = 0
                while word_count < min_length_doc and retry_count <= 3:
                    generated_text, word_count = generate_text_with_hf_model(hf_model, hf_tokenizer, prompt_ids, max_new_tokens, temperature)
                    retry_count += 1
                
                if word_count < min_length_doc:
                    logger.warning(f"Could not generate {min_length_doc} words after 3 attempts. Got {word_count} words.")
                
                texts_for_author_topic.append({
                    'prompt': prompt,
                    'text': generated_text,
                    'word_count': word_count
                })
            
            # Save generated texts for this author-topic combination
            # Generated texts go to: data/synthetic_data/<run_name>
            output_file = os.path.join(
                save_dir, 
                f"generated_texts__{model_name_safe}__{author}__topic_{topic}.json"
            )
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(texts_for_author_topic, f, ensure_ascii=False, indent=2)
            
            generated_texts[author][topic] = texts_for_author_topic
            logger.info(f"Saved {len(texts_for_author_topic)} texts to {output_file}")
    
    # Clean up HF model to free memory
    logger.info("[GENERATION] Cleaning up HuggingFace model to free memory...")
    #hf_model.to("cpu")
    del hf_model, hf_tokenizer
    # Force garbage collection multiple times
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    
    # Additional cleanup: reset peak memory stats
    #torch.cuda.reset_peak_memory_stats()
    
    log_gpu_memory_usage()
    
    return generated_texts

def load_generated_texts(save_dir: Path, model_name_safe: str, author: str, topic: str):
    """
    Load generated texts for a given author and topic.
    
    Args:
        save_dir: Directory containing generated texts
        model_name_safe: Safe model name
        author: Author identifier
        topic: Topic
        
    Returns:
        List of generated texts
    """
    input_file = os.path.join(
                save_dir,
                f"generated_texts__{model_name_safe}__{author}__topic_{topic}.json"
            )
            
    if not os.path.exists(input_file):
        logger.error(f"Generated texts file not found: {input_file}")
        logger.error("Please run generation phase first!")
        return None
    
    with open(input_file, "r", encoding="utf-8") as f:
        texts_data = json.load(f)

    logger.info(f"Loaded {len(texts_data)} generated texts from {input_file}")

    return texts_data

def get_activations_for_generated_texts(
    model,
    saes: dict,
    saes_ids: dict,
    topics: List[str],
    authors: List[str],
    n_docs_per_author: int,
    model_config: ModelConfig,
    dataset_config: DatasetConfig,
    generated_texts_dir: Path,
    sparse_features_dir: Path,
    dense_features_dir: Path,
    run_name: str,
    device: torch.device = None
):
    """
    Phase 2: Load generated texts and process them with HookedSAETransformer for activation caching.
    
    This function loads previously generated texts and runs them through the HookedSAETransformer
    to capture SAE activations, cross-entropy loss, and entropy.
    
    Args:
        model: HookedSAETransformer model (already loaded)
        saes: Dictionary of SAEs by layer type
        saes_ids: Dictionary of SAE IDs by layer type
        topics: List of topics
        authors: List of author identifiers
        n_docs_per_author: Number of documents per author
        model_config: Model configuration
        dataset_config: Dataset configuration
        generated_texts_dir: Directory containing generated texts (data/synthetic_data/<run_name>)
        sparse_features_dir: Base directory for sparse features (data/raw_features/synthetic)
        dense_features_dir: Base directory for dense features (data/raw_dense_features/synthetic)
        device: Torch device
    """
    
    if device is None:
        device = model.cfg.device
    
    logger.info("[PHASE 2: ANALYSIS] Processing generated texts with HookedSAETransformer...")
    log_gpu_memory_usage()
    
    model_name_safe = model_config.model_name.replace('/', '_')

    layer_type_hook = {
                        "res": "resid_post",
                        "mlp": "mlp_out",
                        "att": "attn_out"
                            }
    
    for author_idx, author in enumerate(authors):
        for topic in topics:
            logger.info(f"\n=== [ANALYSIS] Author {author_idx + 1}/{len(authors)}: {author} on topic: {topic} ===")
            
            # Load generated texts for this author-topic
            texts_data = load_generated_texts(generated_texts_dir, model_name_safe, author, topic)
            if texts_data is None:
                continue
            
            # Create topic-specific directories
            # Sparse: data/raw_features/synthetic/<topic_name>/<model_name>/<run_name>
            # Dense: data/raw_dense_features/synthetic/<topic_name>/<model_name>/<run_name>
            topic_sparse_dir = sparse_features_dir / topic / model_name_safe / run_name
            topic_dense_dir = dense_features_dir / topic / model_name_safe / run_name
            topic_sparse_dir.mkdir(parents=True, exist_ok=True)
            topic_dense_dir.mkdir(parents=True, exist_ok=True)
            
            # Initialize storage for each layer type and SAE
            activations_sparse = {}
            activations_dense = {}
            for layer_type in saes_ids:
                activations_sparse[layer_type] = {}
                activations_dense[layer_type] = {}
                for sae_name, sae in zip(saes_ids[layer_type], saes[layer_type]):
                    layer_index = int(sae_name.split("/")[0].replace("layer_", ""))
                    activations_sparse[layer_type][sae_name] = ActivationStorage(
                        storage_format=StorageFormat.SPARSE,
                        n_docs=n_docs_per_author,
                        max_seq_len=dataset_config.max_sequence_length - 1,
                        n_features=sae.cfg.d_sae
                    )
                    activations_dense[layer_type][layer_index] = ActivationStorage(
                        storage_format=StorageFormat.DENSE,
                        n_docs=n_docs_per_author,
                        max_seq_len=dataset_config.max_sequence_length - 1,
                        n_features=model.cfg.d_model
                    )
            
            # Initialize storage for cross-entropy loss and entropy
            cross_entropy_loss_author = np.zeros((n_docs_per_author, dataset_config.max_sequence_length - 1), dtype=np.float32)
            entropy_author = np.zeros((n_docs_per_author, dataset_config.max_sequence_length - 1), dtype=np.float32)
            doc_lengths = np.zeros(n_docs_per_author, dtype=np.int32)
            
            tokens_per_author = []
            full_texts_per_author = []
            
            # Process each generated document
            for doc_idx, text_data in enumerate(tqdm(texts_data, desc=f"Analyzing {author}")):
                prompt = text_data['prompt']
                generated_text = text_data['text']
                full_texts_per_author.append(generated_text)
                
                # Combine prompt and generated text for processing (use the generated_ids which include prompt)
                full_text = prompt + " " + generated_text
                
                # Tokenize
                encoded_inputs = model.tokenizer(
                    full_text,
                    return_tensors="pt",
                    add_special_tokens=True,
                    max_length=dataset_config.max_sequence_length,
                    truncation=True,
                    padding="max_length"
                )
                
                input_ids = encoded_inputs['input_ids'].to(device)
                attention_mask = encoded_inputs['attention_mask'].to(device)
                
                # Calculate prompt length
                prompt_tokens = model.tokenizer.encode(prompt, add_special_tokens=True)
                prompt_length = len(prompt_tokens)
                
                # Get tokens
                input_tokens = model.tokenizer.convert_ids_to_tokens(input_ids.squeeze(0).tolist())
                tokens_per_author.append(input_tokens[prompt_length:])
                
                # Calculate actual length (excluding padding and prompt)
                actual_length = (attention_mask.sum().item() - prompt_length)
                doc_lengths[doc_idx] = actual_length
                
                # Forward pass with SAE caching
                with torch.no_grad():
                    full_list_saes = []
                    for layer_type in model_config.layer_types:
                        full_list_saes.extend(saes[layer_type])
                    
                    logits, cache = model.run_with_cache_with_saes(input_ids, saes=full_list_saes)
                    print(f"Cache hooks: {cache.keys()}")
                    
                    # Compute cross-entropy loss
                    ce_loss = lm_cross_entropy_loss(
                        logits, input_ids,
                        attention_mask=attention_mask,
                        per_token=True
                    )
                    mask_ce = attention_mask[:, 1:].bool()
                    ce_loss = ce_loss.masked_fill(~mask_ce, 0.0)
                    
                    # Compute entropy
                    entropy = compute_entropy(logits)
                    entropy = entropy.masked_fill(~attention_mask.bool(), 0.0)
                    
                    # Store cross-entropy and entropy (excluding prompt)
                    ce_slice_len = min(ce_loss.shape[1] - (prompt_length - 1), dataset_config.max_sequence_length - 1)
                    entropy_slice_len = min(entropy.shape[1] - prompt_length, dataset_config.max_sequence_length - 1)
                    
                    cross_entropy_loss_author[doc_idx, :ce_slice_len] = \
                        ce_loss[:, prompt_length - 1:prompt_length - 1 + ce_slice_len].cpu().numpy()
                    entropy_author[doc_idx, :entropy_slice_len] = \
                        entropy[:, prompt_length:prompt_length + entropy_slice_len].cpu().numpy()
                    
                    # Store SAE activations and dense activations
                    for layer_type in model_config.layer_types:
                        for layeri, sae_id, sae in zip(model_config.layer_indices, saes_ids[layer_type], saes[layer_type]):
                            # Get SAE activations
                            hook_name = f'{sae.cfg.metadata.hook_name}.hook_sae_acts_post'
                            sae_acts = cache[hook_name][:, prompt_length:].to(device)
                            
                            # Get dense activations (pre-SAE)
                            
                            dense_hook_name = utils.get_act_name(layer_type_hook[layer_type], layeri)
                            dense_acts = cache[f"{dense_hook_name}.hook_sae_input"][:, prompt_length:].to(device)
                            
                            # Ensure proper dimensions
                            seq_len = min(sae_acts.shape[1], dataset_config.max_sequence_length - 1)
                            
                            activations_sparse[layer_type][sae_id].add_batch(
                                sae_acts[:, :seq_len],
                                [doc_idx],
                                [actual_length]
                            )
                            
                            activations_dense[layer_type][layeri].add_batch(
                                dense_acts[:, :seq_len],
                                [doc_idx],
                                [actual_length]
                            )
                    
                    # Clean up
                    del logits, cache, ce_loss, entropy
                    if doc_idx % 10 == 0:
                        torch.cuda.empty_cache()
                        gc.collect()
            
            # Save all outputs
            logger.info(f"Saving outputs for author {author}...")
            
            # Save entropy and cross-entropy loss in both sparse and dense directories
            # Filenames: synthetic___{model_name_safe}__entropy__{author}.npy
            entropy_filename = f"synthetic___{model_name_safe}__entropy__{author}.npy"
            cross_entropy_filename = f"synthetic___{model_name_safe}__cross_entropy_loss__{author}.npy"
            
            np.save(topic_sparse_dir / entropy_filename, entropy_author)
            np.save(topic_sparse_dir / cross_entropy_filename, cross_entropy_loss_author)
            np.save(topic_dense_dir / entropy_filename, entropy_author)
            np.save(topic_dense_dir / cross_entropy_filename, cross_entropy_loss_author)
            
            # Save tokens and full texts in both sparse and dense directories
            # Filenames: synthetic__{model_name_safe}__tokens__{author}.json
            tokens_filename = f"synthetic__{model_name_safe}__tokens__{author}.json"
            full_texts_filename = f"synthetic__{model_name_safe}__full_texts__{author}.json"
            
            with open(topic_sparse_dir / tokens_filename, "w", encoding="utf-8") as f:
                json.dump(tokens_per_author, f, ensure_ascii=False, indent=4)
            
            with open(topic_sparse_dir / full_texts_filename, "w", encoding="utf-8") as f:
                json.dump(full_texts_per_author, f, ensure_ascii=False, indent=4)
            
            with open(topic_dense_dir / tokens_filename, "w", encoding="utf-8") as f:
                json.dump(tokens_per_author, f, ensure_ascii=False, indent=4)
            
            with open(topic_dense_dir / full_texts_filename, "w", encoding="utf-8") as f:
                json.dump(full_texts_per_author, f, ensure_ascii=False, indent=4)
            
            # Save SAE activations and dense activations
            logger.info("Saving activations...")
            save_times = []
            file_sizes = []
            
            for layer_type in model_config.layer_types:
                for layeri, sae_id in zip(model_config.layer_indices, saes_ids[layer_type]):
                    # Save SAE activations (sparse)
                    # Filename: sae__{model_name_safe}__{layer_type}__activations__{author}__layer_{layeri}
                    base_filename_sae = f"sae__{model_name_safe}__{layer_type}__activations__{author}__layer_{layeri}"
                    filepath_sae = topic_sparse_dir / base_filename_sae
                    
                    save_time_sae, file_size_sae = activations_sparse[layer_type][sae_id].save(
                        filepath=filepath_sae,
                        author_id=author,
                        doc_lengths=doc_lengths,
                        layer_type=layer_type,
                        layer_index=layeri,
                        sae_id=sae_id,
                        model_name=model_name_safe
                    )
                    save_times.append(save_time_sae)
                    file_sizes.append(file_size_sae)
                    
                    # Save dense activations
                    # Filename: dense__{model_name_safe}__{layer_type}__activations__{author}__layer_{layeri}
                    base_filename_dense = f"dense__{model_name_safe}__{layer_type}__activations__{author}__layer_{layeri}"
                    filepath_dense = topic_dense_dir / base_filename_dense
                    
                    save_time_dense, file_size_dense = activations_dense[layer_type][layeri].save(
                        filepath=filepath_dense,
                        author_id=author,
                        doc_lengths=doc_lengths,
                        layer_type=layer_type,
                        layer_index=layeri,
                        sae_id="dense",  # Identifier for dense activations
                        model_name=model_name_safe
                    )
                    save_times.append(save_time_dense)
                    file_sizes.append(file_size_dense)
            
            # Log summary
            if save_times:
                
                logger.info(f"=== Activation Files Summary for {author} ===")
                logger.info(f"Total files saved: {len(save_times)}")
                logger.info(f"Average time to save one file: {np.mean(save_times):.2f}s")
                logger.info(f"Total save time: {np.sum(save_times):.2f}s")
                logger.info(f"Average file size: {format_size(np.mean(file_sizes))}")
                logger.info(f"Total storage used: {format_size(np.sum(file_sizes))}")
                logger.info(f"==========================================")
            
            # Cleanup
            del activations_sparse, activations_dense
            del cross_entropy_loss_author, entropy_author
            del doc_lengths, tokens_per_author, full_texts_per_author
            torch.cuda.empty_cache()
            gc.collect()
            
            logger.info(f"Completed analyzing texts for author: {author}")


def load_canonical_sae(sae_layer_config: SAELayerConfig, device: torch.device):
    """
    Load a canonical SAE for the specified layer configuration.
    
    Args:
        sae_layer_config (SAELayerConfig): Configuration specifying layer type, 
            layer index, width, and model name
        device (torch.device): Device to load the SAE onto
        
    Returns:
        tuple: (sae, sae_id) if successful, None if loading fails
        
    Raises:
        Exception: Logs error and returns None if SAE cannot be loaded
    """ 
    try:
        logger.info(f"Loading canonical SAE for layer_type={sae_layer_config.layer_type}, layer={sae_layer_config.layer_index}, width={sae_layer_config.width} from {sae_layer_config.release_name} with sae_id={sae_layer_config.sae_id}")
        sae, cfg, sparsity = SAE.from_pretrained(
            release=sae_layer_config.release_name,
            sae_id=sae_layer_config.sae_id,
        )
        sae = sae.to(device)
        sae.eval()
        return sae, sae_layer_config.sae_id
    except Exception as e:
        logger.error(f"Could not load canonical SAE for layer_type={sae_layer_config.layer_type}, layer={sae_layer_config.layer_index}, width={sae_layer_config.width}: {e}")
        return None

def load_saes(model_config: ModelConfig, width, device):
    """
    Load multiple SAEs for specified layers and layer types.
    
    Args:
        model_config (ModelConfig): Model configuration containing layer types and indices
        width (str): SAE width specification (e.g., "16k", "32k")
        device (torch.device): Device to load SAEs onto
        
    Returns:
        tuple: (saes, saes_ids) dictionaries mapping layer_type to lists of SAEs and IDs
        
    Note:
        layer_type can be 'res', 'mlp', or 'att'
        width examples: "16k" for 16384 features
    """
    saes = {}
    saes_ids = {}
    for layer_type in model_config.layer_types:
        saes[layer_type] = []
        saes_ids[layer_type] = []
        for layer in model_config.layer_indices:
            sae_layer_config = SAELayerConfig(layer_type=layer_type, layer_index=layer, width=width, model_name=model_config.model_name, canonical=True)
            sae, sae_id = load_canonical_sae(sae_layer_config, device)
            saes[layer_type].append(sae)
            saes_ids[layer_type].append(sae_id)
    
        logger.info(f"Loaded {len(saes[layer_type])} SAEs for layer_type {layer_type}")
    return saes, saes_ids


def lm_cross_entropy_loss(
    logits, #: Float[torch.Tensor, "batch pos d_vocab"],
    tokens, #: int[torch.Tensor, "batch pos"],
    attention_mask = None, #: Optional[int[torch.Tensor, "batch pos"]] = None,
    per_token = False #bool = False,
): # -> Union[float[torch.Tensor, ""], float[torch.Tensor, "batch pos"]]:
    """Cross entropy loss for the language model, gives the loss for predicting the NEXT token.

    Args:
        logits (torch.Tensor): Logits. Shape [batch, pos, d_vocab]
        tokens (torch.Tensor[int64]): Input tokens. Shape [batch, pos]
        attention_mask (torch.Tensor[int64], optional): Attention mask. Shape [batch, pos]. Used to
            mask out padding tokens. Defaults to None.
        per_token (bool, optional): Whether to return the log probs predicted for the correct token, or the loss (ie mean of the predicted log probs). Note that the returned array has shape [batch, seq-1] as we cannot predict the first token (alternately, we ignore the final logit). Defaults to False.
    """
    log_probs = F.log_softmax(logits, dim=-1)
    # Use torch.gather to find the log probs of the correct tokens
    # Offsets needed because we're predicting the NEXT token (this means the final logit is meaningless)
    # None and [..., 0] needed because the tensor used in gather must have the same rank.
    predicted_log_probs = log_probs[..., :-1, :].gather(dim=-1, index=tokens[..., 1:, None])[..., 0]

    if attention_mask is not None:
        # Ignore token positions which are masked out or where the next token is masked out
        # (generally padding tokens)
        next_token_mask = torch.logical_and(attention_mask[:, :-1], attention_mask[:, 1:])
        predicted_log_probs *= next_token_mask
        n_tokens = next_token_mask.sum().item()
    else:
        n_tokens = predicted_log_probs.numel()

    if per_token:
        return -predicted_log_probs
    else:
        return -predicted_log_probs.sum() / n_tokens

def compute_entropy(logits, dim=-1, eps=1e-12):
    """
    Compute the entropy of the probability distribution derived from logits.

    Args:
        logits (torch.Tensor): Logits tensor of shape [batch_size, seq_len, d_vocab].
        dim (int): The dimension along which softmax will be applied (usually -1 for vocab).
        eps (float): Small value to prevent log(0).

    Returns:
        torch.Tensor: Entropy values for each token in the sequence.
    """

    # Apply softmax to get probabilities
    probs = nn.functional.softmax(logits, dim=dim)
    # Compute entropy: H(p) = -sum(p * log(p))
    entropy = -(probs * torch.log(probs + eps)).sum(dim=dim)
    return entropy


def load_hooked_model(model_config: ModelConfig, device):
    """
    Load a HookedSAETransformer model for the specified configuration.
    
    Args:
        model_config (ModelConfig): Model configuration containing model name
        device (torch.device): Device to load the model onto
        
    Returns:
        HookedSAETransformer: Loaded and configured model in eval mode
        
    Note:
        The model is loaded with specific configurations:
        - fold_ln=True: Layer normalization folding
        - center_writing_weights=False: No weight centering
        - center_unembed=False: No unembedding centering
    """
    logger.info(f"Loading model {model_config.model_name} on {device}")
    model = HookedSAETransformer.from_pretrained(
        model_config.model_name,
        fold_ln=True,
        center_writing_weights=False,
        center_unembed=False,
        device = device)
    model.eval()

    # Log initial memory usage
    log_gpu_memory_usage()

    return model


def check_memory():
    """
    Check and log current memory usage for both process and system.
    
    This function forces garbage collection, then logs:
    - Current process memory usage
    - System available memory
    - Overall system memory usage percentage
    """
    # Force garbage collection
    gc.collect()

    # Check memory usage
    process = psutil.Process()
    mem_info = process.memory_info()
    mem_gb = mem_info.rss / 1024 ** 3

    # Check system memory
    sys_mem = psutil.virtual_memory()
    available_gb = sys_mem.available / 1024 ** 3

    logger.info(f"Current process memory: {mem_gb:.2f} GB")
    logger.info(f"System available memory: {available_gb:.2f} GB")
    logger.info(f"Memory usage: {sys_mem.percent:.1f}%")


def log_gpu_memory_usage():
    """
    Log GPU memory usage for all available GPUs.
    
    This function checks all available CUDA devices and logs:
    - Currently allocated memory
    - Reserved memory
    - Maximum allocated memory (peak usage)
    - Device names
    """
    if torch.cuda.is_available():
        logger.info("=== GPU Memory Status ===")
        for i in range(torch.cuda.device_count()):
            allocated = torch.cuda.memory_allocated(i) / 1024 ** 3
            reserved = torch.cuda.memory_reserved(i) / 1024 ** 3
            max_allocated = torch.cuda.max_memory_allocated(i) / 1024 ** 3
            logger.info(f"GPU {i} ({torch.cuda.get_device_name(i)}):")
            logger.info(f"  Allocated: {allocated:.2f} GB")
            logger.info(f"  Reserved: {reserved:.2f} GB") 
            logger.info(f"  Max Allocated: {max_allocated:.2f} GB")
        logger.info("========================")




def generate_and_save_activations(rank, author_subsets, parsed_args, dataset_config, save_dir):
    """
    Run the inference for the given rank and save activations and cross-entropy loss and entropy.
    
    This function now follows a two-phase approach:
    1. PHASE 1: Generate texts using HuggingFace model only (then delete it to free memory)
    2. PHASE 2: Load HookedSAETransformer and SAEs, then analyze the generated texts
    """
    try:
        # Define topics
        topics = parsed_args.topics if hasattr(parsed_args, 'topics') and parsed_args.topics else [
            "healthcare reform",
            "climate change",
            "immigration policy"
        ]
        #    "economic growth",
        #    "national security",
        #    "education policy",
        #    "tax reform",
        #    "foreign relations"
        #]
        author_full_names = parsed_args.author_full_names if hasattr(parsed_args, 'author_full_names') and parsed_args.author_full_names else {
            "obama": "Barack Obama",
            "trump": "Donald Trump",
            "bush": "George W. Bush"
        }

        # Setup multiprocessing environment
        torch.cuda.set_device(rank)
        device = torch.device(f"cuda:{rank}")
        
        model_config = ModelConfig(model_name=parsed_args.model, layer_indices=parsed_args.layers)

        # Assign authors to this GPU 
        authors_for_this_gpu = author_subsets[rank]
        author_full_names_for_this_gpu = {author: author_full_names[author] for author in authors_for_this_gpu}
        logger.info(f"GPU {rank} will process {len(authors_for_this_gpu)} authors: {authors_for_this_gpu}")

        logger.info(f"GPU {rank} generating speeches for authors: {authors_for_this_gpu}")
        logger.info(f"Topics: {topics}")
        logger.info(f"Documents per author: {parsed_args.n_docs_per_author}")
        logger.info(f"Minimum document length: {parsed_args.min_length_doc} words")
        logger.info(f"Temperature: {parsed_args.temperature}")
        
        # ========================================
        # PHASE 1: TEXT GENERATION (HuggingFace model only)
        # ========================================
        logger.info("=" * 80)
        logger.info(f"GPU {rank}: PHASE 1 - TEXT GENERATION")
        logger.info("=" * 80)
        
        # Generated texts go to: data/synthetic_data/<run_name>
        # Get project root (go up from backend/src/get_features to project root)
        project_root = Path(__file__).parent.parent.parent.parent
        generated_texts_dir = project_root / "data" / "synthetic_data" / parsed_args.run_name
        generated_texts_dir.mkdir(parents=True, exist_ok=True)
        
        generate_texts_only(
            model_name=model_config.model_name,
            topics=topics,
            authors=authors_for_this_gpu,
            author_full_names=author_full_names_for_this_gpu,
            n_docs_per_author=parsed_args.n_docs_per_author,
            min_length_doc=parsed_args.min_length_doc,
            save_dir=generated_texts_dir,
            temperature=parsed_args.temperature,
            max_new_tokens=parsed_args.max_new_tokens,
            device=device
        )
        
        logger.info(f"GPU {rank}: ✓ Phase 1 complete. HuggingFace model deleted, memory freed.")
        logger.info(f"GPU {rank}: Generated texts saved to disk.")
        
        # ========================================
        # PHASE 2: ACTIVATION ANALYSIS (HookedSAETransformer)
        # ========================================
        logger.info("=" * 80)
        logger.info(f"GPU {rank}: PHASE 2 - ACTIVATION ANALYSIS")
        logger.info("=" * 80)
        
        # Now load HookedSAETransformer (after HF model is deleted)
        logger.info(f"GPU {rank}: Loading HookedSAETransformer model...")
        model = load_hooked_model(model_config, device)
        
        # Load SAEs
        logger.info(f"GPU {rank}: Loading SAEs...")
        saes, saes_ids = load_saes(model_config, parsed_args.sae_features_width, device)
        
        # Get activations for the generated texts
        # Setup directory paths (relative to project root)
        model_name_safe = model_config.model_name.replace('/', '_')
        # Get project root (go up from backend/src/get_features to project root)
        project_root = Path(__file__).parent.parent.parent.parent
        generated_texts_dir = project_root / "data" / "synthetic_data" / parsed_args.run_name
        sparse_features_dir = project_root / "data" / "raw_features" / "synthetic"
        dense_features_dir = project_root / "data" / "raw_dense_features" / "synthetic"
        
        get_activations_for_generated_texts(
            model=model,
            saes=saes,
            saes_ids=saes_ids,
            topics=topics,
            authors=authors_for_this_gpu,
            n_docs_per_author=parsed_args.n_docs_per_author,
            model_config=model_config,
            dataset_config=dataset_config,
            generated_texts_dir=generated_texts_dir,
            sparse_features_dir=sparse_features_dir,
            dense_features_dir=dense_features_dir,
            run_name=parsed_args.run_name,
            device=device
        )
        
        logger.info(f"GPU {rank}: ✓ Phase 2 complete. All activations and analyses saved.")
        logger.info("=" * 80)
        logger.info(f"GPU {rank}: TWO-PHASE GENERATION COMPLETE!")
        logger.info("=" * 80)
    
    except Exception as e:
        logger.error(f"GPU {rank}: FATAL ERROR in generate_and_save_activations")
        logger.error(f"GPU {rank}: Error type: {type(e).__name__}")
        logger.error(f"GPU {rank}: Error message: {str(e)}")
        import traceback
        logger.error(f"GPU {rank}: Full traceback:\n{traceback.format_exc()}")
        raise  # Re-raise to let multiprocessing handle it

    
    
    
    
def main(parsed_args):

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    model_name_safe = parsed_args.model.replace('/', '_')
    dataset_dir = Path(parsed_args.dataset_name, parsed_args.category_name) if parsed_args.category_name is not None else Path(parsed_args.dataset_name)
    save_dir = Path(parsed_args.output_dir, dataset_dir, model_name_safe, parsed_args.run_name)
    os.makedirs(save_dir, exist_ok=True)

    dataset_config = DatasetConfig(
        dataset_name="synthetic",
        min_length_doc=parsed_args.min_length_doc,
        max_n_docs_per_author=parsed_args.n_docs_per_author
    )
    
    world_size = torch.cuda.device_count()
    
    # Define authors and their full names
    authors = parsed_args.authors if hasattr(parsed_args, 'authors') and parsed_args.authors else ["obama", "trump", "bush"]
    author_subsets = [authors[i::world_size] for i in range(world_size)]

    # Check if debug mode (single GPU, no multiprocessing)
    debug_mode = getattr(parsed_args, 'debug_single_gpu', False)
    
    if debug_mode:
        logger.info("DEBUG MODE: Running on single GPU without multiprocessing...")
        start_time = time.time()
        # Run directly on GPU 0 with all authors
        generate_and_save_activations(0, [authors], parsed_args, dataset_config, save_dir)
    else:
        # Run multi-GPU inference 
        logger.info("Starting multi-GPU inference...")
        start_time = time.time()
        
        mp.spawn(
        generate_and_save_activations,
        args=(author_subsets, parsed_args, dataset_config, save_dir),  
        nprocs=world_size,
        join=True
    )
    
    total_time = time.time() - start_time
    logger.info(f"=== Total runtime: {total_time:.2f}s ({total_time/60:.2f} minutes) ===")
    
    # Register runs in the tracking system
    # Create separate entries for sparse and dense features, with topic as category
    try:
        tracker = get_tracker()
        authors = parsed_args.authors if hasattr(parsed_args, 'authors') and parsed_args.authors else ["obama", "trump", "bush"]
        
        # Define topics (same as in generate_and_save_activations)
        topics = parsed_args.topics if hasattr(parsed_args, 'topics') and parsed_args.topics else [
            "healthcare reform",
            "climate change",
            "immigration policy"
        ]
        
        model_name_safe = parsed_args.model.replace('/', '_')
        
        # Get project root for consistent path handling
        project_root = Path(__file__).parent.parent.parent.parent
        
        # Register sparse features for each topic
        for topic in topics:
            sparse_path = project_root / "data" / "raw_features" / "synthetic" / topic / model_name_safe / parsed_args.run_name
            run_id_sparse = tracker.register_run(
                model=parsed_args.model,
                dataset="synthetic",
                run_name=parsed_args.run_name,
                layers=parsed_args.layers,
                activation_path=str(sparse_path),
                category=topic,  # Use topic as category
                authors=authors,
                storage_format="sparse",
                n_docs_per_author=parsed_args.n_docs_per_author,
                min_length_doc=parsed_args.min_length_doc,
                setting=None
            )
            logger.info(f"✓ Sparse activation run registered for topic '{topic}' with ID: {run_id_sparse}")
        
        # Register dense features for each topic
        for topic in topics:
            dense_path = project_root / "data" / "raw_dense_features" / "synthetic" / topic / model_name_safe / parsed_args.run_name
            run_id_dense = tracker.register_run(
                model=parsed_args.model,
                dataset="synthetic",
                run_name=parsed_args.run_name,
                layers=parsed_args.layers,
                activation_path=str(dense_path),
                category=topic,  # Use topic as category
                authors=authors,
                storage_format="dense",
                n_docs_per_author=parsed_args.n_docs_per_author,
                min_length_doc=parsed_args.min_length_doc,
                setting=None
            )
            logger.info(f"✓ Dense activation run registered for topic '{topic}' with ID: {run_id_dense}")
    except Exception as e:
        logger.warning(f"Failed to register activation run in tracking system: {e}")

            
if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Model arguments
    parser.add_argument(
        "--model",
        type=str,
        default="google/gemma-2-9b-it",
        choices=["google/gemma-2-2b", "google/gemma-2-9b", "google/gemma-2-9b-it"],
        help="Model to use for feature extraction"
    )
    
    parser.add_argument(
        '--layers',
        nargs='+',
        type=int,
        required=True,
        help='Layer indices to process'
    )
    
    parser.add_argument(
        "--sae_features_width",
        type=str,
        default="16k",
        help="Width of SAE features (e.g., 16k, 32k, 65k, 131k)"
    )
    
    # Dataset arguments
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="synthetic",
        choices=["AuthorMix", "news", "synthetic"],
        help="Dataset to use"
    )
    parser.add_argument(
        "--category_name",
        type=str,
        default=None,
        help="Category for news dataset"
    )
    parser.add_argument(
        "--n_docs",
        type=int,
        default=None,
        help="Total number of documents to analyze"
    )
    parser.add_argument(
        "--n_docs_per_author",
        type=int,
        default=200,
        help="Documents per author"
    )
    parser.add_argument(
        "--min_length_doc",
        type=int,
        default=35,
        help="Minimum document length (in words)"
    )
    
    # Speech generation arguments
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.2,
        help="Sampling temperature for text generation (higher = more diverse)"
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=512,
        help="Maximum number of tokens to generate per document"
    )
    parser.add_argument(
        "--authors",
        nargs='+',
        type=str,
        default=None,
        help="List of author identifiers for speech generation"
    )
    parser.add_argument(
        "--author_full_names",
        type=json.loads,
        default=None,
        help='JSON dict mapping author IDs to full names, e.g. \'{"obama": "Barack Obama"}\''
    )
    parser.add_argument(
        "--topics",
        nargs='+',
        type=str,
        default=None,
        help="List of topics for speech generation"
    )

    
    parser.add_argument(
        "--debug_single_gpu",
        action="store_true",
        help="Debug mode: run on single GPU without multiprocessing (helps see error messages)"
    )
    
    # Storage arguments
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/raw_features",
        help="Directory to save results"
    )
    parser.add_argument(
        "--run_name",
        type=str,
        default="synthetic_200_politics",
        help="Name of the run to create a folder in outputs"
    )

    parsed_args = parser.parse_args()
    
    main(parsed_args)
