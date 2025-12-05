""" 
Counts the number of times each feature fires across some corpus.
"""
import os
import argparse
import psutil
import gc
import traceback
import json
import logging
import time
from typing import List
from pathlib import Path

from tqdm.auto import tqdm
import numpy as np

from dotenv import load_dotenv
import torch
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from huggingface_hub import login
from transformer_lens import HookedTransformer
import transformer_lens.utils as utils

from backend.src.utils.load_dataset import load_news_data, load_AuthorMix_data
from backend.src.utils.run_configuration import ModelConfig, DatasetConfig
from backend.src.utils.shared_utilities import ActivationMetadata
from backend.src.get_features.activation_tracking import get_tracker



# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

load_dotenv()
login(token=os.environ["HF_TOKEN"])



class ActivationStorage:
    """
    Manages activation storage in either dense or sparse format.
    Handles memory-efficient accumulation and saving.
    """
    
    def __init__(
        self, 
        model: HookedTransformer,
        n_docs: int,
        max_seq_len: int,
    ):
        self.model = model
        self.n_docs = n_docs
        self.max_seq_len = max_seq_len
        self.n_features = model.cfg.d_model

        # Initialize storage
        self.activations = np.zeros((n_docs, max_seq_len, self.n_features), dtype=np.float32)
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
            
            
            # Dense storage: directly assign
            self.activations[doc_idx, :seq_len, :] = batch_acts_np[local_idx, :seq_len, :]
            
    
    def save(
        self,
        filepath: Path,
        author_id: str,
        doc_lengths: np.ndarray,
        layer_type: str,
        layer_index: int,
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
            storage_format="dense",
            sae_id=None,
            original_shape=(n_docs, max_seq, self.n_features),
            n_features=self.n_features,
            layer_type=layer_type,
            layer_index=layer_index,
            model_name=model_name
        )
        
        # Save 
        data_path, meta_path = self._save_dense(filepath, metadata)
        
        save_time = time.time() - start_time
        
        # Calculate file sizes
        data_size = os.path.getsize(data_path)
        meta_size = os.path.getsize(meta_path)
        total_size = data_size + meta_size
        
        # Convert to human-readable format
        def format_size(size_bytes):
            for unit in ['B', 'KB', 'MB', 'GB']:
                if size_bytes < 1024.0:
                    return f"{size_bytes:.2f} {unit}"
                size_bytes /= 1024.0
            return f"{size_bytes:.2f} TB"
        
        logger.info(f"Saved dense activations to {filepath} in {save_time:.2f}s")
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
        #np.savez(data_path, self.activations)
        metadata.save(meta_path)
        
        return data_path, meta_path
    


class MyDataset(Dataset):
    def __init__(self, documents, model, max_length=360, setting="baseline"):
        self.docs = documents
        self.tokenizer = model.tokenizer
        self.setting = setting
        self.max_length = max_length

    def __len__(self):
        return len(self.docs)

    def __getitem__(self, idx):
        doc_idx, doc = self.docs[idx]  # Unpack the tuple (i, doc)
        if self.setting == "prompted":
            prompt = f"The text in style of {doc['style']}: \n"
            text = prompt + doc['text']
        else:
            text = doc['text']
        prompt_length = len(self.tokenizer.encode(prompt)) if self.setting == "prompted" else 1 # so that if it is not on prompted setting, <bos> is not included in further steps
        encoded_inputs = self.tokenizer(
            text,
            return_tensors="pt",
            add_special_tokens=True,
            max_length=self.max_length+prompt_length-1,
            truncation=True,
            padding="max_length"
        )
        
        input_tokens = self.tokenizer.convert_ids_to_tokens(encoded_inputs['input_ids'].squeeze(0).tolist()) 
        return doc['text'], encoded_inputs, input_tokens, prompt_length

def custom_collate_fn(batch):
    """
    Custom collate function to handle the mixed data types returned by MyDataset.
    
    Args:
        batch: List of tuples (text, encoded_inputs, input_tokens)
    
    Returns:
        Tuple of (full_texts_batch, input_ids_batch, input_tokens_batch)
    """
    # Separate the components
    texts, encoded_inputs_list, input_tokens_list, prompt_length_list = zip(*batch)
    
    # Stack the encoded inputs (dictionaries with tensors)
    # The tokenizer already returns batched tensors, so we need to stack them
    input_ids_batch = {
        'input_ids': torch.cat([item['input_ids'] for item in encoded_inputs_list], dim=0),
        'attention_mask': torch.cat([item['attention_mask'] for item in encoded_inputs_list], dim=0)
    }
    
    return list(texts), input_ids_batch, list(input_tokens_list), list(prompt_length_list)

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


def load_model(model_config: ModelConfig, device):
    """
    Load the model for the given rank (given GPU).
    """
    logger.info(f"Loading model {model_config.model_name} on {device}")
    model = HookedTransformer.from_pretrained(
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
    """Log GPU memory usage for all available GPUs"""
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




def run_inference_on_gpu(rank, author_subsets, parsed_args, author_to_docs, dataset_config, batch_size, save_dir):
    """
    Run the inference for the given rank and save activations and cross-entropy loss and entropy.
    """

    # Setup DDP environment
    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")

    model_config = ModelConfig(model_name=parsed_args.model, layer_indices=parsed_args.layers)
    model = load_model(model_config, device)


    # Assign authors to this GPU 
    authors_for_this_gpu = author_subsets[rank]
    
    logger.info(f"GPU {rank} will process {len(authors_for_this_gpu)} authors: {authors_for_this_gpu}")

    # Process each author assigned to this GPU
    for author_idx, author in enumerate(authors_for_this_gpu):
        logger.info(f"\n=== GPU {rank} Processing Author {author_idx + 1}/{len(authors_for_this_gpu)}: {author} ===")
        docs = author_to_docs[author]
        
        logger.info(f"Number of documents for author {author}: {len(docs)}")

        # Initialize storage for each layer 
        activations = {}
        for layer_type in model_config.layer_types:
            activations[layer_type] = {}
            for layer_ind in model_config.layer_indices:
                activations[layer_type][layer_ind] = ActivationStorage(
                    model=model,
                    n_docs=dataset_config.max_n_docs_per_author,
                    max_seq_len=dataset_config.max_sequence_length - 1
                )
        # Initialize storage for cross-entropy loss and entropy
        cross_entropy_loss_author = np.zeros((dataset_config.max_n_docs_per_author, dataset_config.max_sequence_length-1), dtype=np.float32)
        entropy_author = np.zeros((dataset_config.max_n_docs_per_author, dataset_config.max_sequence_length-1), dtype=np.float32)
        logger.debug(f"Initialized storage for cross-entropy loss and entropy")
        
        # Add token count tracking
        doc_lengths = np.zeros(dataset_config.max_n_docs_per_author, dtype=np.int32)
        
        tokens_per_author = []
        full_texts_per_author = []

        # Create dataloader
        dataset = MyDataset(docs, model, dataset_config.max_sequence_length, parsed_args.setting)
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=False,
            collate_fn=custom_collate_fn
        )

        logger.info(f"GPU Memory usage after creating dataloader:")
        log_gpu_memory_usage()
        
        with torch.no_grad():
            try:
                for batch_idx, batch in enumerate(tqdm(loader, desc=f"Processing {author}")):

                    start_idx = batch_idx * batch_size
                    end_idx = start_idx + len(batch[0])

                    full_texts_batch, input_ids_batch, input_tokens_batch, prompt_length_batch = batch
                    prompt_length_doc = prompt_length_batch[0] 
                    
                    # Move tensors to the correct device (rank)
                    input_ids = input_ids_batch['input_ids'].to(device)
                    attention_mask = input_ids_batch['attention_mask'].to(device)
                    actual_lengths = (attention_mask.sum(dim=1) - prompt_length_doc).tolist() # substract either prompt length including <bos> token or only <bos> token

                    tokens_per_author.extend([input_tokens_doc[prompt_length_doc:] for input_tokens_doc in input_tokens_batch])
                    full_texts_per_author.extend(full_texts_batch)
                    
                    logits, cache = model.run_with_cache(input_ids)
                

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
                    
                    cross_entropy_loss_author[start_idx:end_idx, :] = \
                        ce_loss[:, prompt_length_doc - 1:].cpu().numpy()
                    entropy_author[start_idx:end_idx, :] = \
                        entropy[:, prompt_length_doc:].cpu().numpy()
                    doc_lengths[start_idx:end_idx] = actual_lengths
                    
                    # Store activations
                    batch_doc_indices = list(range(start_idx, end_idx))

                    
                    for layer_type in model_config.layer_types:
                        for layeri in model_config.layer_indices:
                            
                            logger.debug(f"Processing {layer_type} layer_type for layer {layeri} ")
                            layer_type_hook = {
                                "res": "resid_post",
                                "mlp": "mlp_out",
                                "att": "attn_out"
                            }
                            hook_name = utils.get_act_name(layer_type_hook[layer_type], layeri)
                            dense_acts = cache[hook_name][:, prompt_length_doc:].to(device)
                            
                            activations[layer_type][layeri].add_batch(
                                dense_acts,
                                batch_doc_indices,
                                actual_lengths
                            )

                    # Clean up memory after each batch
                    if batch_idx % 5 == 0:  # Every 5 batches 
                        torch.cuda.empty_cache()
                        gc.collect()
                        
                    del logits, cache, ce_loss, entropy

            except Exception as e:
                logger.error(f"Error processing author {author}: {e}")
                traceback.print_exc()
                # Clean up and re-raise
                torch.cuda.empty_cache()
                gc.collect()
                raise

        logger.info(f"GPU Memory usage after processing author {author}:")
        log_gpu_memory_usage()

        # Save results
        model_name_safe = parsed_args.model.replace('/', '_')


        logger.info("Saving entropy and cross-entropy loss...")
        np.save(
            save_dir / f"dense_{parsed_args.setting}__{model_name_safe}__entropy__{author}.npy",
            entropy_author
        )
        np.save(
            save_dir / f"dense_{parsed_args.setting}__{model_name_safe}__cross_entropy_loss__{author}.npy",
            cross_entropy_loss_author
        )

        # Save tokens and full texts
        logger.info("Saving tokens and full texts...")

        with open(save_dir / f"dense_{parsed_args.setting}__{model_name_safe}__tokens__{author}.json",
                 "w", encoding="utf-8") as f:
            json.dump(tokens_per_author, f, ensure_ascii=False, indent=4)
        
        with open(save_dir / f"dense_{parsed_args.setting}__{model_name_safe}__full_texts__{author}.json",
                 "w", encoding="utf-8") as f:
            json.dump(full_texts_per_author, f, ensure_ascii=False, indent=4)
        
        
        # Save activations
        logger.info("Saving activations...")
        save_times = []
        file_sizes = []
        for layer_type in model_config.layer_types:
            for layeri in model_config.layer_indices:
                base_filename = (f"dense_{parsed_args.setting}__{model_name_safe}__{layer_type}"
                               f"__activations__{author}__layer_{layeri}")
                filepath = save_dir / base_filename
                
                save_time, file_size = activations[layer_type][layeri].save(
                    filepath=Path(filepath),
                    author_id=author,
                    doc_lengths=doc_lengths,
                    layer_type=layer_type,
                    layer_index=layeri,
                    model_name=model_name_safe
                )
                save_times.append(save_time)
                file_sizes.append(file_size)
        
        # Log average save time and file sizes
        if save_times:
            avg_save_time = np.mean(save_times)
            avg_file_size = np.mean(file_sizes)
            total_size = np.sum(file_sizes)
            
            def format_size(size_bytes):
                for unit in ['B', 'KB', 'MB', 'GB']:
                    if size_bytes < 1024.0:
                        return f"{size_bytes:.2f} {unit}"
                    size_bytes /= 1024.0
                return f"{size_bytes:.2f} TB"
            
            logger.info(f"=== Activation Files Summary ===")
            logger.info(f"Total files saved: {len(save_times)}")
            logger.info(f"Average time to save one file: {avg_save_time:.2f}s")
            logger.info(f"Total save time: {np.sum(save_times):.2f}s")
            logger.info(f"Average file size: {format_size(avg_file_size)}")
            logger.info(f"Total storage used: {format_size(total_size)}")
            logger.info(f"================================")


        # Cleanup
        del activations, cross_entropy_loss_author, entropy_author
        del doc_lengths, tokens_per_author, full_texts_per_author
        del dataset, loader
        torch.cuda.empty_cache()
        gc.collect()
        
        logger.info(f"Completed processing author: {author}")


def main(parsed_args):

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    model_name_safe = parsed_args.model.replace('/', '_')
    dataset_dir = Path(parsed_args.dataset_name, parsed_args.category_name) if parsed_args.category_name is not None else Path(parsed_args.dataset_name)
    save_dir = Path(parsed_args.output_dir) / dataset_dir / model_name_safe / parsed_args.run_name
    save_dir.mkdir(parents=True, exist_ok=True)

    dataset_config = DatasetConfig(dataset_name=parsed_args.dataset_name, min_length_doc=parsed_args.min_length_doc, max_n_docs_per_author=parsed_args.n_docs_per_author)

    if parsed_args.dataset_name == "AuthorMix":
        author_to_docs, dataset_config = load_AuthorMix_data(dataset_config, parsed_args.n_docs, parsed_args.exclude_authors)
    elif parsed_args.dataset_name == "news":
        author_to_docs, dataset_config = load_news_data(dataset_config, parsed_args.category_name, parsed_args.n_docs, parsed_args.exclude_authors)
    else:
        raise ValueError(f"Dataset {parsed_args.dataset_name} is not supported.")

    for author in dataset_config.author_list:
        logger.info(f"  {author}: {len(author_to_docs[author])} documents")
    
    world_size = torch.cuda.device_count()

    valid_authors = [
    author for author in dataset_config.author_list
    if len(author_to_docs[author]) >= dataset_config.max_n_docs_per_author
    ]
    author_subsets = [valid_authors[i::world_size] for i in range(world_size)]

    # Run multi-GPU inference 
    logger.info("Starting multi-GPU inference...")
    start_time = time.time()
    
    mp.spawn(
    run_inference_on_gpu,
    args=(author_subsets, parsed_args, author_to_docs, dataset_config, parsed_args.batch_size, save_dir),  
    nprocs=world_size,
    join=True
)
    
    total_time = time.time() - start_time
    logger.info(f"=== Total runtime: {total_time:.2f}s ({total_time/60:.2f} minutes) ===")
    
    # Register this run in the tracking system
    try:
        tracker = get_tracker()
        run_id = tracker.register_run(
            model=parsed_args.model,
            dataset=parsed_args.dataset_name,
            run_name=parsed_args.run_name,
            layers=parsed_args.layers,
            activation_path=str(save_dir),
            category=parsed_args.category_name,
            authors=dataset_config.author_list,
            storage_format="dense",  # Classic activations are always dense
            n_docs_per_author=parsed_args.n_docs_per_author,
            min_length_doc=parsed_args.min_length_doc,
            setting=parsed_args.setting
        )
        logger.info(f"✓ Activation run registered with ID: {run_id}")
    except Exception as e:
        logger.warning(f"Failed to register activation run in tracking system: {e}")

            
if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Model arguments
    parser.add_argument(
        "--model",
        type=str,
        default="google/gemma-2-2b",
        choices=["google/gemma-2-2b", "google/gemma-2-9b"],
        help="Model to use for feature extraction"
    )
    
    parser.add_argument(
        '--layers',
        nargs='+',
        type=int,
        required=True,
        help='Layer indices to process'
    )
    
    # Dataset arguments
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="news",
        choices=["AuthorMix", "news"],
        help="Dataset to use"
    )
    parser.add_argument(
        "--category_name",
        type=str,
        default="politics",
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
        default=500,
        help="Documents per author"
    )
    parser.add_argument(
        "--min_length_doc",
        type=int,
        default=35,
        help="Minimum document length"
    )
    parser.add_argument(
        "--exclude_authors",
        nargs='+',
        type=str,
        default=None,
        help="Authors to exclude"
    )
    
    # Processing arguments
    parser.add_argument(
        "--batch_size",
        type=int,
        default=2,
        help="Batch size for processing"
    )
    parser.add_argument(
        "--setting",
        type=str,
        default="baseline",
        choices=["prompted", "baseline"],
        help="Feature extraction setting"
    )
    
    # Storage arguments
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/raw_dense_features",
        help="Directory to save results"
    )
    parser.add_argument(
        "--run_name",
        type=str,
        default="politics_500",
        help="Name of the run to create a folder in outputs"
    )

    parsed_args = parser.parse_args()
    main(parsed_args)
