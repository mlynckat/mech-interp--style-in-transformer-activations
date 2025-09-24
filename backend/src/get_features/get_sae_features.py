""" 
Counts the number of times each SAE feature fires across some corpus.
"""
import os
import argparse
from collections import defaultdict
import psutil
import gc
import traceback
import json
import logging
from tqdm.auto import tqdm
import numpy as np
from dotenv import  load_dotenv
import torch
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from huggingface_hub import login
from sae_lens import SAE, HookedSAETransformer

from backend.src.utils.load_dataset import load_news_data, load_AuthorMix_data
from backend.src.utils.run_configuration import ModelConfig, SAELayerConfig, DatasetConfig

# Set up logging
logger = logging.getLogger(__name__)

# Enable CUDA debugging for better error reporting
#os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
#os.environ['TORCH_USE_CUDA_DSA'] = '1'


load_dotenv()
login(token=os.environ["HF_TOKEN"])


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


def initialize_storage_for_layer_type(layer_type_names, saes, max_docs_per_author, max_seq_len):
    """
    Initialize storage for all available layer types and SAEs.

    Args:
        layer_type_names: Dictionary with layer types as keys (e.g. "res", "mlp", "att") and list of SAE names as values
        saes: Dictionary with layer types as keys (e.g. "res", "mlp", "att") and list of SAEs as values
        max_docs_per_author: Maximum number of documents per author
        max_seq_len: Maximum sequence length
    
    Returns:
        Dictionary of activations initiated with zeroes for each layer type
    """
    activations = {}
    for layer_type in layer_type_names:
        activations[layer_type] = defaultdict()
        for sae_name, sae in zip(layer_type_names[layer_type], saes[layer_type]):
            activations[layer_type][sae_name] = np.zeros((max_docs_per_author, max_seq_len, sae.cfg.d_sae), dtype=np.float16)
    
    return activations



def safe_save_npz(filepath, **data):
    try:
        # Check if directory exists and is writable
        directory = os.path.dirname(filepath)
        if not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)

        # Check available disk space
        stat = os.statvfs(directory)
        available_bytes = stat.f_bavail * stat.f_frsize
        logger.info(f"Available disk space: {available_bytes / (1024 ** 3):.2f} GB")

        # Check data before saving
        total_size = 0
        for key, value in data.items():
            if hasattr(value, 'nbytes'):
                total_size += value.nbytes
                logger.debug(f"{key}: shape={value.shape}, dtype={value.dtype}, size={value.nbytes / (1024 ** 2):.2f} MB")


                # Check for problematic values
                if np.issubdtype(value.dtype, np.floating):
                    nan_count = np.isnan(value).sum()
                    inf_count = np.isinf(value).sum()
                    if nan_count > 0 or inf_count > 0:
                        logger.warning(f"{key} contains {nan_count} NaN and {inf_count} inf values")

        logger.info(f"Total estimated size: {total_size / (1024 ** 2):.2f} MB")

        if "activations" in data:
            # Check for total non-zero values
            arrays = data['activations']
            count_nonzero = np.count_nonzero(arrays)
            logger.info(f"Non zero values is {count_nonzero}.")
            logger.info(f"Shape of the array: {arrays.shape}")

        # Try saving with compression disabled first
        np.savez_compressed(filepath, **data)
        logger.info(f"Successfully saved to {filepath}")

    except Exception as e:
        logger.error(f"Error saving to {filepath}: {e}")
        traceback.print_exc()

        # Try alternative save method
        try:
            # Save individual arrays
            for key, value in data.items():
                np.save(f"{filepath}_{key}.npy", value)
            logger.info("Saved as separate .npy files instead")
        except Exception as e2:
            logger.error(f"Alternative save also failed: {e2}")

def load_canonical_sae(sae_layer_config: SAELayerConfig, device: torch.device):
    """
    Attempts to load the canonical SAE for the given layer_type / layer / width.
    Returns (sae, sae_id) if successful, else None.
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

def load_saes(model_config: ModelConfig, layers, width, device):
    """
    layer_type: 'res', 'mlp', or 'att'
    width: e.g. 16384 (for 16k)
    """
    saes = {}
    saes_ids = {}
    for layer_type in model_config.layer_types:
        saes[layer_type] = []
        saes_ids[layer_type] = []
        for layer in layers:
            sae_layer_config = SAELayerConfig(layer_type=layer_type, layer_index=layer, width=width, model_name=model_config.model_name, canonical=True)
            sae, sae_id = load_canonical_sae(sae_layer_config, device)
            saes[layer_type].append(sae)
            saes_ids[layer_type].append(sae_id)
    
        logger.info(f"Loaded {len(saes[layer_type])} SAEs for layer_type {layer_type}")
    return saes, saes_ids

def run_inference(rank, parsed_args, author_to_docs, dataset_config, batch_size, world_size):
    """
    Run the inference for the given rank and save SAE features and cross-entropy loss and entropy.
    """

    # Setup DDP environment
    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")

    model_config = ModelConfig(model_name=parsed_args.model)
    model = load_model(model_config, device)


    # Assign authors to this GPU using round-robin distribution
    # Each GPU processes multiple authors if there are more authors than GPUs
    authors_for_this_gpu = []
    for i in range(len(dataset_config.author_list)):
        if len(author_to_docs[dataset_config.author_list[i]]) < dataset_config.max_n_docs_per_author:
            continue
        if i % world_size == rank:
            authors_for_this_gpu.append(dataset_config.author_list[i])
    
    logger.info(f"GPU {rank} will process {len(authors_for_this_gpu)} authors: {authors_for_this_gpu}")
    
    # Load SAEs once for all authors to avoid repeated loading
    logger.info(f"GPU {rank} loading SAEs for all authors...")
    saes, saes_ids = load_saes(model_config, parsed_args.layers, parsed_args.sae_features_width, device)

    # Process each author assigned to this GPU
    for author_idx, author in enumerate(authors_for_this_gpu):
        logger.info(f"\n=== GPU {rank} Processing Author {author_idx + 1}/{len(authors_for_this_gpu)}: {author} ===")
        docs = author_to_docs[author]
        
        logger.info(f"Number of documents for author {author}: {len(docs)}")

        # Initialize storage for each SAE
        # Each will have shape: [max_docs_per_author, max_seq_len-1, n_sae_features]
        activations = initialize_storage_for_layer_type(saes_ids, saes, dataset_config.max_n_docs_per_author, dataset_config.max_sequence_length-1)

        # Initialize storage for cross-entropy loss and entropy
        cross_entropy_loss_author = np.zeros((dataset_config.max_n_docs_per_author, dataset_config.max_sequence_length-1), dtype=np.float16)
        entropy_author = np.zeros((dataset_config.max_n_docs_per_author, dataset_config.max_sequence_length-1), dtype=np.float16)
        logger.debug(f"Initialized storage for cross-entropy loss and entropy")
        
        dataset = MyDataset(docs, model, dataset_config.max_sequence_length, parsed_args.setting)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=False, collate_fn=custom_collate_fn)
        logger.debug(f"Created dataloader for author {author}")
        
        # Add token count tracking
        tokens_length_per_doc = np.zeros(dataset_config.max_n_docs_per_author, dtype=np.int32)
        
        tokens_per_author = []
        full_texts_per_author = []

        
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
                    actual_tokens_length_batch = (attention_mask.sum(dim=1) - prompt_length_doc).tolist() # substract either prompt length including <bos> token or only <bos> token

                    tokens_per_author.extend([input_tokens_doc[prompt_length_doc:] for input_tokens_doc in input_tokens_batch])
                    full_texts_per_author.extend(full_texts_batch)

                    log_gpu_memory_usage()

                    # Forward pass
                    full_list_saes = []
                    for layer_type in model_config.layer_types:
                        full_list_saes.extend([sae for sae in saes[layer_type]])
                    
                    logits, cache = model.run_with_cache_with_saes(input_ids, saes=full_list_saes)
                    

                    # Validate tensors before processing
                    if torch.isnan(logits).any() or torch.isinf(logits).any():
                        logger.warning("Logits contain NaN or Inf values, skipping this batch")
                        continue
                    
                    if torch.isnan(attention_mask).any() or torch.isinf(attention_mask).any():
                        logger.warning("Attention mask contains NaN or Inf values, skipping this batch")
                        continue

                    log_gpu_memory_usage()

                    # Get cross-entropy loss
                    cross_entropy_loss_temp = lm_cross_entropy_loss(logits, input_ids, attention_mask=attention_mask, per_token=True)
                    mask = attention_mask.bool().to(cross_entropy_loss_temp.device)
                    mask_ce = mask[:, 1:]
                    cross_entropy_loss_temp = cross_entropy_loss_temp.masked_fill(~mask_ce, 0.0)
                    
                    
                    entropy_temp = compute_entropy(logits)
                    mask_entropy = attention_mask.bool().to(entropy_temp.device)

                    entropy_temp = entropy_temp.masked_fill(~mask_entropy, 0.0)

                    
                    cross_entropy_loss_author[start_idx:end_idx, :] = cross_entropy_loss_temp[:, prompt_length_doc-1:].cpu().numpy()
                    entropy_author[start_idx:end_idx, :] = entropy_temp[:, prompt_length_doc:].cpu().numpy()
                    tokens_length_per_doc[start_idx:end_idx] = actual_tokens_length_batch 

                    
                    for layer_type in model_config.layer_types:
                        for layeri, sae_id, sae in zip(model_config.layer_indices, saes_ids[layer_type], saes[layer_type]):
                            logger.debug(f"Processing {layer_type} layer_type SAE for layer {layeri} and SAE {sae_id} with shape {sae.cfg.d_sae}")
                            """for name, param in cache.items():
                                logger.debug(f"{name}: {param.shape}")"""
                            target_acts = cache[f'{sae.cfg.metadata.hook_name}.hook_sae_acts_post']  # shape: [B, T, D]
                            sae_acts = target_acts.to(device)  # Ensure same device as SAE

                            sae_acts = sae_acts[:, prompt_length_doc:]  # remove first token
                            for local_doc_idx, (sae_act, actual_tokens) in enumerate(zip(sae_acts, actual_tokens_length_batch)):
                                doc_idx = start_idx + local_doc_idx
                                seq_len = min(sae_act.shape[0], actual_tokens)
                                activations[layer_type][sae_id][doc_idx, :seq_len, :] = sae_act[:seq_len].cpu().numpy()

                    # Clean up memory after each batch
                    if batch_idx % 5 == 0:  # Every 5 batches 
                        torch.cuda.empty_cache()
                        gc.collect()
                        logger.debug(f"Memory cleanup after batch {batch_idx}")
                        log_gpu_memory_usage()
                    
                    # Additional cleanup after each batch to prevent CUDA errors
                    del logits, cache
                    if 'cross_entropy_loss_temp' in locals():
                        del cross_entropy_loss_temp
                    if 'entropy_temp' in locals():
                        del entropy_temp

            except Exception as e:
                logger.error(f"Error processing author {author}: {e}")
                traceback.print_exc()
                # Clean up and re-raise
                torch.cuda.empty_cache()
                gc.collect()
                raise

        logger.debug("Memory status before saving:")
        check_memory()

        # Save results
        logger.info("Saving entropy and cross-entropy loss...")
        output_file = f"sae_{parsed_args.setting}__{parsed_args.model.replace('/', '_')}__entropy_loss__{author}.npy"
        np.save(os.path.join(parsed_args.save_dir, output_file), entropy_author)
        output_file = f"sae_{parsed_args.setting}__{parsed_args.model.replace('/', '_')}__cross_entropy_loss__{author}.npy"
        np.save(os.path.join(parsed_args.save_dir, output_file), cross_entropy_loss_author)

        # Save tokens and full texts
        output_file = f"sae_{parsed_args.setting}__{parsed_args.model.replace('/', '_')}__tokens__{author}.json"
        with open(os.path.join(parsed_args.save_dir, output_file), "w", encoding="utf-8") as f:
            json.dump(tokens_per_author, f, ensure_ascii=False, indent=4)
        output_file = f"sae_{parsed_args.setting}__{parsed_args.model.replace('/', '_')}__full_texts__{author}.json"
        with open(os.path.join(parsed_args.save_dir, output_file), "w", encoding="utf-8") as f:
            json.dump(full_texts_per_author, f, ensure_ascii=False, indent=4)

        # Save activations
        for layer_type in model_config.layer_types:
                
            for layeri, sae_id in zip(model_config.layer_indices, saes_ids[layer_type]):
                output_file = f"sae_{parsed_args.setting}__{parsed_args.model.replace('/', '_')}__{layer_type}__activations__{author}__layer_{layeri}.npz"
                # Save metadata
                metadata = {
                    'author': author,
                    'docs_count': len(docs),
                    'layer': layeri,
                    'tokens_per_doc': tokens_length_per_doc,
                    'sae_name': sae_id
                }

                # Save all data
                save_data = {
                    'activations': activations[layer_type][sae_id],
                    'metadata': metadata
                }
                safe_save_npz(os.path.join(parsed_args.save_dir, output_file), **save_data)

        
        

        # Clear author-specific variables
        del tokens_length_per_doc
        del activations
        del cross_entropy_loss_author
        del entropy_author
        del tokens_per_author
        del full_texts_per_author
        del dataset
        del loader
        torch.cuda.empty_cache()
        
        # Log memory usage after cleanup
        logger.debug(f"Memory status after processing author {author}:")
        check_memory()


def main(parsed_args):
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    os.makedirs(parsed_args.save_dir, exist_ok=True)

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

    mp.spawn(
        run_inference,
        args=(parsed_args, author_to_docs, dataset_config, parsed_args.batch_size, world_size),
        nprocs=world_size,
        join=True
    )

            
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, help="The model to use for feature extraction.", default="google/gemma-2-2b", choices=["google/gemma-2-2b", "google/gemma-2-9b"])
    parser.add_argument("--sae_features_width", type=str, help="Number of features in SAE, formatted like 16k, 32k, 65k, 131k, etc.", default="16k")
    parser.add_argument('--layers', nargs='+', type=int, help='List of integers specifying layer indices.')
    parser.add_argument("--n_docs", type=int, help="The number of documents to analyze.", default=None)
    parser.add_argument("--n_docs_per_author", type=int, help="The number of documents to analyze per author.", default=500)
    parser.add_argument("--save_dir", type=str, help="The directory to save the histogram to.", default="data/raw_features/newsPolitics500canonical-2b")
    parser.add_argument("--batch_size", type=int, help="The size of the batch to be processed at once", default=1)
    parser.add_argument("--exclude_authors", nargs='+', type=str, help="The authors to exclude from the analysis", default=None)
    parser.add_argument("--min_length_doc", type=int, help="The minimum length of the document to be analyzed", default=50)
    parser.add_argument("--dataset_name", type=str, help="The name of the dataset to use for feature extraction", default="news", choices=["AuthorMix", "news"])
    parser.add_argument("--category_name", type=str, help="The name of the category to use for feature extraction for news dataset", default="politics")
    parser.add_argument("--setting", type=str, help="The setting in which the features are retrieved", default="baseline", choices=["prompted", "baseline"])
    parsed_args = parser.parse_args()
    main(parsed_args)
