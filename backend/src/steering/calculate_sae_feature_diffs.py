"""
Calculate SAE feature diffs between original articles and baseline-generated texts.

This script computes the difference between aggregated SAE features of original texts
and baseline-generated texts for each author. These diffs can then be used for
SAE-diff-based steering in text generation.

The diff is calculated as: diff = mean(original_features) - mean(baseline_features)
This represents the direction from baseline to original author style.

Usage:
    python -m backend.src.steering.calculate_sae_feature_diffs \
        --features_dir data/raw_features/generated_texts \
        --output_dir data/steering/sae_diffs \
        --layer 15 \
        --authors "Sam Levine" "Paige Lavender" "Lee Moran" "Amanda Terkel"
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from collections import defaultdict

import numpy as np
import scipy.sparse as sp
import torch

from backend.src.utils.shared_utilities import ActivationMetadata

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# Default configurations
DEFAULT_MODEL_NAME = "google_gemma-2-9b-it"
DEFAULT_LAYER_TYPE = "res"
DEFAULT_NUM_SAE_FEATURES = 16384
DEFAULT_AUTHORS = ["Sam Levine", "Paige Lavender", "Lee Moran", "Amanda Terkel"]


def load_activations(filepath: Path) -> Tuple[np.ndarray, ActivationMetadata]:
    """
    Load sparse activations from npz file.
    
    Args:
        filepath: Path to the .npz file (without extension)
        
    Returns:
        Tuple of (activations array, metadata)
    """
    data_path = Path(str(filepath) + ".sparse.npz")
    meta_path = Path(str(filepath) + ".meta.pkl")
    
    if not data_path.exists():
        raise FileNotFoundError(f"Activation data file not found: {data_path}")
    if not meta_path.exists():
        raise FileNotFoundError(f"Metadata file not found: {meta_path}")
    
    # Load sparse data
    sparse_data = sp.load_npz(data_path)
    
    # Load .pkl metadata - pass the activation filename
    meta_dict = ActivationMetadata.load(str(data_path))
    original_shape = meta_dict.original_shape

    print(f"Sparse data shape: {sparse_data.shape}")
    # Convert sparse data to dense array
    dense_data = sparse_data.toarray().reshape(original_shape)
    print(f"Dense data shape: {dense_data.shape}")
    
    return dense_data, meta_dict


def aggregate_features(
    activations: np.ndarray,
    metadata: ActivationMetadata,
    from_token: int = 10
) -> np.ndarray:
    """
    Aggregate features over tokens for each document.
    
    Args:
        activations: Dense array of shape (n_docs, max_seq_len, n_features)
        doc_lengths: Array of actual document lengths (optional)
        from_token: Start token for aggregation (to skip prompt tokens)
        
    Returns:
        Aggregated features of shape (n_docs, n_features)
    """


    n_docs, max_seq_len, n_features = metadata.original_shape
    aggregated = np.zeros((n_docs, n_features), dtype=np.float32)
    
    for doc_idx in range(n_docs):
        doc_length = metadata.doc_lengths[doc_idx]
        start_token = min(from_token, doc_length)
        end_token = doc_length
        aggregated[doc_idx] = activations[doc_idx, start_token:end_token, :].mean(axis=0)
    
    return aggregated


def calculate_author_diff(
    features_dir: Path,
    author: str,
    layer_ind: int,
    layer_type: str = DEFAULT_LAYER_TYPE,
    model_name: str = DEFAULT_MODEL_NAME,
    from_token: int = 10
) -> Tuple[np.ndarray, Dict]:
    """
    Calculate the diff between original and generated (baseline) features for an author.
    
    Args:
        features_dir: Directory containing extracted features
        author: Author name
        layer_ind: Layer index
        layer_type: Layer type (e.g., "res")
        model_name: Model name (sanitized for filenames)
        from_token: Start token for aggregation
        
    Returns:
        Tuple of (diff vector, metadata dict)
    """
    # Construct file paths
    # Pattern: sae_generated_texts__{model}__{layer_type}__activations__{author}__{field_suffix}__layer_{layer}
    original_base = (
        f"sae_generated_texts__{model_name}__{layer_type}"
        f"__activations__{author}__original__layer_{layer_ind}"
    )
    generated_base = (
        f"sae_generated_texts__{model_name}__{layer_type}"
        f"__activations__{author}__generated__layer_{layer_ind}"
    )
    
    original_path = features_dir / original_base
    generated_path = features_dir / generated_base
    
    logger.info(f"Loading original features from: {original_path}")
    original_activations, original_meta = load_activations(original_path)
    
    logger.info(f"Loading generated features from: {generated_path}")
    generated_activations, generated_meta = load_activations(generated_path)
    
    # Aggregate features per document
    logger.info(f"Aggregating features (from_token={from_token})...")
    original_aggregated = aggregate_features(original_activations, original_meta, from_token=from_token)
    generated_aggregated = aggregate_features(generated_activations, generated_meta, from_token=from_token)
    
    # Calculate mean across all documents for each type
    original_mean = original_aggregated.mean(axis=0)  # shape: (n_features,)
    generated_mean = generated_aggregated.mean(axis=0)  # shape: (n_features,)
    
    # Calculate diff: direction from baseline to original style
    diff = original_mean - generated_mean
    
    # Prepare metadata
    meta = {
        "author": author,
        "layer_ind": layer_ind,
        "layer_type": layer_type,
        "model_name": model_name,
        "from_token": from_token,
        "n_docs_original": int(original_meta.original_shape[0]),
        "n_docs_generated": int(generated_meta.original_shape[0]),
        "n_features": int(diff.shape[0]),
        "diff_l2_norm": float(np.linalg.norm(diff)),
        "diff_mean": float(diff.mean()),
        "diff_std": float(diff.std()),
        "diff_max": float(diff.max()),
        "diff_min": float(diff.min()),
        "original_mean_norm": float(np.linalg.norm(original_mean)),
        "generated_mean_norm": float(np.linalg.norm(generated_mean)),
    }
    
    logger.info(f"  Diff L2 norm: {meta['diff_l2_norm']:.4f}")
    logger.info(f"  Diff mean: {meta['diff_mean']:.6f}, std: {meta['diff_std']:.6f}")
    
    return diff, meta


def save_diffs(
    diffs: Dict[str, np.ndarray],
    metadata_diffs: Dict[str, Dict],
    output_dir: Path,
    layer_ind: int,
    layer_type: str = DEFAULT_LAYER_TYPE
):
    """
    Save computed diffs to disk.
    
    Args:
        diffs: Dictionary mapping author names to diff vectors
        metadata_diffs: Dictionary mapping author names to metadata_diffs dicts
        output_dir: Output directory
        layer_ind: Layer index
        layer_type: Layer type
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save individual diff files per author
    for author, diff in diffs.items():
        author_safe = author.replace(" ", "_")
        diff_filename = f"sae_diff__{layer_type}__layer_{layer_ind}__{author_safe}.npy"
        meta_filename = f"sae_diff__{layer_type}__layer_{layer_ind}__{author_safe}_meta.json"
        
        diff_path = output_dir / diff_filename
        meta_path = output_dir / meta_filename
        
        np.save(diff_path, diff)
        with open(meta_path, "w") as f:
            json.dump(metadata_diffs[author], f, indent=2)
        
        logger.info(f"Saved diff for {author} to {diff_path}")
    
    # Save combined file for easy loading
    combined_filename = f"sae_diffs_combined__{layer_type}__layer_{layer_ind}.npz"
    combined_meta_filename = f"sae_diffs_combined__{layer_type}__layer_{layer_ind}_meta.json"
    
    combined_path = output_dir / combined_filename
    combined_meta_path = output_dir / combined_meta_filename
    
    # Convert author names to valid numpy keywords
    safe_diffs = {author.replace(" ", "_"): diff for author, diff in diffs.items()}
    np.savez(combined_path, **safe_diffs)
    
    # Save combined metadata
    combined_meta = {
        "layer_ind": layer_ind,
        "layer_type": layer_type,
        "authors": list(diffs.keys()),
        "author_metadata": {author.replace(" ", "_"): meta for author, meta in metadata_diffs.items()}
    }
    with open(combined_meta_path, "w") as f:
        json.dump(combined_meta, f, indent=2)
    
    logger.info(f"Saved combined diffs to {combined_path}")


def load_sae_diffs(
    output_dir: Path,
    layer_ind: int,
    layer_type: str = DEFAULT_LAYER_TYPE,
    author: Optional[str] = None
) -> Dict[str, np.ndarray]:
    """
    Load SAE diffs from disk.
    
    Args:
        output_dir: Directory containing saved diffs
        layer_ind: Layer index
        layer_type: Layer type
        author: Specific author to load (optional, loads all if None)
        
    Returns:
        Dictionary mapping author names to diff vectors
    """
    if author is not None:
        # Load single author
        author_safe = author.replace(" ", "_")
        diff_filename = f"sae_diff__{layer_type}__layer_{layer_ind}__{author_safe}.npy"
        diff_path = output_dir / diff_filename
        
        if not diff_path.exists():
            raise FileNotFoundError(f"Diff file not found: {diff_path}")
        
        diff = np.load(diff_path)
        return {author: diff}
    else:
        # Load combined file
        combined_filename = f"sae_diffs_combined__{layer_type}__layer_{layer_ind}.npz"
        combined_meta_filename = f"sae_diffs_combined__{layer_type}__layer_{layer_ind}_meta.json"
        
        combined_path = output_dir / combined_filename
        combined_meta_path = output_dir / combined_meta_filename
        
        if not combined_path.exists():
            raise FileNotFoundError(f"Combined diffs file not found: {combined_path}")
        
        # Load diffs
        data = np.load(combined_path)
        
        # Load metadata to get original author names
        with open(combined_meta_path, "r") as f:
            meta = json.load(f)
        
        # Map back to original author names
        diffs = {}
        for author in meta["authors"]:
            author_safe = author.replace(" ", "_")
            diffs[author] = data[author_safe]
        
        return diffs


def main():
    """Main function to calculate SAE feature diffs."""
    parser = argparse.ArgumentParser(
        description="Calculate SAE feature diffs between original and baseline-generated texts."
    )
    
    parser.add_argument(
        "--features_dir",
        type=str,
        default="data/raw_features/generated_texts",
        help="Directory containing extracted SAE features"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/steering/sae_diffs",
        help="Directory to save computed diffs"
    )
    parser.add_argument(
        "--layer",
        type=int,
        default=15,
        help="Layer index to process"
    )
    parser.add_argument(
        "--layer_type",
        type=str,
        default=DEFAULT_LAYER_TYPE,
        choices=["res", "mlp", "att"],
        help="Layer type"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default=DEFAULT_MODEL_NAME,
        help="Model name (sanitized for filenames)"
    )
    parser.add_argument(
        "--authors",
        type=str,
        nargs="+",
        default=DEFAULT_AUTHORS,
        help="Authors to process"
    )
    parser.add_argument(
        "--from_token",
        type=int,
        default=10,
        help="Token position to start aggregation from (to skip prompt tokens)"
    )
    
    args = parser.parse_args()
    
    features_dir = Path(args.features_dir)
    output_dir = Path(args.output_dir)
    
    logger.info(f"Features directory: {features_dir}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Layer: {args.layer} ({args.layer_type})")
    logger.info(f"Authors: {args.authors}")
    
    # Calculate diffs for each author
    diffs = {}
    metadata_diffs = {}
    
    for author in args.authors:
        logger.info(f"\n=== Processing author: {author} ===")
        try:
            diff, meta = calculate_author_diff(
                features_dir=features_dir,
                author=author,
                layer_ind=args.layer,
                layer_type=args.layer_type,
                model_name=args.model_name,
                from_token=args.from_token
            )
            diffs[author] = diff
            metadata_diffs[author] = meta
        except FileNotFoundError as e:
            logger.warning(f"Skipping author {author}: {e}")
            continue
    
    if not diffs:
        logger.error("No diffs computed. Check that feature files exist.")
        return
    
    # Save diffs
    logger.info("\n=== Saving diffs ===")
    save_diffs(
        diffs=diffs,
        metadata_diffs=metadata_diffs,
        output_dir=output_dir,
        layer_ind=args.layer,
        layer_type=args.layer_type
    )
    
    logger.info("\nDone!")


if __name__ == "__main__":
    main()


