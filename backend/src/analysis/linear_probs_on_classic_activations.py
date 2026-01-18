"""
Linear probing and RSA on activations for author style classification.

This script applies linear probing and Representational Similarity Analysis (RSA) 
to classify authors based on their writing style using one of three activation types:
- Classic transformer activations (residual, MLP, attention outputs) - dense format
- SAE (Sparse Autoencoder) features - sparse or dense format  
- Residual activations (content-regressed out, from further_layer_style_search_methods.py)

The script:
1. Loads activations by run_id or path (supports classic, SAE, and residual activations)
2. Splits data into train/test (80/20 by documents, or uses pre-split for residuals)
3. For each layer type and index:
   - Fits multiple classifiers (one author vs rest) for linear probing
   - Computes RSA by comparing cosine similarity matrix to ideal author matrix
4. Saves results to JSON including:
   - Linear probing metrics (accuracy, precision, recall, F1, ROC-AUC)
   - RSA metrics (Spearman's ρ and p-value with ideal author matrix)
5. Generates visualizations:
   - ROC curves (aggregated by authors, layers, classifiers)
   - F1 score bar plots
   - RSA similarity matrices and correlation scatter plots
   - RSA summary across layers

Usage for residual activations:
------------------------------
    # Point to the residualization output directory (containing 'residuals' subfolder)
    python -m backend.src.analysis.linear_probs_on_classic_activations \\
        --path_to_data /path/to/output_data/.../residualization \\
        --activation_type residual \\
        --include_layer_types res mlp

Note: Residual activations come from further_layer_style_search_methods.py and
represent the component of activations orthogonal to content (i.e., style).
The --path_to_data should point to the directory containing the 'residuals' subfolder,
NOT to the 'residuals' folder itself.
"""

import argparse
import json
import logging
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_curve, auc
)
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import spearmanr
import scipy.sparse as sp
import seaborn as sns
from tqdm.auto import tqdm

# Import project utilities
from backend.src.utils.plot_styling import PlotStyle, apply_style, create_figure
from backend.src.utils.shared_utilities import (
    AuthorColorManager,
    ActivationMetadata,
    DataLoader,
    FilenamesLoader,
    ActivationFilenamesLoader
)
from backend.src.analysis.analysis_run_tracking import (
    get_data_and_output_paths,
    AnalysisRunTracker
)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Apply global matplotlib styling (Nordic Ocean Theme)
apply_style()

# Constants
TRAIN_SPLIT_RATIO = 0.8
RANDOM_STATE = 42
FIGURE_DPI = 300


# =============================================================================
# DATA LOADING
# =============================================================================

class ClassicActivationFilenamesLoader(FilenamesLoader):
    """Handles loading and filtering of classic (dense) activation filenames."""

    def __init__(
        self, 
        data_dir: Path, 
        include_authors: List[str] = None, 
        include_layer_types: List[str] = None, 
        include_layer_inds: List[int] = None, 
        include_setting: str = "baseline"
    ):
        """
        Initialize the classic activation filename loader.
        
        Args:
            data_dir: Directory containing activation files
            include_authors: List of authors to include (None = all)
            include_layer_types: List of layer types to include (res, mlp, att)
            include_layer_inds: List of layer indices to include
            include_setting: Setting filter (baseline or prompted)
        """
        self.ACTIVATION_TYPES = ["res", "mlp", "att"]
        self.data_dir = Path(data_dir)
        self.include_authors = include_authors
        self.include_layer_types = include_layer_types
        self.include_layer_inds = include_layer_inds
        self.include_setting = include_setting
        self.filenames = self.load_filenames()

    @staticmethod
    def parse_filename(filename: str) -> Dict[str, str]:
        """
        Parse classic activation filename into components.
        
        Expected format: dense_{setting}__{model}__{layer_type}__activations__{author}__layer_{layer_ind}.npz
        Example: dense_baseline__google_gemma-2-9b-it__res__activations__Amanda Terkel__layer_5.npz
        """
        if not filename.endswith(".npz"):
            raise ValueError(f"Expected .npz file, got: {filename}")

        # Remove extension
        name_without_ext = filename[:-4]
        
        # Split by double underscores
        parts = name_without_ext.split("__")
        
        if len(parts) < 6:
            raise ValueError(
                f"Invalid filename format. Expected at least 6 parts "
                f"separated by '__', got {len(parts)}: {filename}"
            )
        
        # Extract components
        setting = "prompted" if "prompted" in parts[0] else "baseline"
        model = parts[1]
        layer_type = parts[2]
        author = parts[4]
        
        # Extract layer from the last part (layer_5 -> 5)
        layer_part = parts[5]
        if not layer_part.startswith("layer_"):
            raise ValueError(f"Invalid layer format. Expected 'layer_X', got: {layer_part}")
        layer_ind = layer_part[6:]
        
        return {
            "setting": setting,
            "model": model,
            "layer_type": layer_type,
            "author": author,
            "layer_ind": layer_ind
        }

    def load_filenames(self) -> List[str]:
        """Load and filter filenames based on specified criteria."""
        if not self.data_dir.exists():
            logger.warning(f"Data directory does not exist: {self.data_dir}")
            return []
            
        filenames = [f for f in os.listdir(self.data_dir) 
                     if f.endswith(".npz") and f.startswith("dense_")]
        filtered_filenames = []
        
        for filename in filenames:
            try:
                parsed = self.parse_filename(filename)
                
                # Apply filters
                if self.include_authors and parsed['author'] not in self.include_authors:
                    continue
                if self.include_layer_types and parsed['layer_type'] not in self.include_layer_types:
                    continue
                if self.include_layer_inds and int(parsed['layer_ind']) not in self.include_layer_inds:
                    continue
                if self.include_setting and self.include_setting != parsed['setting']:
                    continue
                
                filtered_filenames.append(filename)
                
            except (ValueError, KeyError, IndexError) as e:
                logger.debug(f"Could not parse filename '{filename}': {e}")
                continue
        
        return filtered_filenames

    def get_structured_filenames(self) -> Dict[str, Dict[str, Dict[str, str]]]:
        """
        Parse filenames and organize in hierarchy: layer_type -> layer_ind -> author
        
        Returns:
            Dict[layer_type, Dict[layer_ind, Dict[author, filename_without_ext]]]
        """
        filenames_structured = defaultdict(lambda: defaultdict(lambda: defaultdict(str)))
        
        for filename in self.filenames:
            try:
                parsed = self.parse_filename(filename)
                layer_type = parsed["layer_type"]
                layer_ind = parsed["layer_ind"]
                author = parsed["author"]
                
                # Store filename without extension
                name_without_ext = filename[:-4]
                filenames_structured[layer_type][layer_ind][author] = name_without_ext
                
            except (ValueError, KeyError) as e:
                logger.warning(f"Could not parse filename '{filename}': {e}")
                continue
                
        return filenames_structured


def load_classic_activations(filepath: Path) -> Tuple[np.ndarray, ActivationMetadata]:
    """
    Load classic activations from .npz file.
    
    Args:
        filepath: Path to the activation file (without extension)
        
    Returns:
        Tuple of (activations array, metadata)
    """
    data_path = Path(str(filepath) + '.npz')
    metadata = ActivationMetadata.load(str(data_path))
    
    data = np.load(data_path)
    activations = data['activations']
    
    return activations, metadata


def load_sae_activations(filepath: Path) -> Tuple[np.ndarray, ActivationMetadata]:
    """
    Load SAE activations from .npz or .sparse.npz file.
    
    Args:
        filepath: Path to the activation file (without extension)
        
    Returns:
        Tuple of (activations array or sparse matrix, metadata)
    """
    # Try sparse format first, then dense
    sparse_path = Path(str(filepath) + '.sparse.npz')
    dense_path = Path(str(filepath) + '.npz')
    
    if sparse_path.exists():
        data_path = sparse_path
    elif dense_path.exists():
        data_path = dense_path
    else:
        raise FileNotFoundError(f"Could not find activation file: {filepath}")
    
    metadata = ActivationMetadata.load(str(data_path))
    
    if metadata.storage_format == 'sparse':
        # Load sparse matrix
        activations = sp.load_npz(data_path)
    else:
        # Load dense array
        data = np.load(data_path)
        activations = data['activations']
    
    return activations, metadata


def load_activations(
    filepath: Path, 
    activation_type: str = "classic"
) -> Tuple[np.ndarray, ActivationMetadata]:
    """
    Unified loader for both classic and SAE activations.
    
    Args:
        filepath: Path to the activation file (without extension)
        activation_type: Either "classic" or "sae"
        
    Returns:
        Tuple of (activations array, metadata)
    """
    if activation_type == "classic":
        return load_classic_activations(filepath)
    elif activation_type == "sae":
        return load_sae_activations(filepath)
    else:
        raise ValueError(f"Unknown activation type: {activation_type}")


# =============================================================================
# RESIDUAL ACTIVATION LOADING (from further_layer_style_search_methods.py)
# =============================================================================

class ResidualActivationFilenamesLoader(FilenamesLoader):
    """Handles loading and filtering of residual activation filenames.
    
    Residual activations are generated by further_layer_style_search_methods.py
    and stored in a 'residuals' subdirectory with format:
    residuals__{layer_type}__layer_{layer_ind}.npz
    """
    
    def __init__(
        self,
        data_dir: Path,
        include_layer_types: List[str] = None,
        include_layer_inds: List[int] = None
    ):
        """
        Initialize the residual activation filename loader.
        
        Args:
            data_dir: Directory containing the 'residuals' subdirectory
            include_layer_types: List of layer types to include (res, mlp, att)
            include_layer_inds: List of layer indices to include
        """
        self.ACTIVATION_TYPES = ["res", "mlp", "att"]
        self.data_dir = Path(data_dir)
        # Look for residuals subdirectory
        self.residuals_dir = self.data_dir / "residuals"
        if not self.residuals_dir.exists():
            # Try parent directory's residuals folder if data_dir is already deep
            logger.warning(f"Residuals directory not found at {self.residuals_dir}")
            self.residuals_dir = self.data_dir
        self.include_layer_types = include_layer_types
        self.include_layer_inds = include_layer_inds
        self.filenames = self.load_filenames()
    
    @staticmethod
    def parse_filename(filename: str) -> Dict[str, str]:
        """
        Parse residual activation filename into components.
        
        Expected format: residuals__{layer_type}__layer_{layer_ind}.npz
        Example: residuals__res__layer_5.npz
        """
        if not filename.endswith(".npz"):
            raise ValueError(f"Expected .npz file, got: {filename}")
        
        # Skip metadata files
        if filename.endswith("__meta.json"):
            raise ValueError(f"Metadata file, not activation file: {filename}")
        
        # Remove extension
        name_without_ext = filename[:-4]
        
        # Split by double underscores
        parts = name_without_ext.split("__")
        
        if len(parts) != 3 or parts[0] != "residuals":
            raise ValueError(
                f"Invalid residual filename format. Expected 'residuals__{{layer_type}}__layer_{{N}}', "
                f"got: {filename}"
            )
        
        layer_type = parts[1]
        layer_part = parts[2]
        
        if not layer_part.startswith("layer_"):
            raise ValueError(f"Invalid layer format. Expected 'layer_X', got: {layer_part}")
        layer_ind = layer_part[6:]
        
        return {
            "layer_type": layer_type,
            "layer_ind": layer_ind
        }
    
    def load_filenames(self) -> List[str]:
        """Load and filter filenames based on specified criteria."""
        if not self.residuals_dir.exists():
            logger.warning(f"Residuals directory does not exist: {self.residuals_dir}")
            return []
        
        filenames = [f for f in os.listdir(self.residuals_dir)
                     if f.endswith(".npz") and f.startswith("residuals__")]
        filtered_filenames = []
        
        for filename in filenames:
            try:
                parsed = self.parse_filename(filename)
                
                # Apply filters
                if self.include_layer_types and parsed['layer_type'] not in self.include_layer_types:
                    continue
                if self.include_layer_inds and int(parsed['layer_ind']) not in self.include_layer_inds:
                    continue
                
                filtered_filenames.append(filename)
                
            except (ValueError, KeyError, IndexError) as e:
                logger.debug(f"Could not parse filename '{filename}': {e}")
                continue
        
        return filtered_filenames
    
    def get_structured_filenames(self) -> Dict[str, Dict[str, str]]:
        """
        Parse filenames and organize in hierarchy: layer_type -> layer_ind -> filename
        
        Returns:
            Dict[layer_type, Dict[layer_ind, filename_without_ext]]
        
        Note: For residuals, we don't have author-level files. Each file contains
        all authors for a given layer type and index.
        """
        filenames_structured = defaultdict(dict)
        
        for filename in self.filenames:
            try:
                parsed = self.parse_filename(filename)
                layer_type = parsed["layer_type"]
                layer_ind = parsed["layer_ind"]
                
                # Store filename without extension
                name_without_ext = filename[:-4]
                filenames_structured[layer_type][layer_ind] = name_without_ext
                
            except (ValueError, KeyError) as e:
                logger.warning(f"Could not parse filename '{filename}': {e}")
                continue
        
        return filenames_structured


@dataclass
class ResidualActivationData:
    """Container for residual activation data loaded from file."""
    residuals_train: np.ndarray
    residuals_test: np.ndarray
    train_indices: np.ndarray
    test_indices: np.ndarray
    train_author_labels: np.ndarray
    test_author_labels: np.ndarray
    metadata: Dict[str, Any]


def load_residual_activations(filepath: Path) -> ResidualActivationData:
    """
    Load residual activations from .npz file generated by residualization analysis.
    
    Args:
        filepath: Path to the residual activation file (without extension)
        
    Returns:
        ResidualActivationData containing train/test residuals and author labels
    """
    data_path = Path(str(filepath) + '.npz')
    meta_path = Path(str(filepath) + '__meta.json')
    
    if not data_path.exists():
        raise FileNotFoundError(f"Residual activation file not found: {data_path}")
    
    # Load data
    data = np.load(data_path, allow_pickle=True)
    
    # Load metadata if available
    metadata = {}
    if meta_path.exists():
        with open(meta_path, 'r') as f:
            metadata = json.load(f)
    
    return ResidualActivationData(
        residuals_train=data['residuals_train'],
        residuals_test=data['residuals_test'],
        train_indices=data['train_indices'],
        test_indices=data['test_indices'],
        train_author_labels=data['train_author_labels'],
        test_author_labels=data['test_author_labels'],
        metadata=metadata
    )


# =============================================================================
# CLASSIFIERS FOR LINEAR PROBING
# =============================================================================

@dataclass
class ClassifierConfig:
    """Configuration for a classifier."""
    name: str
    model_class: type
    params: Dict[str, Any]


def get_linear_probing_classifiers() -> List[ClassifierConfig]:
    """
    Get list of classifiers commonly used for linear probing.
    
    Returns:
        List of ClassifierConfig objects
    """
    return [
        ClassifierConfig(
            name="LogisticRegression",
            model_class=LogisticRegression,
            params={"max_iter": 2000, "random_state": RANDOM_STATE, "solver": "lbfgs"}
        ),
        ClassifierConfig(
            name="LogisticRegression_L1",
            model_class=LogisticRegression,
            params={"max_iter": 2000, "random_state": RANDOM_STATE, "penalty": "l1", "solver": "saga"}
        ),
        # ClassifierConfig(
        #     name="RidgeClassifier",
        #     model_class=RidgeClassifier,
        #     params={"random_state": RANDOM_STATE}
        # ),
        # ClassifierConfig(
        #     name="LinearSVM",
        #     model_class=SVC,
        #     params={"kernel": "linear", "probability": True, "random_state": RANDOM_STATE}
        # ),
        ClassifierConfig(
            name="MLP_1Layer",
            model_class=MLPClassifier,
            params={
                "hidden_layer_sizes": (64,),  # Single hidden layer with 64 neurons
                "activation": "relu",
                "solver": "adam",
                "max_iter": 500,
                "early_stopping": True,
                "validation_fraction": 0.1,
                "random_state": RANDOM_STATE
            }
        ),
    ]


# =============================================================================
# DATA PREPARATION
# =============================================================================

def aggregate_activations_per_document(
    activations: np.ndarray, 
    doc_lengths: np.ndarray,
    aggregation: str = "mean",
    metadata: ActivationMetadata = None
) -> np.ndarray:
    """
    Aggregate token-level activations to document-level representations.
    
    Supports both dense numpy arrays and sparse matrices.
    
    Args:
        activations: Array of shape (n_docs, max_seq_len, n_features) for dense,
                    or sparse matrix of shape (n_docs * max_seq_len, n_features)
        doc_lengths: Array of actual document lengths
        aggregation: Aggregation method ("mean", "max", "sum")
        metadata: ActivationMetadata (required for sparse activations)
        
    Returns:
        Array of shape (n_docs, n_features)
    """
    # Handle sparse matrices
    if sp.issparse(activations):
        return _aggregate_sparse_activations(activations, doc_lengths, aggregation, metadata)
    
    # Dense activations
    n_docs = activations.shape[0]
    n_features = activations.shape[2]
    aggregated = np.zeros((n_docs, n_features), dtype=np.float32)
    
    for doc_idx in range(n_docs):
        doc_len = doc_lengths[doc_idx]
        if doc_len == 0:
            continue
            
        doc_acts = activations[doc_idx, :doc_len, :]
        
        if aggregation == "mean":
            aggregated[doc_idx] = np.mean(doc_acts, axis=0)
        elif aggregation == "max":
            aggregated[doc_idx] = np.max(doc_acts, axis=0)
        elif aggregation == "sum":
            aggregated[doc_idx] = np.sum(doc_acts, axis=0)
        elif aggregation == "last":
            aggregated[doc_idx] = doc_acts[-1, :]
        else:
            raise ValueError(f"Unknown aggregation method: {aggregation}")
    
    return aggregated


def _aggregate_sparse_activations(
    activations: sp.spmatrix,
    doc_lengths: np.ndarray,
    aggregation: str,
    metadata: ActivationMetadata
) -> np.ndarray:
    """
    Aggregate sparse token-level activations to document-level representations.
    
    Args:
        activations: Sparse matrix of shape (n_docs * max_seq_len, n_features)
        doc_lengths: Array of actual document lengths
        aggregation: Aggregation method ("mean", "max", "sum")
        metadata: ActivationMetadata with original_shape info
        
    Returns:
        Dense array of shape (n_docs, n_features)
    """
    if metadata is None:
        raise ValueError("Metadata is required for sparse activations")
    
    n_docs, max_seq_len, n_features = metadata.original_shape
    aggregated = np.zeros((n_docs, n_features), dtype=np.float32)
    
    # Convert to CSR for efficient row slicing
    if not sp.isspmatrix_csr(activations):
        activations = activations.tocsr()
    
    for doc_idx in range(n_docs):
        doc_len = doc_lengths[doc_idx]
        if doc_len == 0:
            continue
        
        # Calculate row indices for this document
        start_row = doc_idx * max_seq_len
        end_row = start_row + doc_len
        
        # Extract document tokens
        doc_acts = activations[start_row:end_row, :]
        
        if aggregation == "mean":
            # Mean of sparse matrix
            doc_sum = np.asarray(doc_acts.sum(axis=0)).flatten()
            aggregated[doc_idx] = doc_sum / doc_len
        elif aggregation == "max":
            # Max of sparse matrix (convert to dense for this operation)
            doc_dense = doc_acts.toarray()
            aggregated[doc_idx] = np.max(doc_dense, axis=0)
        elif aggregation == "sum":
            doc_sum = np.asarray(doc_acts.sum(axis=0)).flatten()
            aggregated[doc_idx] = doc_sum
        else:
            raise ValueError(f"Unknown aggregation method for sparse: {aggregation}")
    
    return aggregated


def prepare_train_test_data(
    author_to_activations: Dict[str, np.ndarray],
    train_ratio: float = TRAIN_SPLIT_RATIO
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """
    Prepare train/test splits for all authors.
    
    Args:
        author_to_activations: Dict mapping author names to aggregated activations
        train_ratio: Ratio of documents to use for training
        
    Returns:
        Tuple of (X_train, X_test, y_train, y_test, author_list)
    """
    X_train_list, X_test_list = [], []
    y_train_list, y_test_list = [], []
    author_list = list(author_to_activations.keys())
    
    for author_idx, author in enumerate(author_list):
        activations = author_to_activations[author]
        n_docs = activations.shape[0]
        n_train = int(n_docs * train_ratio)
        
        # Split documents
        X_train_list.append(activations[:n_train])
        X_test_list.append(activations[n_train:])
        y_train_list.extend([author_idx] * n_train)
        y_test_list.extend([author_idx] * (n_docs - n_train))
    
    X_train = np.vstack(X_train_list)
    X_test = np.vstack(X_test_list)
    y_train = np.array(y_train_list)
    y_test = np.array(y_test_list)
    
    return X_train, X_test, y_train, y_test, author_list


# =============================================================================
# LINEAR PROBING
# =============================================================================

def run_linear_probing_one_vs_rest(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    author_list: List[str],
    classifiers: List[ClassifierConfig],
    scale_features: bool = True
) -> Dict[str, Dict[str, Dict[str, Any]]]:
    """
    Run linear probing for each author (one vs rest classification).
    
    Args:
        X_train, X_test: Feature arrays
        y_train, y_test: Label arrays (author indices)
        author_list: List of author names
        classifiers: List of classifier configurations
        scale_features: Whether to standardize features
        
    Returns:
        Results dict: {classifier_name: {author: {metrics + roc_data}}}
    """
    results = defaultdict(lambda: defaultdict(dict))
    
    # Scale features if requested
    if scale_features:
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
    else:
        X_train_scaled = X_train
        X_test_scaled = X_test
    
    for classifier_config in classifiers:
        logger.info(f"Running classifier: {classifier_config.name}")
        
        for author_idx, author in enumerate(tqdm(author_list, desc=f"  {classifier_config.name}")):
            # Create binary labels (one vs rest)
            y_train_binary = (y_train == author_idx).astype(int)
            y_test_binary = (y_test == author_idx).astype(int)
            
            # Skip if not enough samples of the positive class
            if y_train_binary.sum() < 5 or y_test_binary.sum() < 2:
                logger.warning(f"Skipping {author} - insufficient samples")
                continue
            
            try:
                # Initialize and train classifier
                clf = classifier_config.model_class(**classifier_config.params)
                clf.fit(X_train_scaled, y_train_binary)
                
                # Predictions
                y_pred = clf.predict(X_test_scaled)
                
                # Get probabilities for ROC curve
                if hasattr(clf, 'predict_proba'):
                    y_proba = clf.predict_proba(X_test_scaled)[:, 1]
                elif hasattr(clf, 'decision_function'):
                    y_scores = clf.decision_function(X_test_scaled)
                    # Convert to probability-like values
                    y_proba = 1 / (1 + np.exp(-y_scores))
                else:
                    y_proba = y_pred.astype(float)
                
                # Compute metrics for class 1 (the target author)
                accuracy = accuracy_score(y_test_binary, y_pred)
                precision = precision_score(y_test_binary, y_pred, pos_label=1, zero_division=0)
                recall = recall_score(y_test_binary, y_pred, pos_label=1, zero_division=0)
                f1 = f1_score(y_test_binary, y_pred, pos_label=1, zero_division=0)
                
                # ROC curve data
                fpr, tpr, _ = roc_curve(y_test_binary, y_proba)
                roc_auc = auc(fpr, tpr)
                
                results[classifier_config.name][author] = {
                    "accuracy": float(accuracy),
                    "precision": float(precision),
                    "recall": float(recall),
                    "f1_score": float(f1),
                    "roc_auc": float(roc_auc),
                    "fpr": fpr.tolist(),
                    "tpr": tpr.tolist()
                }
                
            except Exception as e:
                logger.error(f"Error training {classifier_config.name} for {author}: {e}")
                continue
    
    return results


# =============================================================================
# REPRESENTATIONAL SIMILARITY ANALYSIS (RSA)
# =============================================================================

@dataclass
class RSAResults:
    """Results from Representational Similarity Analysis."""
    similarity_matrix: np.ndarray
    ideal_matrix: np.ndarray
    spearman_rho: float
    spearman_pvalue: float
    author_labels: List[str]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "spearman_rho": float(self.spearman_rho),
            "spearman_pvalue": float(self.spearman_pvalue),
            "n_samples": len(self.author_labels),
            "n_authors": len(set(self.author_labels))
        }


def compute_cosine_similarity_matrix(X: np.ndarray) -> np.ndarray:
    """
    Compute cosine similarity matrix between all samples.
    
    Args:
        X: Feature matrix of shape (n_samples, n_features)
        
    Returns:
        Similarity matrix of shape (n_samples, n_samples)
    """
    return cosine_similarity(X)


def create_ideal_author_matrix(author_labels: List[str]) -> np.ndarray:
    """
    Create ideal author similarity matrix.
    
    The ideal matrix has 1 where two samples are from the same author,
    and 0 where they are from different authors.
    
    Args:
        author_labels: List of author labels for each sample
        
    Returns:
        Ideal matrix of shape (n_samples, n_samples)
    """
    n_samples = len(author_labels)
    ideal_matrix = np.zeros((n_samples, n_samples), dtype=np.float32)
    
    for i in range(n_samples):
        for j in range(n_samples):
            if author_labels[i] == author_labels[j]:
                ideal_matrix[i, j] = 1.0
    
    return ideal_matrix


def compute_rsa_correlation(
    similarity_matrix: np.ndarray, 
    ideal_matrix: np.ndarray
) -> Tuple[float, float]:
    """
    Compute Spearman's correlation between similarity matrix and ideal matrix.
    
    Uses only the upper triangle (excluding diagonal) to avoid redundancy
    and self-comparisons.
    
    Args:
        similarity_matrix: Cosine similarity matrix
        ideal_matrix: Ideal author similarity matrix
        
    Returns:
        Tuple of (spearman_rho, p_value)
    """
    # Get upper triangle indices (excluding diagonal)
    upper_tri_indices = np.triu_indices_from(similarity_matrix, k=1)
    
    # Extract upper triangle values
    sim_values = similarity_matrix[upper_tri_indices]
    ideal_values = ideal_matrix[upper_tri_indices]
    
    # Compute Spearman correlation
    rho, pvalue = spearmanr(sim_values, ideal_values)
    
    return rho, pvalue


def run_rsa_analysis(
    X: np.ndarray,
    author_labels: List[str]
) -> RSAResults:
    """
    Run complete RSA analysis.
    
    Args:
        X: Feature matrix of shape (n_samples, n_features)
        author_labels: List of author labels for each sample
        
    Returns:
        RSAResults object containing all RSA outputs
    """
    # Compute cosine similarity matrix
    similarity_matrix = compute_cosine_similarity_matrix(X)
    
    # Create ideal author matrix
    ideal_matrix = create_ideal_author_matrix(author_labels)
    
    # Compute Spearman correlation
    spearman_rho, spearman_pvalue = compute_rsa_correlation(
        similarity_matrix, ideal_matrix
    )
    
    return RSAResults(
        similarity_matrix=similarity_matrix,
        ideal_matrix=ideal_matrix,
        spearman_rho=spearman_rho,
        spearman_pvalue=spearman_pvalue,
        author_labels=author_labels
    )


# =============================================================================
# VISUALIZATION
# =============================================================================

class LinearProbingVisualizer:
    """Handles all visualization tasks for linear probing results."""
    
    def __init__(self, save_dir: Path, color_manager: AuthorColorManager):
        """
        Initialize the visualizer.
        
        Args:
            save_dir: Directory to save visualizations
            color_manager: AuthorColorManager instance for consistent colors
        """
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True, parents=True)
        self.color_manager = color_manager
    
    def _save_plot(self, fig: plt.Figure, filename: str) -> None:
        """Save plot to file with proper styling."""
        plt.tight_layout()
        plt.savefig(
            self.save_dir / filename, 
            dpi=FIGURE_DPI, 
            bbox_inches='tight',
            facecolor=PlotStyle.COLORS['bg_white']
        )
        plt.close()
        logger.info(f"Saved plot: {filename}")
    
    def plot_roc_by_authors(
        self, 
        results: Dict[str, Dict[str, Dict[str, Any]]],
        classifier_name: str,
        layer_type: str,
        layer_ind: str
    ) -> None:
        """
        Plot ROC curves for all authors (Aggregation 1).
        
        One file per classifier, layer type, layer index with all authors.
        """
        fig, ax = plt.subplots(figsize=(10, 8))
        
        classifier_results = results.get(classifier_name, {})
        authors = list(classifier_results.keys())
        author_colors = self.color_manager.get_author_colors(authors)
        
        for author in authors:
            author_data = classifier_results[author]
            fpr = author_data.get("fpr", [])
            tpr = author_data.get("tpr", [])
            roc_auc = author_data.get("roc_auc", 0)
            
            if fpr and tpr:
                ax.plot(fpr, tpr, color=author_colors[author], lw=2,
                       label=f'{author} (AUC = {roc_auc:.3f})')
        
        # Add diagonal reference line
        ax.plot([0, 1], [0, 1], 'k--', lw=1, alpha=0.5)
        
        PlotStyle.style_axis(
            ax,
            title=f'ROC Curves - {classifier_name} - {layer_type} Layer {layer_ind}',
            xlabel='False Positive Rate',
            ylabel='True Positive Rate',
            grid_axis='both'
        )
        ax.legend(loc='lower right', frameon=False, fontsize=8)
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        
        filename = f"roc_by_authors__{classifier_name}__{layer_type}__layer_{layer_ind}.png"
        self._save_plot(fig, filename)
    
    def plot_roc_by_layers(
        self,
        all_results: Dict[str, Dict[str, Dict[str, Dict[str, Dict[str, Any]]]]],
        classifier_name: str,
        layer_type: str,
        author: str
    ) -> None:
        """
        Plot ROC curves for all layer indices (Aggregation 2).
        
        One file per classifier, layer type, author with all layer indices.
        """
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Get layer indices
        layer_inds = list(all_results.get(classifier_name, {}).get(layer_type, {}).keys())
        n_layers = len(layer_inds)
        layer_colors = PlotStyle.get_gradient_colors(n_layers)
        
        for idx, layer_ind in enumerate(sorted(layer_inds, key=int)):
            layer_data = all_results.get(classifier_name, {}).get(layer_type, {}).get(layer_ind, {})
            author_data = layer_data.get(author, {})
            
            fpr = author_data.get("fpr", [])
            tpr = author_data.get("tpr", [])
            roc_auc = author_data.get("roc_auc", 0)
            
            if fpr and tpr:
                ax.plot(fpr, tpr, color=layer_colors[idx], lw=2,
                       label=f'Layer {layer_ind} (AUC = {roc_auc:.3f})')
        
        ax.plot([0, 1], [0, 1], 'k--', lw=1, alpha=0.5)
        
        PlotStyle.style_axis(
            ax,
            title=f'ROC Curves - {classifier_name} - {layer_type} - {author}',
            xlabel='False Positive Rate',
            ylabel='True Positive Rate',
            grid_axis='both'
        )
        ax.legend(loc='lower right', frameon=False, fontsize=8)
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        
        # Clean filename
        author_safe = author.replace(' ', '_').replace('/', '_')
        filename = f"roc_by_layers__{classifier_name}__{layer_type}__{author_safe}.png"
        self._save_plot(fig, filename)
    
    def plot_roc_by_classifiers(
        self,
        all_results: Dict[str, Dict[str, Dict[str, Dict[str, Dict[str, Any]]]]],
        layer_type: str,
        layer_ind: str,
        author: str,
        classifier_names: List[str]
    ) -> None:
        """
        Plot ROC curves for all classifiers (Aggregation 3).
        
        One file per layer type, layer index, author with all classifiers.
        """
        fig, ax = plt.subplots(figsize=(10, 8))
        
        n_classifiers = len(classifier_names)
        classifier_colors = PlotStyle.get_gradient_colors(n_classifiers)
        
        for idx, classifier_name in enumerate(classifier_names):
            layer_data = all_results.get(classifier_name, {}).get(layer_type, {}).get(layer_ind, {})
            author_data = layer_data.get(author, {})
            
            fpr = author_data.get("fpr", [])
            tpr = author_data.get("tpr", [])
            roc_auc = author_data.get("roc_auc", 0)
            
            if fpr and tpr:
                ax.plot(fpr, tpr, color=classifier_colors[idx], lw=2,
                       label=f'{classifier_name} (AUC = {roc_auc:.3f})')
        
        ax.plot([0, 1], [0, 1], 'k--', lw=1, alpha=0.5)
        
        PlotStyle.style_axis(
            ax,
            title=f'ROC Curves - {layer_type} Layer {layer_ind} - {author}',
            xlabel='False Positive Rate',
            ylabel='True Positive Rate',
            grid_axis='both'
        )
        ax.legend(loc='lower right', frameon=False, fontsize=8)
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        
        author_safe = author.replace(' ', '_').replace('/', '_')
        filename = f"roc_by_classifiers__{layer_type}__layer_{layer_ind}__{author_safe}.png"
        self._save_plot(fig, filename)
    
    def plot_f1_barplot(
        self,
        all_results: Dict[str, Dict[str, Dict[str, Dict[str, Dict[str, Any]]]]],
        layer_type: str,
        classifier_names: List[str],
        author_list: List[str]
    ) -> None:
        """
        Create bar plot of F1 scores per layer type.
        
        One subplot per classifier, grouped bars by layer with one bar per author.
        Each layer shows a horizontal line at the mean F1 value with annotation.
        """
        # Get all layer indices for this layer type
        layer_inds = set()
        for clf_name in classifier_names:
            if clf_name in all_results and layer_type in all_results[clf_name]:
                layer_inds.update(all_results[clf_name][layer_type].keys())
        layer_inds = sorted(layer_inds, key=int)
        
        if not layer_inds:
            logger.warning(f"No layer indices found for {layer_type}")
            return
        
        n_classifiers = len(classifier_names)
        n_layers = len(layer_inds)
        n_authors = len(author_list)
        
        # Create figure with subplots (one per classifier)
        fig_width = max(20, n_layers * 1.8)  # Wide figure for many layers
        fig_height = 5 * n_classifiers + 1.0  # Extra space for bottom legend
        fig, axes = create_figure(n_classifiers, 1, figsize=(fig_width, fig_height))
        plt.subplots_adjust(left=0.05, right=0.98, top=0.95, bottom=0.12, hspace=0.8)
        
        if n_classifiers == 1:
            axes = [axes]
        
        # Get author colors
        author_colors = self.color_manager.get_author_colors(author_list)
        
        bar_width = 0.8 / n_authors
        
        # Color for mean line and annotation
        mean_line_color = PlotStyle.COLORS['accent_dark']  # Dark blue for visibility
        
        for clf_idx, classifier_name in enumerate(classifier_names):
            ax = axes[clf_idx]
            
            # Store F1 scores per layer for mean calculation
            layer_f1_scores = {layer_idx: [] for layer_idx in range(n_layers)}
            
            for author_idx, author in enumerate(author_list):
                f1_scores = []
                x_positions = []
                
                for layer_idx, layer_ind in enumerate(layer_inds):
                    layer_data = all_results.get(classifier_name, {}).get(layer_type, {}).get(layer_ind, {})
                    author_data = layer_data.get(author, {})
                    f1 = author_data.get("f1_score", 0)
                    
                    f1_scores.append(f1)
                    x_positions.append(layer_idx + author_idx * bar_width - (n_authors - 1) * bar_width / 2)
                    
                    # Collect F1 scores for mean calculation
                    if f1 > 0:  # Only include non-zero scores
                        layer_f1_scores[layer_idx].append(f1)
                
                ax.bar(x_positions, f1_scores, bar_width, 
                      label=author if clf_idx == 0 else "",
                      color=author_colors[author], alpha=0.85,
                      edgecolor='none')
            
            # Add horizontal mean lines and annotations for each layer
            for layer_idx, layer_ind in enumerate(layer_inds):
                scores = layer_f1_scores[layer_idx]
                if scores:
                    mean_f1 = np.mean(scores)
                    
                    # Define horizontal span for this layer's bars
                    x_start = layer_idx - 0.4
                    x_end = layer_idx + 0.4
                    
                    # Draw horizontal line at mean
                    ax.hlines(y=mean_f1, xmin=x_start, xmax=x_end, 
                             colors=mean_line_color, linestyles='--', linewidth=2, alpha=0.8)
                    
                    # Add annotation with mean value
                    ax.annotate(f'{mean_f1:.2f}', 
                               xy=(layer_idx, mean_f1),
                               xytext=(0, 6),  # Offset above the line
                               textcoords='offset points',
                               ha='center', va='bottom',
                               fontsize=12, fontweight='bold',
                               color=mean_line_color)
            
            ax.set_xticks(range(n_layers))
            ax.set_xticklabels([f'L{ind}' for ind in layer_inds], fontsize=13)
            ax.tick_params(axis='y', labelsize=13)
            
            PlotStyle.style_axis(
                ax,
                title=f'{classifier_name}',
                xlabel='Layer Index',
                ylabel='F1 Score',
                grid_axis='y',
                title_loc='center'
            )
            # Apply larger font sizes for title and labels
            ax.title.set_fontsize(18)
            ax.xaxis.label.set_fontsize(14)
            ax.yaxis.label.set_fontsize(14)
            
            ax.set_ylim([0, 1.0])
        
        # Add legend at the bottom of the figure
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='upper right', ncol=min(n_authors, 8), 
                   frameon=False, fontsize=12, bbox_to_anchor=(1.02, 1))
        
        filename = f"f1_barplot__{layer_type}.png"
        self._save_plot(fig, filename)
    
    def plot_rsa_similarity_matrix(
        self,
        rsa_results: RSAResults,
        layer_type: str,
        layer_ind: str,
        title_suffix: str = ""
    ) -> None:
        """
        Plot the cosine similarity matrix with author groupings.
        
        Args:
            rsa_results: RSA results containing similarity matrix
            layer_type: Type of layer (res, mlp, att)
            layer_ind: Layer index
            title_suffix: Additional text for title
        """
        fig, axes = plt.subplots(1, 2, figsize=(16, 7))
        
        # Create colormap
        cmap = PlotStyle.create_full_gradient_cmap()
        
        # Sort samples by author for better visualization
        sorted_indices = np.argsort(rsa_results.author_labels)
        sorted_labels = [rsa_results.author_labels[i] for i in sorted_indices]
        sorted_sim_matrix = rsa_results.similarity_matrix[sorted_indices][:, sorted_indices]
        sorted_ideal_matrix = rsa_results.ideal_matrix[sorted_indices][:, sorted_indices]
        
        # Find author boundaries for grid lines
        unique_authors = []
        boundaries = [0]
        current_author = sorted_labels[0]
        for i, author in enumerate(sorted_labels):
            if author != current_author:
                boundaries.append(i)
                unique_authors.append(current_author)
                current_author = author
        boundaries.append(len(sorted_labels))
        unique_authors.append(current_author)
        
        # Plot similarity matrix
        im1 = axes[0].imshow(sorted_sim_matrix, cmap=cmap, aspect='auto', vmin=0, vmax=1)
        axes[0].set_title(f'Cosine Similarity Matrix\n{layer_type} Layer {layer_ind}',
                         fontsize=12, color=PlotStyle.COLORS['text_dark'])
        
        # Add author boundary lines
        for boundary in boundaries[1:-1]:
            axes[0].axhline(y=boundary - 0.5, color='white', linewidth=1, alpha=0.8)
            axes[0].axvline(x=boundary - 0.5, color='white', linewidth=1, alpha=0.8)
        
        # Add colorbar
        cbar1 = plt.colorbar(im1, ax=axes[0], shrink=0.8)
        cbar1.set_label('Cosine Similarity', fontsize=10)
        
        # Plot ideal matrix
        im2 = axes[1].imshow(sorted_ideal_matrix, cmap=cmap, aspect='auto', vmin=0, vmax=1)
        axes[1].set_title(f'Ideal Author Matrix\n(Same Author = 1, Different = 0)',
                         fontsize=12, color=PlotStyle.COLORS['text_dark'])
        
        for boundary in boundaries[1:-1]:
            axes[1].axhline(y=boundary - 0.5, color='white', linewidth=1, alpha=0.8)
            axes[1].axvline(x=boundary - 0.5, color='white', linewidth=1, alpha=0.8)
        
        cbar2 = plt.colorbar(im2, ax=axes[1], shrink=0.8)
        cbar2.set_label('Similarity', fontsize=10)
        
        # Style axes
        for ax in axes:
            ax.set_xlabel('Document Index', fontsize=10, color=PlotStyle.COLORS['text_medium'])
            ax.set_ylabel('Document Index', fontsize=10, color=PlotStyle.COLORS['text_medium'])
        
        # Add correlation annotation
        rho = rsa_results.spearman_rho
        pvalue = rsa_results.spearman_pvalue
        pvalue_str = f"p < 0.001" if pvalue < 0.001 else f"p = {pvalue:.3f}"
        
        fig.suptitle(
            f"RSA Analysis - Spearman's ρ = {rho:.4f} ({pvalue_str}){title_suffix}",
            fontsize=14, color=PlotStyle.COLORS['text_dark'], y=1.02
        )
        
        filename = f"rsa_similarity_matrix__{layer_type}__layer_{layer_ind}.png"
        self._save_plot(fig, filename)
    
    def plot_rsa_correlation_scatter(
        self,
        rsa_results: RSAResults,
        layer_type: str,
        layer_ind: str
    ) -> None:
        """
        Plot scatter plot of actual vs ideal similarities.
        
        Args:
            rsa_results: RSA results
            layer_type: Type of layer
            layer_ind: Layer index
        """
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Get upper triangle values
        upper_tri_indices = np.triu_indices_from(rsa_results.similarity_matrix, k=1)
        sim_values = rsa_results.similarity_matrix[upper_tri_indices]
        ideal_values = rsa_results.ideal_matrix[upper_tri_indices]
        
        # Add jitter to ideal values for better visualization
        jitter = np.random.normal(0, 0.02, size=ideal_values.shape)
        ideal_jittered = ideal_values + jitter
        
        # Color by same/different author
        colors = [PlotStyle.COLORS['primary'] if v == 1 else PlotStyle.COLORS['accent'] 
                 for v in ideal_values]
        
        ax.scatter(ideal_jittered, sim_values, c=colors, alpha=0.3, s=20)
        
        # Add box plots for each group
        same_author_sims = sim_values[ideal_values == 1]
        diff_author_sims = sim_values[ideal_values == 0]
        
        # Calculate means
        mean_same = np.mean(same_author_sims) if len(same_author_sims) > 0 else 0
        mean_diff = np.mean(diff_author_sims) if len(diff_author_sims) > 0 else 0
        
        ax.axhline(y=mean_same, xmin=0.75, xmax=1.0, color=PlotStyle.COLORS['primary'], 
                  linestyle='--', linewidth=2, label=f'Same Author Mean: {mean_same:.3f}')
        ax.axhline(y=mean_diff, xmin=0.0, xmax=0.25, color=PlotStyle.COLORS['accent'], 
                  linestyle='--', linewidth=2, label=f'Diff Author Mean: {mean_diff:.3f}')
        
        PlotStyle.style_axis(
            ax,
            title=f'RSA Correlation - {layer_type} Layer {layer_ind}\n'
                  f"Spearman's ρ = {rsa_results.spearman_rho:.4f}",
            xlabel='Ideal Similarity (0 = Different Author, 1 = Same Author)',
            ylabel='Cosine Similarity',
            grid_axis='both'
        )
        
        ax.set_xlim([-0.2, 1.2])
        ax.set_ylim([0, 1.05])
        ax.legend(loc='upper left', frameon=False)
        
        filename = f"rsa_correlation_scatter__{layer_type}__layer_{layer_ind}.png"
        self._save_plot(fig, filename)
    
    def plot_rsa_summary(
        self,
        rsa_all_results: Dict[str, Dict[str, RSAResults]],
        layer_types: List[str]
    ) -> None:
        """
        Plot summary of RSA results across all layers.
        
        Shows Spearman's rho for each layer type and index.
        
        Args:
            rsa_all_results: Dict[layer_type][layer_ind] -> RSAResults
            layer_types: List of layer types
        """
        fig, ax = plt.subplots(figsize=(14, 8))
        
        # Get gradient colors for layer types
        layer_type_colors = {
            'res': PlotStyle.COLORS['primary'],
            'mlp': PlotStyle.COLORS['secondary'],
            'att': PlotStyle.COLORS['accent']
        }
        
        for layer_type in layer_types:
            if layer_type not in rsa_all_results:
                continue
                
            layer_inds = sorted(rsa_all_results[layer_type].keys(), key=int)
            rhos = []
            
            for layer_ind in layer_inds:
                rsa_result = rsa_all_results[layer_type][layer_ind]
                rhos.append(rsa_result.spearman_rho)
            
            color = layer_type_colors.get(layer_type, PlotStyle.COLORS['primary'])
            ax.plot([int(li) for li in layer_inds], rhos, 
                   marker='o', markersize=8, linewidth=2,
                   color=color, label=f'{layer_type.upper()} layers')
        
        PlotStyle.style_axis(
            ax,
            title="RSA Summary: Spearman's ρ Across Layers",
            xlabel='Layer Index',
            ylabel="Spearman's ρ (correlation with ideal author matrix)",
            grid_axis='both'
        )
        
        ax.legend(loc='best', frameon=False)
        ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        
        filename = "rsa_summary_across_layers.png"
        self._save_plot(fig, filename)
    
    def plot_rsa_heatmap_by_layer(
        self,
        rsa_all_results: Dict[str, Dict[str, RSAResults]]
    ) -> None:
        """
        Create heatmap showing Spearman's rho for each layer type and index.
        
        Args:
            rsa_all_results: Dict[layer_type][layer_ind] -> RSAResults
        """
        # Collect all layer indices and types
        all_layer_inds = set()
        layer_types = list(rsa_all_results.keys())
        
        for layer_type in layer_types:
            all_layer_inds.update(rsa_all_results[layer_type].keys())
        
        layer_inds = sorted(all_layer_inds, key=int)
        
        # Create data matrix
        data = np.zeros((len(layer_types), len(layer_inds)))
        data[:] = np.nan
        
        for i, layer_type in enumerate(layer_types):
            for j, layer_ind in enumerate(layer_inds):
                if layer_ind in rsa_all_results[layer_type]:
                    data[i, j] = rsa_all_results[layer_type][layer_ind].spearman_rho
        
        # Create heatmap
        fig, ax = plt.subplots(figsize=(max(12, len(layer_inds) * 0.8), 6))
        
        cmap = PlotStyle.create_full_gradient_cmap()
        im = sns.heatmap(
            data, 
            annot=True, 
            fmt='.3f', 
            cmap=cmap,
            xticklabels=[f'L{li}' for li in layer_inds],
            yticklabels=[lt.upper() for lt in layer_types],
            ax=ax,
            vmin=-0.2,
            vmax=0.8,
            cbar_kws={'label': "Spearman's ρ"}
        )
        
        PlotStyle.style_axis(
            ax,
            title="RSA: Spearman's ρ by Layer Type and Index",
            xlabel='Layer Index',
            ylabel='Layer Type',
            grid_axis=''
        )
        
        filename = "rsa_heatmap_by_layer.png"
        self._save_plot(fig, filename)


# =============================================================================
# MAIN ANALYSIS
# =============================================================================

class LinearProbingAnalyzer:
    """Main analyzer for linear probing on classic activations."""
    
    def __init__(self, data_dir: Path, output_dir: Path, run_name: str = ""):
        """
        Initialize the analyzer.
        
        Args:
            data_dir: Directory containing activation files
            output_dir: Directory to save results
            run_name: Name prefix for output files
        """
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        self.run_name = run_name
        
        # Initialize components
        self.color_manager = AuthorColorManager()
        self.visualizer = LinearProbingVisualizer(self.output_dir, self.color_manager)
        self.classifiers = get_linear_probing_classifiers()
    
    def run_analysis(
        self,
        include_authors: List[str] = None,
        include_layer_types: List[str] = None,
        include_layer_inds: List[int] = None,
        include_setting: str = "baseline",
        aggregation: str = "mean",
        activation_type: str = "classic"
    ) -> Dict[str, Any]:
        """
        Run the full linear probing analysis.
        
        Args:
            include_authors: List of authors to analyze
            include_layer_types: List of layer types (res, mlp, att)
            include_layer_inds: List of layer indices
            include_setting: Setting filter (baseline/prompted)
            aggregation: Document aggregation method (mean/max/sum/last)
            activation_type: Type of activations ('classic', 'sae', or 'residual')
            
        Returns:
            Complete results dictionary
        """
        self.activation_type = activation_type
        logger.info(f"Activation type: {activation_type}")
        
        # Handle residual activations separately (different file structure)
        if activation_type == "residual":
            return self._run_residual_analysis(
                include_authors=include_authors,
                include_layer_types=include_layer_types,
                include_layer_inds=include_layer_inds
            )
        
        # Load filenames using appropriate loader
        if activation_type == "classic":
            filename_loader = ClassicActivationFilenamesLoader(
                data_dir=self.data_dir,
                include_authors=include_authors,
                include_layer_types=include_layer_types,
                include_layer_inds=include_layer_inds,
                include_setting=include_setting
            )
        elif activation_type == "sae":
            filename_loader = ActivationFilenamesLoader(
                data_dir=self.data_dir,
                include_authors=include_authors,
                include_layer_types=include_layer_types,
                include_layer_inds=include_layer_inds,
                include_prompted=include_setting
            )
        else:
            raise ValueError(f"Unknown activation type: {activation_type}")
        
        structured_filenames = filename_loader.get_structured_filenames()
        
        if not structured_filenames:
            logger.error("No activation files found!")
            return {}
        
        logger.info(f"Found layer types: {list(structured_filenames.keys())}")
        
        # Structure for all results
        # {classifier_name: {layer_type: {layer_ind: {author: metrics}}}}
        all_results = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
        
        # Structure for RSA results
        # {layer_type: {layer_ind: RSAResults}}
        rsa_all_results = defaultdict(dict)
        
        # Process each layer type and index
        for layer_type, layer_ind_dict in structured_filenames.items():
            logger.info(f"\n{'='*60}")
            logger.info(f"Processing layer type: {layer_type}")
            logger.info(f"{'='*60}")
            
            for layer_ind, author_filename_dict in layer_ind_dict.items():
                logger.info(f"\nLayer {layer_ind}: {len(author_filename_dict)} authors")
                
                # Load and aggregate activations for all authors
                author_to_activations = {}
                
                for author, filename_base in author_filename_dict.items():
                    try:
                        filepath = self.data_dir / filename_base
                        activations, metadata = load_activations(filepath, self.activation_type)
                        
                        # Aggregate to document level
                        aggregated = aggregate_activations_per_document(
                            activations, 
                            metadata.doc_lengths,
                            aggregation=aggregation,
                            metadata=metadata  # Pass metadata for sparse handling
                        )
                        
                        author_to_activations[author] = aggregated
                        logger.debug(f"  {author}: {aggregated.shape[0]} docs, {aggregated.shape[1]} features")
                        
                    except Exception as e:
                        logger.error(f"Error loading {author}: {e}")
                        import traceback
                        logger.debug(traceback.format_exc())
                        continue
                
                if len(author_to_activations) < 2:
                    logger.warning(f"Skipping layer {layer_ind} - insufficient authors")
                    continue
                
                # Prepare train/test splits
                X_train, X_test, y_train, y_test, author_list = prepare_train_test_data(
                    author_to_activations
                )
                
                logger.info(f"  Train: {X_train.shape}, Test: {X_test.shape}")
                
                # Run linear probing
                results = run_linear_probing_one_vs_rest(
                    X_train, X_test, y_train, y_test,
                    author_list, self.classifiers
                )
                
                # Store results
                for clf_name, clf_results in results.items():
                    all_results[clf_name][layer_type][layer_ind] = clf_results
                
                # Generate ROC plots (Aggregation 1)
                for clf_name in results.keys():
                    self.visualizer.plot_roc_by_authors(
                        results, clf_name, layer_type, layer_ind
                    )
                
                # Run RSA analysis
                logger.info(f"  Running RSA analysis...")
                
                # Combine all data for RSA (use all data, not just train)
                all_X = np.vstack([author_to_activations[a] for a in author_list])
                all_author_labels = []
                for author in author_list:
                    n_docs = author_to_activations[author].shape[0]
                    all_author_labels.extend([author] * n_docs)
                
                # Run RSA
                rsa_results = run_rsa_analysis(all_X, all_author_labels)
                rsa_all_results[layer_type][layer_ind] = rsa_results
                
                logger.info(f"    RSA Spearman's ρ = {rsa_results.spearman_rho:.4f} "
                           f"(p = {rsa_results.spearman_pvalue:.2e})")
                
                # Generate RSA visualizations
                self.visualizer.plot_rsa_similarity_matrix(
                    rsa_results, layer_type, layer_ind
                )
                self.visualizer.plot_rsa_correlation_scatter(
                    rsa_results, layer_type, layer_ind
                )
        
        # Get complete list of authors and classifiers
        all_authors = set()
        classifier_names = [c.name for c in self.classifiers]
        
        for clf_name in all_results:
            for layer_type in all_results[clf_name]:
                for layer_ind in all_results[clf_name][layer_type]:
                    all_authors.update(all_results[clf_name][layer_type][layer_ind].keys())
        
        all_authors = sorted(all_authors)
        
        # Generate additional visualizations
        logger.info("\nGenerating additional visualizations...")
        
        # Aggregation 2: ROC by layers
        for clf_name in classifier_names:
            for layer_type in all_results.get(clf_name, {}):
                for author in all_authors:
                    self.visualizer.plot_roc_by_layers(
                        all_results, clf_name, layer_type, author
                    )
        
        # Aggregation 3: ROC by classifiers
        for layer_type in structured_filenames.keys():
            layer_inds = list(structured_filenames[layer_type].keys())
            for layer_ind in layer_inds:
                for author in all_authors:
                    self.visualizer.plot_roc_by_classifiers(
                        all_results, layer_type, layer_ind, author, classifier_names
                    )
        
        # F1 bar plots
        for layer_type in structured_filenames.keys():
            self.visualizer.plot_f1_barplot(
                all_results, layer_type, classifier_names, all_authors
            )
        
        # Generate RSA summary visualizations
        logger.info("\nGenerating RSA summary visualizations...")
        if rsa_all_results:
            self.visualizer.plot_rsa_summary(
                rsa_all_results, list(structured_filenames.keys())
            )
            self.visualizer.plot_rsa_heatmap_by_layer(rsa_all_results)
        
        # Prepare final results structure
        final_results = {
            "run_name": self.run_name,
            "metadata": {
                "data_dir": str(self.data_dir),
                "output_dir": str(self.output_dir),
                "activation_type": self.activation_type,
                "classifiers": classifier_names,
                "layer_types": list(structured_filenames.keys()),
                "authors": all_authors,
                "aggregation": aggregation,
                "train_split_ratio": TRAIN_SPLIT_RATIO
            },
            "results": {},
            "rsa_results": {}
        }
        
        # Convert defaultdict to regular dict and remove non-serializable data
        for clf_name in all_results:
            final_results["results"][clf_name] = {}
            for layer_type in all_results[clf_name]:
                final_results["results"][clf_name][layer_type] = {}
                for layer_ind in all_results[clf_name][layer_type]:
                    final_results["results"][clf_name][layer_type][layer_ind] = {}
                    for author, metrics in all_results[clf_name][layer_type][layer_ind].items():
                        # Store metrics without ROC curve data (too large for JSON)
                        final_results["results"][clf_name][layer_type][layer_ind][author] = {
                            "accuracy": metrics.get("accuracy"),
                            "precision": metrics.get("precision"),
                            "recall": metrics.get("recall"),
                            "f1_score": metrics.get("f1_score"),
                            "roc_auc": metrics.get("roc_auc")
                        }
        
        # Add RSA results to final output
        for layer_type in rsa_all_results:
            final_results["rsa_results"][layer_type] = {}
            for layer_ind, rsa_result in rsa_all_results[layer_type].items():
                final_results["rsa_results"][layer_type][layer_ind] = rsa_result.to_dict()
        
        # Save results
        results_path = self.output_dir / "linear_probing_results.json"
        with open(results_path, 'w') as f:
            json.dump(final_results, f, indent=2)
        logger.info(f"\nResults saved to: {results_path}")
        
        return final_results

    def _run_residual_analysis(
        self,
        include_authors: List[str] = None,
        include_layer_types: List[str] = None,
        include_layer_inds: List[int] = None
    ) -> Dict[str, Any]:
        """
        Run linear probing analysis on residualized activations.
        
        Residual activations come from further_layer_style_search_methods.py and
        have a different structure: pre-split train/test data with author labels.
        
        Args:
            include_authors: List of authors to analyze (filter after loading)
            include_layer_types: List of layer types (res, mlp, att)
            include_layer_inds: List of layer indices
            
        Returns:
            Complete results dictionary
        """
        # Load filenames
        filename_loader = ResidualActivationFilenamesLoader(
            data_dir=self.data_dir,
            include_layer_types=include_layer_types,
            include_layer_inds=include_layer_inds
        )
        
        structured_filenames = filename_loader.get_structured_filenames()
        
        if not structured_filenames:
            logger.error("No residual activation files found!")
            logger.info(f"Looked in: {filename_loader.residuals_dir}")
            return {}
        
        logger.info(f"Found layer types: {list(structured_filenames.keys())}")
        
        # Structure for all results
        all_results = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
        rsa_all_results = defaultdict(dict)
        
        # Process each layer type and index
        for layer_type, layer_ind_dict in structured_filenames.items():
            logger.info(f"\n{'='*60}")
            logger.info(f"Processing layer type: {layer_type} (residuals)")
            logger.info(f"{'='*60}")
            
            for layer_ind, filename_base in layer_ind_dict.items():
                logger.info(f"\nLayer {layer_ind}")
                
                try:
                    # Load residual data
                    filepath = filename_loader.residuals_dir / filename_base
                    residual_data = load_residual_activations(filepath)
                    
                    # Get data
                    X_train = residual_data.residuals_train
                    X_test = residual_data.residuals_test
                    train_author_labels = residual_data.train_author_labels
                    test_author_labels = residual_data.test_author_labels
                    
                    # Get unique authors
                    all_unique_authors = sorted(set(train_author_labels.tolist()))
                    
                    # Filter authors if requested
                    if include_authors:
                        authors_to_include = set(include_authors)
                        train_mask = np.array([a in authors_to_include for a in train_author_labels])
                        test_mask = np.array([a in authors_to_include for a in test_author_labels])
                        
                        X_train = X_train[train_mask]
                        X_test = X_test[test_mask]
                        train_author_labels = train_author_labels[train_mask]
                        test_author_labels = test_author_labels[test_mask]
                        all_unique_authors = sorted(set(train_author_labels.tolist()))
                    
                    # Create author list and convert labels to indices
                    author_list = all_unique_authors
                    author_to_idx = {a: i for i, a in enumerate(author_list)}
                    
                    y_train = np.array([author_to_idx[a] for a in train_author_labels])
                    y_test = np.array([author_to_idx[a] for a in test_author_labels])
                    
                    logger.info(f"  Authors: {len(author_list)}")
                    logger.info(f"  Train: {X_train.shape}, Test: {X_test.shape}")
                    
                    if len(author_list) < 2:
                        logger.warning(f"Skipping layer {layer_ind} - insufficient authors")
                        continue
                    
                    # Run linear probing
                    results = run_linear_probing_one_vs_rest(
                        X_train, X_test, y_train, y_test,
                        author_list, self.classifiers
                    )
                    
                    # Store results
                    for clf_name, clf_results in results.items():
                        all_results[clf_name][layer_type][layer_ind] = clf_results
                    
                    # Generate ROC plots (Aggregation 1)
                    for clf_name in results.keys():
                        self.visualizer.plot_roc_by_authors(
                            results, clf_name, layer_type, layer_ind
                        )
                    
                    # Run RSA analysis
                    logger.info(f"  Running RSA analysis...")
                    
                    # Combine all data for RSA
                    all_X = np.vstack([X_train, X_test])
                    all_author_labels_combined = np.concatenate([train_author_labels, test_author_labels])
                    
                    rsa_results = run_rsa_analysis(all_X, all_author_labels_combined.tolist())
                    rsa_all_results[layer_type][layer_ind] = rsa_results
                    
                    logger.info(f"    RSA Spearman's ρ = {rsa_results.spearman_rho:.4f} "
                               f"(p = {rsa_results.spearman_pvalue:.2e})")
                    
                    # Generate RSA visualizations
                    self.visualizer.plot_rsa_similarity_matrix(
                        rsa_results, layer_type, layer_ind
                    )
                    self.visualizer.plot_rsa_correlation_scatter(
                        rsa_results, layer_type, layer_ind
                    )
                    
                except Exception as e:
                    logger.error(f"Error processing residuals for {layer_type} layer {layer_ind}: {e}")
                    import traceback
                    logger.debug(traceback.format_exc())
                    continue
        
        # Get complete list of authors and classifiers
        all_authors = set()
        classifier_names = [c.name for c in self.classifiers]
        
        for clf_name in all_results:
            for layer_type in all_results[clf_name]:
                for layer_ind in all_results[clf_name][layer_type]:
                    all_authors.update(all_results[clf_name][layer_type][layer_ind].keys())
        
        all_authors = sorted(all_authors)
        
        # Generate additional visualizations
        logger.info("\nGenerating additional visualizations...")
        
        # Aggregation 2: ROC by layers
        for clf_name in classifier_names:
            for layer_type in all_results.get(clf_name, {}):
                for author in all_authors:
                    self.visualizer.plot_roc_by_layers(
                        all_results, clf_name, layer_type, author
                    )
        
        # Aggregation 3: ROC by classifiers
        for layer_type in structured_filenames.keys():
            layer_inds = list(structured_filenames[layer_type].keys())
            for layer_ind in layer_inds:
                for author in all_authors:
                    self.visualizer.plot_roc_by_classifiers(
                        all_results, layer_type, layer_ind, author, classifier_names
                    )
        
        # F1 bar plots
        for layer_type in structured_filenames.keys():
            self.visualizer.plot_f1_barplot(
                all_results, layer_type, classifier_names, all_authors
            )
        
        # Generate RSA summary visualizations
        logger.info("\nGenerating RSA summary visualizations...")
        if rsa_all_results:
            self.visualizer.plot_rsa_summary(
                rsa_all_results, list(structured_filenames.keys())
            )
            self.visualizer.plot_rsa_heatmap_by_layer(rsa_all_results)
        
        # Prepare final results structure
        final_results = {
            "run_name": self.run_name,
            "metadata": {
                "data_dir": str(self.data_dir),
                "output_dir": str(self.output_dir),
                "activation_type": self.activation_type,
                "classifiers": classifier_names,
                "layer_types": list(structured_filenames.keys()),
                "authors": all_authors,
                "train_split_ratio": "pre-split (from residualization)",
                "note": "Residual activations from linear residualization analysis"
            },
            "results": {},
            "rsa_results": {}
        }
        
        # Convert defaultdict to regular dict
        for clf_name in all_results:
            final_results["results"][clf_name] = {}
            for layer_type in all_results[clf_name]:
                final_results["results"][clf_name][layer_type] = {}
                for layer_ind in all_results[clf_name][layer_type]:
                    final_results["results"][clf_name][layer_type][layer_ind] = {}
                    for author, metrics in all_results[clf_name][layer_type][layer_ind].items():
                        final_results["results"][clf_name][layer_type][layer_ind][author] = {
                            "accuracy": metrics.get("accuracy"),
                            "precision": metrics.get("precision"),
                            "recall": metrics.get("recall"),
                            "f1_score": metrics.get("f1_score"),
                            "roc_auc": metrics.get("roc_auc")
                        }
        
        # Add RSA results
        for layer_type in rsa_all_results:
            final_results["rsa_results"][layer_type] = {}
            for layer_ind, rsa_result in rsa_all_results[layer_type].items():
                final_results["rsa_results"][layer_type][layer_ind] = rsa_result.to_dict()
        
        # Save results
        results_path = self.output_dir / "linear_probing_results.json"
        with open(results_path, 'w') as f:
            json.dump(final_results, f, indent=2)
        logger.info(f"\nResults saved to: {results_path}")
        
        return final_results


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Linear Probing and RSA on Activations for Author Style Classification. "
                    "Supports classic (dense), SAE (sparse), and residual activations."
    )
    
    parser.add_argument(
        "--run_id",
        type=str,
        default=None,
        help="Activation run ID (takes precedence over --path_to_data)"
    )
    parser.add_argument(
        "--path_to_data",
        type=str,
        default=None,
        help="Directory containing activation files. For 'residual' type, should be the "
             "output directory from residualization analysis (containing 'residuals' subfolder)"
    )
    parser.add_argument(
        "--activation_type",
        type=str,
        default="classic",
        choices=["classic", "sae", "residual"],
        help="Type of activations: 'classic' for dense transformer activations, "
             "'sae' for sparse autoencoder features, "
             "'residual' for content-residualized activations from further_layer_style_search_methods.py"
    )
    parser.add_argument(
        "--run_name",
        type=str,
        default=None,
        help="Optional run name override for output directory"
    )
    parser.add_argument(
        "--include_authors",
        type=str,
        nargs="+",
        default=None,
        help="Authors to include in the analysis"
    )
    parser.add_argument(
        "--include_layer_types",
        type=str,
        nargs="+",
        default=None,
        choices=["res", "mlp", "att"],
        help="Layer types to include (res, mlp, att)"
    )
    parser.add_argument(
        "--include_layer_inds",
        type=int,
        nargs="+",
        default=None,
        help="Layer indices to include"
    )
    parser.add_argument(
        "--include_setting",
        type=str,
        default="baseline",
        choices=["prompted", "baseline"],
        help="Setting filter (prompted or baseline)"
    )
    parser.add_argument(
        "--aggregation",
        type=str,
        default="mean",
        choices=["mean", "max", "sum", "last"],
        help="Document aggregation method (note: 'last' not supported for sparse)"
    )
    
    args = parser.parse_args()
    
    if not args.run_id and not args.path_to_data:
        parser.error("Either --run_id or --path_to_data must be provided")
    
    return args


def main():
    """Main entry point."""
    args = parse_arguments()
    
    # Get data and output paths
    data_path, output_path, activation_run_info = get_data_and_output_paths(
        run_id=args.run_id,
        data_path=args.path_to_data,
        analysis_type="linear_probing",
        run_name_override=args.run_name
    )
    
    # Register analysis run
    analysis_tracker = AnalysisRunTracker()
    activation_run_id = activation_run_info.get('id') if activation_run_info else None
    if activation_run_id:
        analysis_id = analysis_tracker.register_analysis(
            activation_run_id=activation_run_id,
            analysis_type="linear_probing",
            data_path=str(data_path),
            output_path=str(output_path)
        )
        logger.info(f"Registered analysis run with ID: {analysis_id}")
    
    # Initialize and run analyzer
    run_name = args.run_name or (activation_run_info.get('run_name') if activation_run_info else "linear_probing")
    analyzer = LinearProbingAnalyzer(data_path, output_path, run_name)
    
    results = analyzer.run_analysis(
        include_authors=args.include_authors,
        include_layer_types=args.include_layer_types,
        include_layer_inds=args.include_layer_inds,
        include_setting=args.include_setting,
        aggregation=args.aggregation,
        activation_type=args.activation_type
    )
    
    # Print summary
    if results and "results" in results:
        logger.info("\n" + "="*60)
        logger.info("LINEAR PROBING SUMMARY")
        logger.info("="*60)
        
        for clf_name, clf_data in results["results"].items():
            logger.info(f"\n{clf_name}:")
            for layer_type, layer_data in clf_data.items():
                avg_f1 = []
                for layer_ind, layer_results in layer_data.items():
                    for author, metrics in layer_results.items():
                        if metrics.get("f1_score") is not None:
                            avg_f1.append(metrics["f1_score"])
                
                if avg_f1:
                    logger.info(f"  {layer_type}: avg F1 = {np.mean(avg_f1):.3f} (across {len(avg_f1)} author-layer combinations)")
    
    # Print RSA summary
    if results and "rsa_results" in results:
        logger.info("\n" + "="*60)
        logger.info("RSA SUMMARY (Spearman's ρ with Ideal Author Matrix)")
        logger.info("="*60)
        
        for layer_type, layer_data in results["rsa_results"].items():
            rhos = []
            for layer_ind, rsa_metrics in layer_data.items():
                rho = rsa_metrics.get("spearman_rho")
                if rho is not None:
                    rhos.append(rho)
                    logger.info(f"  {layer_type} Layer {layer_ind}: ρ = {rho:.4f}")
            
            if rhos:
                logger.info(f"  {layer_type} Average: ρ = {np.mean(rhos):.4f}")


if __name__ == "__main__":
    main()
