"""
Shared utilities for SAE analysis scripts.

This module contains common classes and methods used across multiple SAE analysis scripts
to avoid code duplication and ensure consistency.
"""

import os
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import warnings
import json
import logging
import pickle

# Set up logging
logger = logging.getLogger(__name__)
import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sp
import torch
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score, pairwise_distances
from sklearn.preprocessing import StandardScaler, LabelEncoder
from torchmetrics.clustering import DunnIndex

from abc import abstractmethod
from enum import Enum

class StorageFormat(Enum):
    """Enumeration for storage format options."""
    DENSE = "dense"
    SPARSE = "sparse"

@dataclass
class ActivationMetadata:
    """
    Metadata for activation storage with full traceability.
    
    Attributes:
        doc_ids: Array mapping each position to document index
        tok_ids: Array mapping each position to token index within document
        author_id: Author identifier
        doc_lengths: Actual length of each document (excluding padding)
        valid_mask: Boolean mask indicating valid (non-padding) positions
        original_shape: Original 3D shape (n_docs, max_seq_len, n_features)
        n_features: Number of SAE features
        storage_format: 'dense' or 'sparse'
        layer_type: Type of layer (res, mlp, att)
        layer_index: Layer number
        sae_id: SAE identifier
        model_name: Model name
    """
    doc_ids: np.ndarray
    tok_ids: np.ndarray
    author_id: str
    doc_lengths: np.ndarray
    valid_mask: np.ndarray
    original_shape: Tuple[int, int, int]
    n_features: int
    storage_format: str
    layer_type: str
    layer_index: int
    sae_id: str
    model_name: str


    @staticmethod
    def get_metadata_filename(activation_filename: str) -> str:
        """
        Get base filename without .npz or .sparse.npz extension.
        
        Args:
            filename: Filename with extension (e.g., 'file.sparse.npz' or 'file.npz')
            
        Returns:
            Base filename without extension (e.g., 'file')
        """
        if activation_filename.endswith('.sparse.npz'):
            return activation_filename[:-11] + '.meta.pkl'  # Remove .sparse.npz and add .meta.pkl
        elif activation_filename.endswith('.npz'):
            return activation_filename[:-4] + '.meta.pkl'   # Remove .npz and add .meta.pkl
        else:
            return activation_filename + '.meta.pkl'
        return activation_filename
    
    def save(self, activation_filename: str):
        """Save metadata to pickle file."""
        metadata_filename = ActivationMetadata.get_metadata_filename(activation_filename)
        with open(metadata_filename, 'wb') as f:
            pickle.dump(self, f)
    
    @staticmethod
    def load(activation_filename: str):
        """Load metadata from pickle file."""
        metadata_filename = ActivationMetadata.get_metadata_filename(activation_filename)
        with open(metadata_filename, 'rb') as f:
            return pickle.load(f)
    
    def get_position(self, doc_idx: int, tok_idx: int) -> Optional[int]:
        """
        Get the position in flattened array for a given doc_idx and tok_idx.
        Returns None if position is padding.
        """
        max_seq_len = self.original_shape[1]
        flat_pos = doc_idx * max_seq_len + tok_idx
        
        if flat_pos >= len(self.valid_mask):
            return None
        
        return flat_pos if self.valid_mask[flat_pos] else None


@dataclass
class ClusterMetrics:
    """Data class for clustering metrics"""
    silhouette_score: float
    silhouette_score_binary_jaccard: float
    dunn_index: float
    calinski_harabasz_score: float
    davies_bouldin_score: float

    def to_dict(self) -> Dict[str, float]:
        return {
            "silhouette_score": self.silhouette_score,
            "silhouette_score_binary_jaccard": self.silhouette_score_binary_jaccard,
            "dunn_index": self.dunn_index,
            "calinski_harabasz_score": self.calinski_harabasz_score,
            "davies_bouldin_score": self.davies_bouldin_score
        }


@dataclass
class FeatureImportanceMetrics:
    """Data class for feature importance metrics"""
    feature_indices: np.ndarray
    importance_scores: np.ndarray
    activation_frequencies: np.ndarray
    author_specific_scores: Dict[str, np.ndarray]
    cross_author_importance: np.ndarray
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "feature_indices": self.feature_indices,
            "importance_scores": self.importance_scores,
            "activation_frequencies": self.activation_frequencies,
            "author_specific_scores": self.author_specific_scores,
            "cross_author_importance": self.cross_author_importance
        }


class AuthorColorManager:
    """
    Manages consistent color mapping for authors across all visualizations.
    
    Uses the Nordic Ocean Extended Theme color palette for cohesive aesthetics
    across all visualizations in the project.
    """
    
    def __init__(self, color_palette: str = 'nordic_ocean'):
        """
        Initialize the color manager with a specific color palette.
        
        Args:
            color_palette: Name of color palette to use ('nordic_ocean' or matplotlib palette name)
        """
        self.color_palette = color_palette
        self.author_to_color = {}
        self.author_to_index = {}
        self._next_index = 0
        
        # Import PlotStyle for Nordic Ocean author colors
        from backend.src.utils.plot_styling import PlotStyle
        
        # Use Nordic Ocean Extended Theme - Author Colors
        # These colors are designed to be distinguishable while maintaining
        # aesthetic harmony with the Nordic Ocean gradient theme
        self._distinct_colors = PlotStyle.AUTHOR_COLORS.copy()
    
    def get_author_color(self, author: str) -> str:
        """
        Get the color for a specific author, creating a new mapping if needed.
        
        Args:
            author: Author name
            
        Returns:
            Color string (hex code)
        """
        if author not in self.author_to_color:
            # Use pre-defined colors first, then fall back to palette
            if self._next_index < len(self._distinct_colors):
                color = self._distinct_colors[self._next_index]
            else:
                # Generate color from palette for additional authors
                palette_colors = plt.cm.get_cmap(self.color_palette)
                color = palette_colors(self._next_index / max(1, self._next_index))
                if isinstance(color, tuple):
                    color = f'#{int(color[0]*255):02x}{int(color[1]*255):02x}{int(color[2]*255):02x}'
            
            self.author_to_color[author] = color
            self.author_to_index[author] = self._next_index
            self._next_index += 1
            
        return self.author_to_color[author]
    
    def get_author_colors(self, authors: List[str]) -> Dict[str, str]:
        """
        Get color mapping for a list of authors.
        
        Args:
            authors: List of author names
            
        Returns:
            Dictionary mapping author names to colors
        """
        return {author: self.get_author_color(author) for author in authors}
    
    def get_matplotlib_colors(self, authors: List[str]) -> List[str]:
        """
        Get list of colors in matplotlib format for a list of authors.
        
        Args:
            authors: List of author names
            
        Returns:
            List of color strings
        """
        return [self.get_author_color(author) for author in authors]
    
    def get_seaborn_palette(self, authors: List[str]) -> Dict[str, str]:
        """
        Get color palette in seaborn format for a list of authors.
        
        Args:
            authors: List of author names
            
        Returns:
            Dictionary mapping author names to colors (seaborn format)
        """
        return self.get_author_colors(authors)
    
    def get_altair_domain_range(self, authors: List[str]) -> Tuple[List[str], List[str]]:
        """
        Get domain and range for Altair color encoding.
        
        Args:
            authors: List of author names
            
        Returns:
            Tuple of (domain, range) for Altair color encoding
        """
        colors = self.get_matplotlib_colors(authors)
        return authors, colors
    
    def get_all_authors(self) -> List[str]:
        """
        Get list of all authors that have been assigned colors.
        
        Returns:
            List of author names
        """
        return list(self.author_to_color.keys())
    
    def reset(self):
        """Reset the color manager to start fresh."""
        self.author_to_color = {}
        self.author_to_index = {}
        self._next_index = 0
    
    def save_color_mapping(self, filepath: Path):
        """Save the current color mapping to a JSON file."""
        mapping_data = {
            'author_to_color': self.author_to_color,
            'author_to_index': self.author_to_index,
            'next_index': self._next_index
        }
        with open(filepath, 'w') as f:
            json.dump(mapping_data, f, indent=2)
    
    def load_color_mapping(self, filepath: Path):
        """Load color mapping from a JSON file."""
        with open(filepath, 'r') as f:
            mapping_data = json.load(f)
        
        self.author_to_color = mapping_data['author_to_color']
        self.author_to_index = mapping_data['author_to_index']
        self._next_index = mapping_data['next_index']
    
    def create_color_legend(self, save_path: Path, title: str = "Author Color Legend"):
        """Create and save a color legend showing all author-color mappings."""
        authors = sorted(self.get_all_authors())
        if not authors:
            return
        
        fig, ax = plt.subplots(figsize=(8, max(4, len(authors) * 0.3)))
        
        for i, author in enumerate(authors):
            color = self.get_author_color(author)
            ax.barh(i, 1, color=color, alpha=0.7)
            ax.text(0.5, i, f" {author}", va='center', ha='left', fontsize=12, fontweight='bold')
        
        ax.set_xlim(0, 1)
        ax.set_ylim(-0.5, len(authors) - 0.5)
        ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
        ax.set_xlabel("Color Assignment", fontsize=12)
        ax.set_ylabel("Authors", fontsize=12)
        
        # Remove x-axis ticks and labels
        ax.set_xticks([])
        ax.set_xticklabels([])
        
        # Remove y-axis ticks but keep labels
        ax.set_yticks([])
        ax.set_yticklabels([])
        
        # Remove spines
        for spine in ax.spines.values():
            spine.set_visible(False)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Color legend saved to: {save_path}")


class FilenamesLoader:
    """Handles parsing of structured filenames"""

    def __init__(self, data_dir: Path):
        self.ACTIVATION_TYPES = ["res", "mlp", "att"]
        self.data_dir = data_dir
        self.filenames = self.load_filenames()

    @abstractmethod
    def load_filenames(self) -> List[str]:
        """
        Load and filter filenames based on specified criteria.
        """
        pass

    @abstractmethod
    def parse_filename(self, filename: str) -> Dict[str, str]:
        """
        Parse filename into components.
        """
        pass

    @abstractmethod
    def get_structured_filenames(self):
        """
        Parse filenames and organize in hierarchy
        """
        pass

    def validate_activation_type(self, activation_type: str):
        """Validate activation type"""
        if activation_type not in self.ACTIVATION_TYPES:
            raise ValueError(f"Invalid activation type: {activation_type}. Must be one of {self.ACTIVATION_TYPES}")

    @staticmethod
    def format_layer_string(layer_ind: str) -> str:
        """Format layer index as zero-padded string"""
        return f"0{layer_ind}" if len(layer_ind) == 1 else layer_ind



class ActivationFilenamesLoader(FilenamesLoader):
    """Handles loading and filtering of activation filenames"""

    def __init__(self, data_dir: Path, include_authors: List[str] = None, include_layer_types: List[str] = None, include_layer_inds: List[int] = None, include_prompted: str = "baseline"):
        # Set up parent class attributes manually to avoid calling load_filenames() before setting filters
        self.ACTIVATION_TYPES = ["res", "mlp", "att"]
        self.data_dir = data_dir
        self.include_authors = include_authors
        self.include_layer_types = include_layer_types
        self.include_layer_inds = include_layer_inds
        self.include_prompted = include_prompted
        self.filenames = self.load_filenames()

    @staticmethod
    def parse_filename(filename: str) -> Dict[str, str]:
        """
        Parse structured filename into components.
        
        Expected format: sae_{prompted}__{model}__{activation_type}__activations__{author}__layer_{layeri}.npz
        Example: sae_prompted__google_gemma-2-2b__res__activations__author1__layer_5.npz
        """
        if not (filename.endswith(".npz") or filename.endswith(".sparse.npz")):
            raise ValueError(f"Expected .npz or .sparse.npz file, got: {filename}")

        # Remove extension (.npz or .sparse.npz)
        if filename.endswith(".sparse.npz"):
            name_without_ext = filename[:-11]  # Remove .sparse.npz
        else:
            name_without_ext = filename[:-4]   # Remove .npz
        
        # Split by double underscores to get main components
        parts = name_without_ext.split("__")
        
        if len(parts) < 6:
            raise ValueError(f"Invalid filename format. Expected at least 6 parts separated by '__', got {len(parts)}: {filename}")
        
        # Extract components
        model = parts[1]  # google_gemma-2-2b
        activation_type = parts[2]  # res, mlp, or att
        author = parts[4]  # author name
        prompted = "prompted" if "prompted" in parts[0] else "baseline" # prompted

        # Extract layer from the last part (layer_5 -> 5)
        layer_part = parts[5]  # layer_5
        if not layer_part.startswith("layer_"):
            raise ValueError(f"Invalid layer format. Expected 'layer_X', got: {layer_part}")
        layer_ind = layer_part[6:]  # Remove "layer_" prefix
        
        return {
            "model": model,
            "activation_type": activation_type,
            "author": author,
            "layer_ind": layer_ind,
            "prompted": prompted
        }

    def load_filenames(self) -> List[str]:
        """
        Load and filter filenames based on specified criteria.
            
        Returns:
            Filtered list of filenames
        """
        filenames = [f for f in os.listdir(self.data_dir) if f.endswith(".npz") or f.endswith(".sparse.npz")]
        filtered_filenames = []
        
        for filename in filenames:
            try:
                parsed = self.parse_filename(filename)
                
                # Check author filter
                if self.include_authors and parsed['author'] not in self.include_authors:
                    continue
                
                # Check layer type filter
                if self.include_layer_types and parsed['activation_type'] not in self.include_layer_types:
                    continue
                
                # Check layer index filter
                if self.include_layer_inds and int(parsed['layer_ind']) not in self.include_layer_inds:
                    continue

                if self.include_prompted and self.include_prompted != parsed['prompted']:
                    continue
                
                
                filtered_filenames.append(filename)
                
            except (ValueError, KeyError, IndexError) as e:
                logger.warning(f"Could not parse filename '{filename}' for filtering: {e}")
                continue
        
        return filtered_filenames

    def get_structured_filenames(self) -> Dict[str, Dict[str, Dict[str, str]]]:
        """
        Parse filenames and organize in hierarchy: activation_type -> layer_ind -> author
        
        Args:
            filenames: List of filenames to parse
            
        Returns:
            Dict of (structured_filenames) where:
                - structured_filenames: Dict[activation_type, Dict[layer_ind, Dict[author, filename]]]
        """
        filenames_structured = defaultdict(lambda: defaultdict(lambda: defaultdict(str)))
        
        
        for filename in self.filenames:
            try:
                filename_parsed = self.parse_filename(filename)
                activation_type = filename_parsed["activation_type"]
                layer_ind = filename_parsed["layer_ind"]
                author = filename_parsed["author"]

                if filename.endswith(".sparse.npz"):
                    name_without_ext = filename[:-11]  # Remove .sparse.npz
                else:
                    name_without_ext = filename[:-4]   # Remove .npz
                
                filenames_structured[activation_type][layer_ind][author] = name_without_ext
            except (ValueError, KeyError) as e:
                logger.warning(f"Could not parse filename '{filename}': {e}")
                continue
                
        return filenames_structured

    def create_filename(self, model: str, activation_type: str, author: str, layer_ind: int) -> str:
        """
        Create a filename in the standard format.
        
        Args:
            model: Model name (e.g., "google/gemma-2-2b")
            activation_type: Type of activation (res, mlp, att)
            author: Author name
            layer_ind: Layer index
            
        Returns:
            Formatted filename string
        """
        # Replace slashes with underscores in model name
        model_clean = model.replace('/', '_')
        
        # Validate activation type
        self.validate_activation_type(activation_type)
        
        return f"sae__{model_clean}__{activation_type}__activations__{author}__layer_{layer_ind}.npz"

class EntropyFilenamesLoader(FilenamesLoader):
    """Handles loading and filtering of activation filenames"""

    def __init__(self, data_dir: Path, include_authors: List[str] = None):
        self.data_dir = data_dir
        self.include_authors = include_authors
        self.filenames = self.load_filenames()

    @staticmethod
    def parse_filename(filename: str) -> Dict[str, str]:
        """
        Parse structured filename into components.
        
        Expected format: sae_{setting}__{model}__entropy__{author}.npy or sae_{setting}__{model}__cross_entropy_loss__{author}.npy
        Example: sae_baseline__google_gemma-2-9b__entropy__bush.npy or sae_baseline__google_gemma-2-9b__cross_entropy_loss__bush.npy
        """
        if not filename.endswith(".npy") or "entropy" not in filename:
            raise ValueError(f"Expected .npy file and entropy in name, got: {filename}")

        # Remove .npy extension
        name_without_ext = filename[:-4]
        
        # Split by double underscores to get main components
        parts = name_without_ext.split("__")
        
        if len(parts) < 4:  # 4 parts for sae_{setting}__{model}__entropy__{author}.npy
            raise ValueError(f"Invalid filename format. Expected at least 4 parts separated by '__', got {len(parts)}: {filename}")
        
        # Extract components
        prompted = "prompted" if "prompted" in parts[0] else "baseline"
        model = parts[1]  # google_gemma-2-9b
        
        # Handle both entropy types
        if "cross_entropy_loss" in filename:
            entropy_type = "cross_entropy_loss"
        elif "entropy" in filename:
            entropy_type = "entropy"
        else:
            entropy_type = parts[2]  # fallback
            
        author = parts[-1]  # author name (last part)

        
        return {
            "model": model,
            "entropy_type": entropy_type,
            "author": author,
            "prompted": prompted
        }

    def load_filenames(self) -> List[str]:
        """
        Load filenames for entropy and cross_entropy in one list.
            
        Returns:
            List of filenames
        """
        filenames = [f for f in os.listdir(self.data_dir) if f.endswith(".npy") and "entropy" in f]
        filtered_filenames = []
        for filename in filenames:
            filename_parsed = self.parse_filename(filename)
            author = filename_parsed["author"]
            if self.include_authors and author not in self.include_authors:
                continue
            filtered_filenames.append(filename)
        return filtered_filenames

    def get_structured_filenames(self) -> Dict[str, Dict[str, Dict[str, str]]]:
        """
        Parse filenames and organize in hierarchy: entropy_type -> author
        
        Args:
            filenames: List of filenames to parse
            
        Returns:
            Dict of (structured_filenames) where:
                - structured_filenames: Dict[entropy_type, Dict[author, Dict[str, str]]]
        """
        filenames_structured = defaultdict(lambda: defaultdict(lambda: defaultdict(str)))
        
        for filename in self.filenames:
            try:
                filename_parsed = self.parse_filename(filename)
                entropy_type = filename_parsed["entropy_type"]
                author = filename_parsed["author"]
                prompted = filename_parsed["prompted"]
                filenames_structured[entropy_type][author][prompted] = filename 
            except (ValueError, KeyError) as e:
                logger.warning(f"Could not parse filename '{filename}': {e}")
                continue
                
        return filenames_structured


class TokenandFullTextFilenamesLoader(FilenamesLoader):
    """Handles loading and filtering of token and full text filenames"""

    def __init__(self, data_dir: Path, include_authors: List[str] = None):
        self.data_dir = data_dir
        self.include_authors = include_authors
        self.filenames = self.load_filenames()
        
    def load_filenames(self) -> List[str]:
        """
        Load filenames for tokens and full texts.
        """
        filenames = [f for f in os.listdir(self.data_dir) if f.endswith(".json") and ("tokens" in f or "full_texts" in f)]
        filtered_filenames = []
        for filename in filenames:
            filename_parsed = self.parse_filename(filename)
            author = filename_parsed["author"]
            if self.include_authors and author not in self.include_authors:
                continue
            filtered_filenames.append(filename)
        return filtered_filenames
        
    def get_structured_filenames(self) -> Dict[str, Dict[str, Dict[str, str]]]:
        """
        Parse filenames and organize in hierarchy: information_type -> author
        """
        filenames_structured = defaultdict(lambda: defaultdict(lambda: defaultdict(str)))
        for filename in self.filenames:
            filename_parsed = self.parse_filename(filename)
            information_type = filename_parsed["information_type"]
            author = filename_parsed["author"]
            prompted = filename_parsed["prompted"]
            filenames_structured[information_type][author][prompted] = filename
        return filenames_structured

    def parse_filename(self, filename: str) -> Dict[str, str]:
        """
        Parse filename into components.

        Expected format: sae_{setting}__{model}__tokens__{author}.json or sae_{setting}__{model}__full_texts__{author}.json
        Example: sae_baseline__google_gemma-2-9b__tokens__bush.json or sae_baseline__google_gemma-2-9b__full_texts__bush.json
        """
        if not filename.endswith(".json") or all(info_type not in filename for info_type in ["tokens", "full_texts"]):
            raise ValueError(f"Expected .json file and tokens or full_texts in name, got: {filename}")

        # Remove .json extension
        name_without_ext = filename[:-5]
        
        # Split by double underscores to get main components
        parts = name_without_ext.split("__")
        
        if len(parts) < 4:  # 4 parts for sae_{setting}__{model}__tokens__{author}.json
            raise ValueError(f"Invalid filename format. Expected at least 4 parts separated by '__', got {len(parts)}: {filename}")
        
        # Extract components
        prompted = "prompted" if "prompted" in parts[0] else "baseline"
        model = parts[1]  # google_gemma-2-9b
        information_type = parts[2]  # tokens or full_texts
        author = parts[3]  # author name

        return {"model": model, "information_type": information_type, "author": author, "prompted": prompted}


class DataLoader:
    """Handles loading and validation of SAE data files"""
        

    @staticmethod
    def validate_activations(activations: np.ndarray) -> bool:
        """Check if activations data is valid (no empty rows)"""
        return np.all(np.any(np.any(activations, axis=2), axis=1))

    def load_sae_activations(self, filepath: Path) -> Tuple[np.ndarray, ActivationMetadata]:
        """
        Load activations from either dense or sparse format.
        
        Returns:
            activations: 3D numpy array (n_docs, max_seq_len, n_features)
            metadata: ActivationMetadata object
        """
        # Pass the activation filename to load the corresponding metadata
        metadata = ActivationMetadata.load(str(filepath))
        
        if metadata.storage_format == 'dense':
            data_path = Path(filepath).with_suffix('.npz')
            data = np.load(data_path)
            activations = data['activations']
        else:  # sparse
            data_path = Path(filepath).with_suffix('.npz')
            sparse_matrix = sp.load_npz(data_path)
            activations = sparse_matrix
            
            # Convert back to 3D dense array
            """flat_acts = sparse_matrix.toarray()
            activations = flat_acts.reshape(metadata.original_shape)"""
        
        return activations, metadata


class ActivationProcessor:
    """Handles processing and aggregation of activation data"""

    @staticmethod
    def aggregate_normalized(activations: np.ndarray, tokens_per_doc: np.ndarray, cutoff: int = 0) -> np.ndarray:
        """Aggregate activations by summing and normalizing by document length"""
        if np.any(tokens_per_doc <= 0):
            raise ValueError("Tokens per doc must be positive.")

        # Sum along sequence length axis, starting from cutoff
        summed = np.sum(activations[:, cutoff:, :], axis=1)

        # Normalize by effective token count
        tokens_per_doc_cutoff = np.maximum(tokens_per_doc - cutoff, 1)  # Avoid division by zero
        aggregated = summed / tokens_per_doc_cutoff[:, np.newaxis]
        
        logger.debug(f"Shape of aggregated normalized: {aggregated.shape}")

        return aggregated

    @staticmethod
    def compute_activation_stats(activations: np.ndarray, tokens_per_doc: np.ndarray) -> np.ndarray:
        """Compute average non-zero activations per document"""
        non_zeroes_per_token = np.count_nonzero(activations, axis=2)
        return np.sum(non_zeroes_per_token, axis=1) / tokens_per_doc

    @staticmethod
    def compute_feature_activations_vectorized(activations: np.ndarray) -> np.ndarray:
        """Compute feature activations using vectorized operations"""
        return np.count_nonzero(activations, axis=(0, 1))

    @staticmethod
    def aggregate_data_over_docs_and_tokens(data: np.ndarray, start_token_ind: int) -> np.ndarray:
        """
        Aggregate activation data over documents and tokens starting from a specific token index.
        
        Args:
            data: Activation data of shape (n_docs, n_tokens, n_features)
            start_token_ind: Starting token index for aggregation
            
        Returns:
            Aggregated data of shape (n_features)
        """
        # Filter from start_token_ind to end
        data_filtered = data[:, start_token_ind:, :].copy()

        # Aggregate over docs and tokens
        data_filtered = np.sum(data_filtered, axis=(0, 1))

        return data_filtered

    @staticmethod
    def get_active_tokens_and_docs(activations: np.ndarray, start_token_ind: int) -> Dict[int, List[Tuple[int, int]]]:
        """
        Find active tokens and their document positions for each feature.
        
        Args:
            activations: Activation data of shape (n_docs, n_tokens, n_features)
            start_token_ind: Starting token index to consider
            
        Returns:
            Dictionary mapping feature indices to lists of (doc_ind, token_ind) tuples
        """
        # Find all indices where values > 0
        doc_indices, token_indices, feature_indices = np.where(activations > 0)
        
        # Group by feature index using defaultdict for efficiency
        result = defaultdict(list)
        for doc_idx, token_idx, feature_idx in zip(doc_indices, token_indices, feature_indices):
            if token_idx >= start_token_ind:
                result[int(feature_idx)].append((int(doc_idx), int(token_idx)))
        return result


class MetricsCalculator:
    """Calculates various clustering and separation metrics"""

    def __init__(self):
        self.scaler = StandardScaler()

    def compute_cluster_metrics(self, features: np.ndarray, labels: List[str]) -> ClusterMetrics:
        """Compute all clustering metrics for given features and labels"""

        # Standard silhouette score
        silhouette = silhouette_score(features, labels)

        # Silhouette on binary Jaccard distance
        features_binary = (features > 0).astype(np.int8)
        jaccard_dists = pairwise_distances(features_binary, metric="jaccard", n_jobs=-1)
        silhouette_jaccard = silhouette_score(jaccard_dists, labels, metric="precomputed")

        # Dunn Index using PyTorch
        data_tensor = torch.tensor(features, dtype=torch.float32)
        le = LabelEncoder()
        targets_tensor = torch.as_tensor(le.fit_transform(labels), dtype=torch.long)
        dunn_index = DunnIndex(p=2)(data_tensor, targets_tensor).cpu().numpy()

        # Calinski-Harabasz and Davies-Bouldin scores
        cal_har_score = calinski_harabasz_score(features, labels)
        davies_bouldin = davies_bouldin_score(features, labels)

        return ClusterMetrics(
            silhouette_score=silhouette,
            silhouette_score_binary_jaccard=silhouette_jaccard,
            dunn_index=dunn_index,
            calinski_harabasz_score=cal_har_score,
            davies_bouldin_score=davies_bouldin
        )


class TokenLoader:
    """Handles loading and tokenization of text data"""
    
    @staticmethod
    def load_tokens(data_dir: Path) -> Tuple[Dict[str, List[List[str]]], Dict[str, List[str]]]:
        """
        Load tokens and full documents  for specified authors.
        
        Args:
            data_dir: Directory containing tokens.json and full_texts.json
            
        Returns:
            Tuple of (tokens, full_texts) where:
                - tokens: Dictionary mapping author names to tokenized documents
                - full_texts: Dictionary mapping author names to original text documents
        """
        with open(os.path.join(data_dir, "tokens.json"), "r", encoding="utf-8") as f:
            tokens = json.load(f)
        with open(os.path.join(data_dir, "full_texts.json"), "r", encoding="utf-8") as f:
            full_texts = json.load(f)

        return tokens, full_texts


class BaseAnalyzer:
    """Base class for SAE analyzers with common functionality"""
    
    def __init__(self, data_dir: Path, output_dir: Path, run_prefix: str = ""):
        """
        Initialize the base analyzer.
        
        Args:
            data_dir: Directory containing SAE activation files
            output_dir: Directory to save results
            run_prefix: Prefix for the output directory
        """
        self.data_dir = Path(data_dir)
        self.run_prefix = run_prefix
        self.output_dir = Path(output_dir) / Path(self.run_prefix)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
        # Initialize shared components
        self.color_manager = AuthorColorManager()
        self.data_loader = DataLoader()
        self.activation_processor = ActivationProcessor()
        self.metrics_calculator = MetricsCalculator()
        self.token_loader = TokenLoader()
        self.filename_parser = ActivationFilenamesLoader(self.data_dir)
        
        # Discover authors and set up color mapping
        self._load_or_discover_authors()
    
    def _discover_authors(self):
        """Discover all authors from data files and pre-populate the color manager"""
        if not self.data_dir.exists():
            return
            
        authors = set()
        for filename in os.listdir(self.data_dir):
            if filename.endswith('.npz'):
                try:
                    parsed = self.filename_parser.parse_filename(filename)
                    authors.add(parsed['author'])
                except (ValueError, KeyError, IndexError) as e:
                    # Skip files that don't match expected naming pattern
                    logger.warning(f"Could not parse filename '{filename}': {e}")
                    continue
        
        # Pre-populate color manager with all discovered authors
        for author in sorted(authors):
            self.color_manager.get_author_color(author)
        
        logger.info(f"Discovered {len(authors)} authors: {sorted(authors)}")
        logger.info("Author color assignments:")
        for author in sorted(authors):
            color = self.color_manager.get_author_color(author)
            logger.info(f"  {author}: {color}")
        
        # Save color mapping for future reference
        color_mapping_file = self.output_dir / "author_color_mapping.json"
        self.color_manager.save_color_mapping(color_mapping_file)
        logger.info(f"Color mapping saved to: {color_mapping_file}")
        
        # Create color legend
        legend_file = self.output_dir / "author_color_legend.png"
        self.color_manager.create_color_legend(legend_file)
    
    def _load_or_discover_authors(self):
        """Try to load existing color mapping, otherwise discover authors from data files"""
        color_mapping_file = self.output_dir / "author_color_mapping.json"
        
        if color_mapping_file.exists():
            try:
                self.color_manager.load_color_mapping(color_mapping_file)
                logger.info(f"Loaded existing color mapping from: {color_mapping_file}")
                logger.info("Current author color assignments:")
                for author in sorted(self.color_manager.get_all_authors()):
                    color = self.color_manager.get_author_color(author)
                    logger.info(f"  {author}: {color}")
                
                # Create color legend
                legend_file = self.output_dir / "author_color_legend.png"
                self.color_manager.create_color_legend(legend_file)
                return
            except Exception as e:
                logger.error(f"Failed to load color mapping: {e}")
                logger.info("Falling back to discovering authors from data files...")
        
        # Fall back to discovering authors from data files
        self._discover_authors()
    
    def filter_filenames(self, filenames: List[str], include_authors: List[str] = None, 
                        include_layer_types: List[str] = None, include_layer_inds: List[int] = None) -> List[str]:
        """
        Filter filenames based on specified criteria.
        
        Args:
            filenames: List of filenames to filter
            include_authors: List of authors to include
            include_layer_types: List of layer types to include (res, mlp, att)
            include_layer_inds: List of layer indices to include
            
        Returns:
            Filtered list of filenames
        """
        filtered_filenames = []
        
        for filename in filenames:
            try:
                parsed = self.filename_parser.parse_filename(filename)
                
                # Check author filter
                if include_authors and parsed['author'] not in include_authors:
                    continue
                
                # Check layer type filter
                if include_layer_types and parsed['activation_type'] not in include_layer_types:
                    continue
                
                # Check layer index filter
                if include_layer_inds and int(parsed['layer_ind']) not in include_layer_inds:
                    continue
                
                filtered_filenames.append(filename)
                
            except (ValueError, KeyError, IndexError) as e:
                logger.warning(f"Could not parse filename '{filename}' for filtering: {e}")
                continue
        
        return filtered_filenames
