"""
Data loading utilities for feature selection analysis.

This module handles loading and processing of activation data for both:
- Token-level analysis (sparse matrices)
- Document-level aggregated analysis (dense matrices)
"""

import numpy as np
import scipy.sparse as sp
from pathlib import Path
from typing import Dict, Tuple, Optional
import logging

from backend.src.utils.shared_utilities import DataLoader
from backend.src.analysis.feature_selection_base import FeaturesData

logger = logging.getLogger(__name__)


def retrieve_and_combine_author_data_token_level(
    author_filename_dict: Dict[str, str],
    path_to_data: str,
    binary: bool = True,
    from_token: int = 0
) -> Tuple[sp.csr_matrix, sp.csr_matrix, np.ndarray, np.ndarray, Dict[int, str], Dict, Dict]:
    """
    Retrieve and combine author data for token-level analysis using sparse matrices.
    
    Args:
        author_filename_dict: Dictionary mapping author names to filenames
        path_to_data: Path to data directory
        binary: Whether to apply binary threshold (>1)
        from_token: Starting token position
    
    Returns:
        Tuple of (train_activations, test_activations, train_labels, test_labels, 
                 int_to_author, train_metadata, test_metadata)
    """
    train_activations_list = []
    test_activations_list = []
    
    train_labels = []
    test_labels = []
    
    # Track metadata for proper document/token position mapping
    train_metadata = {'doc_ids': [], 'tok_ids': [], 'valid_mask': [], 'author_ids': []}
    test_metadata = {'doc_ids': [], 'tok_ids': [], 'valid_mask': [], 'author_ids': []}
    
    int_to_author = {}
    n_features = None
    
    for author_ind, (author, filename) in enumerate(author_filename_dict.items()):
        int_to_author[author_ind] = author
        logger.info(f"Loading data for author {author_ind} {author} from {filename}")
        
        # Load sparse activation data
        data, metadata = DataLoader().load_sae_activations(Path(path_to_data) / filename)
        
        if n_features is None:
            n_features = data.shape[2] if len(data.shape) == 3 else data.shape[1]
        
        # Get document lengths from metadata
        if not hasattr(metadata, 'doc_lengths'):
            raise ValueError(f"Doc lengths not found in metadata for {filename}. Possibly old files are used")
        
        doc_lengths = metadata.doc_lengths
        n_docs = len(doc_lengths)
        n_docs_train = int(n_docs * 0.8)
        
        logger.info(f"Author {author}: {n_docs} docs, {n_docs_train} for training")
        
        # Process each document
        for doc_idx in range(n_docs):
            doc_length = doc_lengths[doc_idx]
            
            if from_token >= doc_length:
                continue
            
            # Get valid tokens for this document (excluding padding)
            if sp.issparse(data):
                # For sparse data, extract document tokens
                start_idx = doc_idx * metadata.original_shape[1]
                end_idx = start_idx + metadata.original_shape[1]
                doc_data = data[start_idx:end_idx]
                
                # Filter to valid tokens only
                doc_valid_mask = metadata.valid_mask[start_idx:end_idx]
                valid_token_indices = np.where(doc_valid_mask)[0]
                
                # Apply from_token filter
                valid_token_indices = valid_token_indices[valid_token_indices >= from_token]
                
                if len(valid_token_indices) == 0:
                    continue
                
                # Extract valid tokens
                doc_tokens = doc_data[valid_token_indices]
                
            else:
                # For dense data
                doc_tokens = data[doc_idx, from_token:doc_length, :]
                valid_token_indices = np.arange(from_token, doc_length)
            
            n_valid_tokens = len(valid_token_indices)
            
            if doc_idx < n_docs_train:
                # Training data
                train_activations_list.append(doc_tokens)
                train_labels.extend([author_ind] * n_valid_tokens)
                
                # Update metadata
                train_metadata['doc_ids'].extend([doc_idx] * n_valid_tokens)
                train_metadata['tok_ids'].extend(valid_token_indices.tolist())
                train_metadata['valid_mask'].extend([True] * n_valid_tokens)
                train_metadata['author_ids'].extend([author_ind] * n_valid_tokens)
            else:
                # Test data
                test_activations_list.append(doc_tokens)
                test_labels.extend([author_ind] * n_valid_tokens)
                
                # Update metadata
                test_metadata['doc_ids'].extend([doc_idx] * n_valid_tokens)
                test_metadata['tok_ids'].extend(valid_token_indices.tolist())
                test_metadata['valid_mask'].extend([True] * n_valid_tokens)
                test_metadata['author_ids'].extend([author_ind] * n_valid_tokens)
    
    # Combine all activations into sparse matrices
    if train_activations_list:
        if sp.issparse(train_activations_list[0]):
            train_activations = sp.vstack(train_activations_list)
        else:
            train_activations = sp.csr_matrix(np.vstack(train_activations_list))
    else:
        train_activations = sp.csr_matrix((0, n_features))
    
    if test_activations_list:
        if sp.issparse(test_activations_list[0]):
            test_activations = sp.vstack(test_activations_list)
        else:
            test_activations = sp.csr_matrix(np.vstack(test_activations_list))
    else:
        test_activations = sp.csr_matrix((0, n_features))
    
    # Convert lists to arrays
    train_labels = np.array(train_labels)
    test_labels = np.array(test_labels)
    
    # Convert metadata lists to arrays
    for key in train_metadata:
        train_metadata[key] = np.array(train_metadata[key])
    for key in test_metadata:
        test_metadata[key] = np.array(test_metadata[key])
    
    logger.info(f"Shape of train data (sparse): {train_activations.shape}")
    logger.info(f"Shape of test data (sparse): {test_activations.shape}")
    logger.info(f"Train data sparsity: {1 - train_activations.nnz / (train_activations.shape[0] * train_activations.shape[1]):.4f}")
    logger.info(f"Test data sparsity: {1 - test_activations.nnz / (test_activations.shape[0] * test_activations.shape[1]):.4f}")
    
    if binary:
        # Apply binary threshold to sparse matrices
        train_activations.data = (train_activations.data > 1).astype(np.int8)
        test_activations.data = (test_activations.data > 1).astype(np.int8)
        train_activations.eliminate_zeros()  # Remove zeros created by thresholding
        test_activations.eliminate_zeros()
    
    return train_activations, test_activations, train_labels, test_labels, int_to_author, train_metadata, test_metadata


def retrieve_and_combine_author_data_aggregated(
    author_filename_dict: Dict[str, str],
    path_to_data: str,
    from_token: int = 0
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[int, str], np.ndarray, np.ndarray]:
    """
    Retrieve and combine author data for document-level aggregated analysis using dense matrices.
    
    Args:
        author_filename_dict: Dictionary mapping author names to filenames
        path_to_data: Path to data directory
        from_token: Starting token position
    
    Returns:
        Tuple of (train_activations, test_activations, train_labels, test_labels, 
                 int_to_author, train_doc_ids, test_doc_ids)
    """
    train_activations_list = []
    test_activations_list = []
    
    train_labels = []
    test_labels = []
    
    # Track document IDs
    doc_ids_train = []
    doc_ids_test = []
    
    int_to_author = {}
    n_features = None
    
    for author_ind, (author, filename) in enumerate(author_filename_dict.items()):
        int_to_author[author_ind] = author
        logger.info(f"Loading data for author {author_ind} {author} from {filename}")
        
        # Load dense activation data
        data, metadata = DataLoader().load_sae_activations(Path(path_to_data) / filename)
        
        if n_features is None:
            n_features = data.shape[2] if len(data.shape) == 3 else data.shape[1]
        
        # Get document lengths from metadata
        if not hasattr(metadata, 'doc_lengths'):
            raise ValueError(f"Doc lengths not found in metadata for {filename}. Possibly old files are used")
        
        doc_lengths = metadata.doc_lengths
        n_docs = len(doc_lengths)
        n_docs_train = int(n_docs * 0.8)
        
        logger.info(f"Author {author}: {n_docs} docs, {n_docs_train} for training")
        
        if sp.issparse(data):
            data = data.toarray()
            data = data.reshape(metadata.original_shape)
        
        logger.info(f"Shape of data: {data.shape}")
        
        # Process each document
        for doc_idx in range(n_docs):
            doc_length = doc_lengths[doc_idx]
            
            if from_token >= doc_length:
                continue
            
            # For dense data - aggregate per document
            doc_tokens = data[doc_idx, from_token:doc_length, :]
            valid_token_indices = np.arange(from_token, doc_length)
            
            n_valid_tokens = len(valid_token_indices)
            
            if doc_idx < n_docs_train:
                # Training data - average over tokens
                train_activations_list.append(doc_tokens.sum(axis=0) / n_valid_tokens)
                train_labels.append(author_ind)
                doc_ids_train.append(doc_idx)
            else:
                # Test data - average over tokens
                test_activations_list.append(doc_tokens.sum(axis=0) / n_valid_tokens)
                test_labels.append(author_ind)
                doc_ids_test.append(doc_idx)
    
    # Convert lists to arrays
    train_activations = np.array(train_activations_list)
    test_activations = np.array(test_activations_list)
    train_labels = np.array(train_labels)
    test_labels = np.array(test_labels)
    doc_ids_train = np.array(doc_ids_train)
    doc_ids_test = np.array(doc_ids_test)
    
    logger.info(f"Shape of train data: {train_activations.shape}")
    logger.info(f"Shape of test data: {test_activations.shape}")
    logger.info(f"Train data labels shape: {train_labels.shape}")
    logger.info(f"Test data labels shape: {test_labels.shape}")
    
    return train_activations, test_activations, train_labels, test_labels, int_to_author, doc_ids_train, doc_ids_test

