"""
Base classes and utilities for feature selection analysis.

This module contains shared functionality used across different feature selection modes:
- Token-level feature selection
- Aggregated (document-level) feature selection with SHAP
- Iterative feature selection
"""

import numpy as np
import scipy.sparse as sp
from dataclasses import dataclass
from typing import Optional, List, Dict, Any, Tuple
import logging

from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_classif
from sklearn.feature_selection import SequentialFeatureSelector as SklearnSequentialFeatureSelector
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import classification_report, precision_score, recall_score, f1_score

logger = logging.getLogger(__name__)


@dataclass
class FeaturesData:
    """Container for feature data with train/test splits."""
    train_data: Any  # Can be np.ndarray or sp.csr_matrix
    test_data: Any   # Can be np.ndarray or sp.csr_matrix
    train_labels: np.ndarray
    test_labels: np.ndarray
    train_doc_ids: Optional[np.ndarray] = None
    test_doc_ids: Optional[np.ndarray] = None
    train_metadata: Optional[Dict[str, Any]] = None
    test_metadata: Optional[Dict[str, Any]] = None


class FeatureSelectionBase:
    """Base class for feature selection methods."""
    
    def __init__(self, features_data: FeaturesData):
        self.features_data = features_data
    
    def run_feature_selection(self):
        """Run feature selection. Must be implemented by subclasses."""
        raise NotImplementedError
    
    def print_stats_of_train_data(self, features_data: Optional[FeaturesData] = None):
        """Print statistics about the training data."""
        if features_data is None:
            features_data = self.features_data
        
        logger.info(f"Train data shape: {features_data.train_data.shape}")
        if sp.issparse(features_data.train_data):
            sparsity = 1 - features_data.train_data.nnz / (features_data.train_data.shape[0] * features_data.train_data.shape[1])
            logger.info(f"Train data sparsity: {sparsity:.4f}")
            logger.info(f"Train data non-zero elements: {features_data.train_data.nnz}")
        logger.info(f"Train data labels shape: {features_data.train_labels.shape}")
        logger.info(f"Train data labels sum: {features_data.train_labels.sum()}")


class VarianceThresholdFeatureSelection(FeatureSelectionBase):
    """Apply variance threshold feature selection."""
    
    def __init__(self, features_data: FeaturesData, threshold: float = 0.01):
        super().__init__(features_data)
        self.threshold = threshold
    
    def run_feature_selection(self) -> Tuple[FeaturesData, List[str]]:
        """Run variance threshold feature selection."""
        logger.info(f"Applying VarianceThreshold with threshold {self.threshold}")
        
        variance_threshold = VarianceThreshold(threshold=self.threshold)
        train_activations_reduced = variance_threshold.fit_transform(self.features_data.train_data)
        
        logger.info(f"VarianceThreshold selected {train_activations_reduced.shape[1]} features from {self.features_data.train_data.shape[1]}")
        most_important_features = list(variance_threshold.get_feature_names_out())
        logger.debug(f"VarianceThreshold features: {most_important_features[:10]}...")
        
        # Handle zero-activation rows for sparse matrices
        if sp.issparse(train_activations_reduced):
            non_zero_indices = np.diff(train_activations_reduced.indptr).nonzero()[0]
            train_activations_reduced = train_activations_reduced[non_zero_indices, :]
            train_labels = self.features_data.train_labels[non_zero_indices]
            train_doc_ids = self.features_data.train_doc_ids[non_zero_indices] if self.features_data.train_doc_ids is not None else None
        else:
            train_labels = self.features_data.train_labels
            train_doc_ids = self.features_data.train_doc_ids
        
        # Apply same transformation to test data
        test_activations_reduced = variance_threshold.transform(self.features_data.test_data)
        if sp.issparse(test_activations_reduced):
            non_zero_indices_test = np.diff(test_activations_reduced.indptr).nonzero()[0]
            test_activations_reduced = test_activations_reduced[non_zero_indices_test, :]
            test_labels = self.features_data.test_labels[non_zero_indices_test]
            test_doc_ids = self.features_data.test_doc_ids[non_zero_indices_test] if self.features_data.test_doc_ids is not None else None
        else:
            test_labels = self.features_data.test_labels
            test_doc_ids = self.features_data.test_doc_ids
        
        logger.info(f"After removing zero-activation rows: train {train_activations_reduced.shape}, test {test_activations_reduced.shape}")
        
        out = FeaturesData(
            train_data=train_activations_reduced,
            test_data=test_activations_reduced,
            train_labels=train_labels,
            test_labels=test_labels,
            train_doc_ids=train_doc_ids,
            test_doc_ids=test_doc_ids,
            train_metadata=self.features_data.train_metadata,
            test_metadata=self.features_data.test_metadata
        )
        
        self.print_stats_of_train_data(out)
        
        return out, most_important_features


class SelectKBestFeatureSelection(FeatureSelectionBase):
    """Apply SelectKBest feature selection."""
    
    def __init__(self, features_data: FeaturesData, inherited_features_from_previous_step: List[str], 
                 approach=f_classif, k: int = 10):
        super().__init__(features_data)
        self.inherited_features_from_previous_step = inherited_features_from_previous_step
        self.approach = approach
        self.k = k
    
    def run_feature_selection(self) -> Tuple[FeaturesData, List[str]]:
        """Run SelectKBest feature selection."""
        select_k_best = SelectKBest(self.approach, k=self.k)
        logger.info(f"Fitting SelectKBest with k={self.k}")
        
        train_data_reduced = select_k_best.fit_transform(self.features_data.train_data, self.features_data.train_labels)
        current_feature_names = select_k_best.get_feature_names_out()
        
        # Map back to original feature indices through the inherited feature list
        original_feature_names = [
            self.inherited_features_from_previous_step[int(i.replace("x", ""))] 
            for i in current_feature_names
        ]
        #logger.debug(f"SelectKBest features: {original_feature_names}")
        
        test_data_reduced = select_k_best.transform(self.features_data.test_data)
        
        out = FeaturesData(
            train_data=train_data_reduced,
            test_data=test_data_reduced,
            train_labels=self.features_data.train_labels,
            test_labels=self.features_data.test_labels,
            train_doc_ids=self.features_data.train_doc_ids,
            test_doc_ids=self.features_data.test_doc_ids,
            train_metadata=self.features_data.train_metadata,
            test_metadata=self.features_data.test_metadata
        )
        
        self.print_stats_of_train_data(out)
        return out, original_feature_names


class SequentialFeatureSelectorWrapper(FeatureSelectionBase):
    """Apply SequentialFeatureSelector."""
    
    def __init__(self, features_data: FeaturesData, inherited_features_from_previous_step: List[str],
                 approach=LogisticRegression(), k: int = 10):
        super().__init__(features_data)
        self.inherited_features_from_previous_step = inherited_features_from_previous_step
        self.approach = approach
        self.k = k
    
    def run_feature_selection(self) -> Tuple[FeaturesData, List[str]]:
        """Run SequentialFeatureSelector."""
        sequential_feature_selector = SklearnSequentialFeatureSelector(
            self.approach, n_features_to_select=self.k, n_jobs=-1
        )
        logger.info(f"Applying SequentialFeatureSelector with k={self.k}")
        
        train_data_reduced = sequential_feature_selector.fit_transform(
            self.features_data.train_data, 
            self.features_data.train_labels.ravel()
        )
        
        current_feature_names = sequential_feature_selector.get_feature_names_out()
        original_feature_names = [
            self.inherited_features_from_previous_step[int(i.replace("x", ""))] 
            for i in current_feature_names
        ]
        #logger.debug(f"SequentialFeatureSelector features: {original_feature_names}")
        
        test_data_reduced = sequential_feature_selector.transform(self.features_data.test_data)
        
        out = FeaturesData(
            train_data=train_data_reduced,
            test_data=test_data_reduced,
            train_labels=self.features_data.train_labels,
            test_labels=self.features_data.test_labels,
            train_doc_ids=self.features_data.train_doc_ids,
            test_doc_ids=self.features_data.test_doc_ids,
            train_metadata=self.features_data.train_metadata,
            test_metadata=self.features_data.test_metadata
        )
        
        self.print_stats_of_train_data(out)
        return out, original_feature_names


def run_classification_and_get_metrics(
    train_data: Any,
    train_labels: np.ndarray,
    test_data: Any,
    test_labels: np.ndarray,
    max_iter: int = 1500
) -> Dict[str, float]:
    """
    Run logistic regression and return metrics for class 1.
    
    Args:
        train_data: Training features
        train_labels: Training labels
        test_data: Test features
        test_labels: Test labels
        max_iter: Maximum iterations for LogisticRegression
    
    Returns:
        Dictionary with precision, recall, f1_score for class 1
    """
    model = LogisticRegression(max_iter=max_iter, random_state=42)
    model.fit(train_data, train_labels)
    predictions = model.predict(test_data)
    
    # Calculate metrics for class 1
    precision = precision_score(test_labels, predictions, pos_label=1, zero_division=0)
    recall = recall_score(test_labels, predictions, pos_label=1, zero_division=0)
    f1 = f1_score(test_labels, predictions, pos_label=1, zero_division=0)
    
    return {
        'precision': float(precision),
        'recall': float(recall),
        'f1_score': float(f1),
        'model': model
    }


def compute_adaptive_thresholds(
    train_data: np.ndarray,
    target_iterations: int = 100,
    min_features: int = 10
) -> List[float]:
    """
    Compute adaptive variance thresholds to achieve approximately target_iterations
    with more features removed early and fewer later.
    
    Args:
        train_data: Training data array
        target_iterations: Target number of iterations
        min_features: Minimum number of features to keep
    
    Returns:
        List of variance thresholds for each iteration
    """
    logger.info("Computing adaptive variance thresholds")
    
    # Compute variance for each feature
    variances = np.var(train_data, axis=0)
    n_features = train_data.shape[1]
    
    logger.info(f"Initial features: {n_features}")
    logger.info(f"Variance stats - min: {variances.min():.6f}, max: {variances.max():.6f}, "
                f"mean: {variances.mean():.6f}, median: {np.median(variances):.6f}")
    
    # Sort variances to understand distribution
    sorted_variances = np.sort(variances)
    
    # Calculate how many features to have at each iteration
    # Use exponential decay to remove more features early
    feature_counts = []
    
    # Generate exponential decay for feature counts
    decay_rate = np.log(n_features / min_features) / target_iterations
    for i in range(target_iterations):
        next_count = int(n_features * np.exp(-decay_rate * (i + 1)))
        if next_count <= min_features:
            feature_counts.append(min_features)
            break
        feature_counts.append(next_count)
    
    # Convert feature counts to thresholds
    thresholds = []
    for count in feature_counts:
        if count >= len(sorted_variances):
            thresholds.append(0.0)
        else:
            # Threshold is the variance of the count-th feature from the end
            threshold = sorted_variances[-(count + 1)]
            thresholds.append(threshold)
    
    logger.info(f"Generated {len(thresholds)} thresholds for approximately {len(feature_counts)} iterations")
    logger.info(f"First 5 thresholds: {thresholds[:5]}")
    logger.info(f"First 5 target feature counts: {feature_counts[:5]}")
    
    return thresholds

