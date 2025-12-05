import argparse
import os
from collections import defaultdict
from pathlib import Path
import numpy as np
import scipy.sparse as sp
from dataclasses import dataclass
import logging

import json
import matplotlib.pyplot as plt
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import pandas as pd
from sklearn.metrics import classification_report, precision_score, recall_score, f1_score
import shap


# Import shared utilities
from backend.src.utils.shared_utilities import (
    ActivationFilenamesLoader,
    DataLoader
)
from backend.src.analysis.analysis_run_tracking import (
    get_data_and_output_paths,
    AnalysisRunTracker
)

# Set up logging
logger = logging.getLogger(__name__)


def parse_arguments():
    """Parse arguments for feature importance analysis"""
    parser = argparse.ArgumentParser(description="SAE Feature Importance Analysis Tool")
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
        help="Directory containing raw features (required if --run_id not provided)"
    )
    parser.add_argument(
        "--include_authors",
        type=str,
        nargs="+",
        default=None,
        help="The authors to include in the analysis"
    )
    parser.add_argument(
        "--include_layer_types",
        type=str,
        nargs="+",
        default=None,
        choices=["res", "mlp", "att"],
        help="The layer types to include in the analysis"
    )
    parser.add_argument(
        "--include_layer_inds",
        type=int,
        nargs="+",
        default=None,
        help="The layers to include in the analysis"
    )
    parser.add_argument(
        "--include_prompted",
        type=str,
        default="baseline",
        choices=["prompted", "baseline"],
        help="The prompted to include in the analysis"
    )
    parser.add_argument(
        "--from_token",
        type=int,
        default=10,
        help="The token to start from"
    )
    parser.add_argument(
        "--run_iterative_analysis",
        action="store_true",
        default=True,
        help="Run iterative variance threshold analysis instead of standard analysis"
    )
    parser.add_argument(
        "--min_features",
        type=int,
        default=10,
        help="Minimum number of features to keep in iterative analysis"
    )
    parser.add_argument(
        "--target_iterations",
        type=int,
        default=100,
        help="Target number of iterations for adaptive threshold computation"
    )
    parser.add_argument(
        "--use_adaptive_thresholds",
        action="store_true",
        default=True,
        help="Use adaptive thresholds in iterative analysis (default: True)"
    )
    parser.add_argument(
        "--no_adaptive_thresholds",
        action="store_true",
        help="Disable adaptive thresholds and use fixed threshold"
    )

    args = parser.parse_args()

    # Handle adaptive thresholds flag
    if args.no_adaptive_thresholds:
        args.use_adaptive_thresholds = False

    return args

@dataclass
class FeaturesData:
    train_data: np.ndarray
    test_data: np.ndarray   # Changed to sparse matrix
    train_labels: np.ndarray
    test_labels: np.ndarray
    train_doc_ids: np.ndarray
    test_doc_ids: np.ndarray

class FeatureSelectionForClassification:
    def __init__(self):
        pass

    def run_feature_selection(self):
        """Run feature selection"""
        pass

    def print_stats_of_train_data(self, features_data=None):
        """Print stats of train data to check that the data is being transformed correctly"""

        if features_data is None:
            features_data = self.features_data

        logger.info(f"Train data shape: {features_data.train_data.shape}")
        logger.info(f"Train data labels shape: {features_data.train_labels.shape}")
        logger.info(f"Train data doc ids shape: {features_data.train_doc_ids.shape}")
        logger.info(f"Train data labels sum: {features_data.train_labels.sum()}")


class VarianceThresholdFeatureSelection(FeatureSelectionForClassification):
    def __init__(self, features_data: FeaturesData, threshold=0.01):
        self.threshold = threshold
        self.features_data = features_data

    def run_feature_selection(self):
        """Run feature selection on dense matrices"""
        logger.info(f"Applying VarianceThreshold on dense matrix with threshold {self.threshold}")

        # VarianceThreshold works with dense matrices
        variance_threshold = VarianceThreshold(threshold=self.threshold)
        train_activations_reduced = variance_threshold.fit_transform(self.features_data.train_data)

        logger.info(f"VarianceThreshold selected {train_activations_reduced.shape[1]} features from {self.features_data.train_data.shape[1]}")
        most_important_features_author = list(variance_threshold.get_feature_names_out())
        logger.debug(f"VarianceThreshold features: {most_important_features_author[:10]}...")

        # Apply same transformation to test data
        test_activations_reduced = variance_threshold.transform(self.features_data.test_data)

        logger.info(f"After removing zero-activation rows: train {train_activations_reduced.shape}, test {test_activations_reduced.shape}")

        out = FeaturesData(
            train_data=train_activations_reduced,
            test_data=test_activations_reduced,
            train_labels=self.features_data.train_labels,
            test_labels=self.features_data.test_labels,
            train_doc_ids=self.features_data.train_doc_ids,
            test_doc_ids=self.features_data.test_doc_ids
        )

        self.print_stats_of_train_data(out)

        return out, most_important_features_author

class SelectKBestFeatureSelection(FeatureSelectionForClassification):
    def __init__(self, features_data: FeaturesData, inherited_features_from_previous_step, approach=f_classif, k=10):
        self.inherited_features_from_previous_step = inherited_features_from_previous_step
        self.approach = approach
        self.features_data = features_data
        self.k = k

    def run_feature_selection(self):
        """Run feature selection on dense matrices"""
        # Apply SelectKBest on features chosen with VarianceThreshold
        select_k_best = SelectKBest(self.approach, k=self.k)
        logger.info(f"Fitting SelectKBest with k={self.k} on dense matrix")

        # SelectKBest works with dense matrices
        train_data_reduced = select_k_best.fit_transform(self.features_data.train_data, self.features_data.train_labels)
        current_feature_names = select_k_best.get_feature_names_out()
        original_feature_names = [self.inherited_features_from_previous_step[int(i.replace("x", ""))] for i in current_feature_names]
        logger.debug(f"SelectKBest features: {original_feature_names}")

        test_data_reduced = select_k_best.transform(self.features_data.test_data)

        out = FeaturesData(
            train_data=train_data_reduced,
            test_data=test_data_reduced,
            train_labels=self.features_data.train_labels,
            test_labels=self.features_data.test_labels,
            train_doc_ids=self.features_data.train_doc_ids,
            test_doc_ids=self.features_data.test_doc_ids
        )
        self.print_stats_of_train_data(out)
        return out, original_feature_names

class FeatureSelectionSequential(FeatureSelectionForClassification):
    def __init__(self, features_data: FeaturesData, inherited_features_from_previous_step, approach=LogisticRegression(), k=10):
        self.inherited_features_from_previous_step = inherited_features_from_previous_step
        self.approach = approach
        self.features_data = features_data
        self.k = k

    def run_feature_selection(self):
        """Run feature selection on dense matrices"""
        # Apply SequentialFeatureSelector - works with dense matrices if the estimator does
        sequential_feature_selector = SequentialFeatureSelector(self.approach, n_features_to_select=self.k, n_jobs=-1)
        logger.info(f"Applying SequentialFeatureSelector with k={self.k} on dense matrix")

        train_data_reduced = sequential_feature_selector.fit_transform(self.features_data.train_data, self.features_data.train_labels.ravel())

        current_feature_names = sequential_feature_selector.get_feature_names_out()
        original_feature_names = [self.inherited_features_from_previous_step[int(i.replace("x", ""))] for i in current_feature_names]
        logger.debug(f"SequentialFeatureSelector features: {original_feature_names}")

        test_data_reduced = sequential_feature_selector.transform(self.features_data.test_data)

        out = FeaturesData(
            train_data=train_data_reduced,
            test_data=test_data_reduced,
            train_labels=self.features_data.train_labels,
            test_labels=self.features_data.test_labels,
            train_doc_ids=self.features_data.train_doc_ids,
            test_doc_ids=self.features_data.test_doc_ids
        )
        self.print_stats_of_train_data(out)

        return out, original_feature_names


def compute_adaptive_thresholds(train_data, target_iterations=100, min_features=10):
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
    logger.info(f"Variance stats - min: {variances.min():.6f}, max: {variances.max():.6f}, mean: {variances.mean():.6f}, median: {np.median(variances):.6f}")

    # Sort variances to understand distribution
    sorted_variances = np.sort(variances)

    # Calculate how many features to have at each iteration
    # Use exponential decay to remove more features early
    feature_counts = []
    current = n_features

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


def run_classification_and_get_metrics(train_data, train_labels, test_data, test_labels, max_iter=1500):
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
        'f1_score': float(f1)
    }


class IterativeVarianceThresholdClassification(FeatureSelectionForClassification):
    """
    Iteratively applies VarianceThreshold and classification until reaching min_features.
    At each step, runs classification and stores metrics.
    """

    def __init__(self, features_data: FeaturesData, min_features=10, target_iterations=100,
                 use_adaptive_thresholds=True, fixed_threshold=None):
        """
        Args:
            features_data: FeaturesData object with train/test data and labels
            min_features: Minimum number of features to keep
            target_iterations: Target number of iterations (used for adaptive thresholds)
            use_adaptive_thresholds: If True, compute adaptive thresholds, else use fixed
            fixed_threshold: If use_adaptive_thresholds is False, use this fixed threshold
        """
        self.features_data = features_data
        self.min_features = min_features
        self.target_iterations = target_iterations
        self.use_adaptive_thresholds = use_adaptive_thresholds
        self.fixed_threshold = fixed_threshold if fixed_threshold is not None else 0.01
        self.results = []

    def run_feature_selection(self):
        """
        Run iterative variance threshold and classification.

        Returns:
            List of dictionaries containing results for each iteration
        """
        logger.info("Starting iterative variance threshold classification")
        logger.info(f"Initial features: {self.features_data.train_data.shape[1]}")
        logger.info(f"Target minimum features: {self.min_features}")

        # Compute adaptive thresholds if requested
        if self.use_adaptive_thresholds:
            thresholds = compute_adaptive_thresholds(
                self.features_data.train_data,
                target_iterations=self.target_iterations,
                min_features=self.min_features
            )
        else:
            # Use fixed threshold for all iterations
            thresholds = [self.fixed_threshold] * 1000  # Large number, will stop at min_features

        # Current data (will be updated in each iteration)
        current_train_data = self.features_data.train_data
        current_test_data = self.features_data.test_data

        # Track which features remain (initially all features)
        n_total_features = self.features_data.train_data.shape[1]
        remaining_features = np.arange(n_total_features)

        iteration = 0
        threshold_idx = 0

        while current_train_data.shape[1] > self.min_features and threshold_idx < len(thresholds):
            logger.info(f"\n{'='*60}")
            logger.info(f"Iteration {iteration + 1}")
            logger.info(f"Current features: {current_train_data.shape[1]}")

            # Step 1: Run classification with current features
            logger.info("Running classification...")
            metrics = run_classification_and_get_metrics(
                current_train_data,
                self.features_data.train_labels,
                current_test_data,
                self.features_data.test_labels
            )

            # Store results
            result = {
                'iteration': iteration + 1,
                'n_features': int(current_train_data.shape[1]),
                'selected_features': [f"x{int(idx)}" for idx in remaining_features],
                'precision_class_1': metrics['precision'],
                'recall_class_1': metrics['recall'],
                'f1_score_class_1': metrics['f1_score']
            }
            self.results.append(result)

            logger.info(f"Metrics - Precision: {metrics['precision']:.4f}, Recall: {metrics['recall']:.4f}, F1: {metrics['f1_score']:.4f}")

            # Check if we've reached minimum features
            if current_train_data.shape[1] <= self.min_features:
                logger.info(f"Reached minimum features ({self.min_features}). Stopping.")
                break

            # Step 2: Apply VarianceThreshold to remove features
            threshold = thresholds[threshold_idx]
            logger.info(f"Applying VarianceThreshold with threshold {threshold:.6f}")

            selector = VarianceThreshold(threshold=threshold)
            try:
                new_train_data = selector.fit_transform(current_train_data)

                # If no features were removed, increase threshold slightly or stop
                if new_train_data.shape[1] == current_train_data.shape[1]:
                    logger.warning(f"No features removed with threshold {threshold:.6f}")
                    # Try to increase threshold
                    threshold_idx += 1
                    if threshold_idx >= len(thresholds):
                        logger.info("No more thresholds to try. Stopping.")
                        break
                    continue

                # If we would go below min_features, stop
                if new_train_data.shape[1] < self.min_features:
                    logger.info(f"Next iteration would have {new_train_data.shape[1]} features (below minimum). Stopping.")
                    break

                new_test_data = selector.transform(current_test_data)

                # Update remaining features
                feature_mask = selector.get_support()
                remaining_features = remaining_features[feature_mask]

                logger.info(f"Features after variance threshold: {new_train_data.shape[1]} (removed {current_train_data.shape[1] - new_train_data.shape[1]})")

                # Update current data
                current_train_data = new_train_data
                current_test_data = new_test_data

            except Exception as e:
                logger.error(f"Error in VarianceThreshold: {e}")
                break

            iteration += 1
            threshold_idx += 1

            # Safety check to prevent infinite loops
            if iteration > 200:
                logger.warning("Reached maximum iterations (200). Stopping.")
                break

        logger.info(f"\n{'='*60}")
        logger.info(f"Iterative feature selection completed after {len(self.results)} iterations")
        logger.info(f"Final number of features: {current_train_data.shape[1]}")

        return self.results



def save_updated_metadata(metadata, save_path, description=""):
    """Save updated metadata after filtering operations"""
    metadata_to_save = {
        'description': f"Filtered metadata: {description}",
        'n_samples': len(metadata['doc_ids']),
        'n_unique_docs': len(np.unique(metadata['doc_ids'])),
        'n_unique_authors': len(np.unique(metadata['author_ids'])),
        'doc_ids': metadata['doc_ids'].tolist(),
        'tok_ids': metadata['tok_ids'].tolist(),
        'author_ids': metadata['author_ids'].tolist(),
        'valid_mask': metadata['valid_mask'].tolist()
    }

    with open(save_path, 'w') as f:
        json.dump(metadata_to_save, f, indent=2)

    logger.info(f"Saved updated metadata to {save_path}")
    logger.info(f"  Samples: {metadata_to_save['n_samples']}")
    logger.info(f"  Unique docs: {metadata_to_save['n_unique_docs']}")
    logger.info(f"  Unique authors: {metadata_to_save['n_unique_authors']}")

def run_shap_analysis(model, X_train, X_test, feature_names, output_path, author, layer_type, layer_ind, model_name="LogisticRegression"):
    """Run SHAP analysis on trained model and save results"""
    logger.info(f"Running SHAP analysis for {author} {layer_type} {layer_ind} with {model_name}")

    try:
        # For logistic regression, use LinearExplainer for better performance
        if model_name == "LogisticRegression":
            explainer = shap.LinearExplainer(model, X_train)
            shap_values = explainer.shap_values(X_test)

            # For binary classification, SHAP returns shape (n_samples, n_features)
            if len(shap_values.shape) == 3:
                shap_values = shap_values[1]  # Take positive class

        else:
            # For other models, use Explainer (slower but more general)
            explainer = shap.Explainer(model, X_train)
            shap_values = explainer(X_test)
            if hasattr(shap_values, 'values'):
                shap_values = shap_values.values

        # Calculate feature importance (mean absolute SHAP values)
        feature_importance = np.mean(np.abs(shap_values), axis=0)

        # Create feature importance DataFrame
        importance_df = pd.DataFrame({
            'feature_name': feature_names,
            'importance': feature_importance,
            'feature_index': range(len(feature_names))
        }).sort_values('importance', ascending=False)

        # Save feature importance
        importance_path = output_path / f"shap_feature_importance__{model_name.lower()}__{layer_type}__{layer_ind}__{author}.csv"
        importance_df.to_csv(importance_path, index=False)
        logger.info(f"Saved SHAP feature importance to {importance_path}")

        # Save SHAP values
        shap_values_path = output_path / f"shap_values__{model_name.lower()}__{layer_type}__{layer_ind}__{author}.npy"
        np.save(shap_values_path, shap_values)
        logger.info(f"Saved SHAP values to {shap_values_path}")

        # Create and save SHAP summary plot
        plt.figure(figsize=(10, 8))
        shap.summary_plot(shap_values, X_test, feature_names=feature_names, show=False, max_display=20)
        plot_path = output_path / f"shap_summary_plot__{model_name.lower()}__{layer_type}__{layer_ind}__{author}.png"
        plt.tight_layout()
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"Saved SHAP summary plot to {plot_path}")

        # Log top 10 most important features
        logger.info(f"Top 10 most important features for {author} {layer_type} {layer_ind}:")
        for idx, row in importance_df.head(10).iterrows():
            logger.info(f"  {row['feature_name']}: {row['importance']:.4f}")

        return importance_df, shap_values

    except Exception as e:
        logger.error(f"Error in SHAP analysis for {author} {layer_type} {layer_ind}: {str(e)}")
        return None, None

def retrieve_and_combine_author_data(author_filename_dict, path_to_data, from_token=0):
    """Retrieve and combine author data using dense matrices and proper metadata handling"""

    train_activations_list = []
    test_activations_list = []

    train_labels = []
    test_labels = []

    # Track metadata for proper document/token position mapping
    doc_ids_train =  []
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
        if hasattr(metadata, 'doc_lengths'):
            doc_lengths = metadata.doc_lengths
        else:
            raise ValueError(f"Doc lengths not found in metadata for {filename}. Possibly old files are used")

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
            # For dense data
            doc_tokens = data[doc_idx, from_token:doc_length, :]
            valid_token_indices = np.arange(from_token, doc_length)

            n_valid_tokens = len(valid_token_indices)

            if doc_idx < n_docs_train:
                # Training data
                train_activations_list.append(doc_tokens.sum(axis=0)/n_valid_tokens)
                train_labels.append(author_ind)

                # Update metadata
                doc_ids_train.append(doc_idx)
            else:
                # Test data
                test_activations_list.append(doc_tokens.sum(axis=0)/n_valid_tokens)
                test_labels.append(author_ind)

                # Update metadata
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

def get_features_data_shap(features_data, feature_names, top_features=50):
    feature_indices = [int(name.replace("x", "")) for name in feature_names[:top_features]]
    return FeaturesData(
        train_data=features_data.train_data[:, feature_indices],
        test_data=features_data.test_data[:, feature_indices],
        train_labels=features_data.train_labels,
        test_labels=features_data.test_labels,
        train_doc_ids=features_data.train_doc_ids,
        test_doc_ids=features_data.test_doc_ids
    )


def run_iterative_variance_threshold_analysis(
    author_filename_dict,
    path_to_data,
    output_path,
    layer_type,
    layer_ind,
    from_token=0,
    min_features=10,
    target_iterations=100,
    use_adaptive_thresholds=True
):
    """
    Run iterative variance threshold analysis for all authors in the dataset.

    Args:
        author_filename_dict: Dictionary mapping author names to filenames
        path_to_data: Path to the data directory
        output_path: Path to save results
        layer_type: Layer type (e.g., 'mlp', 'res', 'att')
        layer_ind: Layer index
        from_token: Starting token position
        min_features: Minimum number of features to keep
        target_iterations: Target number of iterations
        use_adaptive_thresholds: Whether to use adaptive thresholds

    Returns:
        Dictionary with results for all authors
    """
    logger.info(f"Starting iterative variance threshold analysis for {layer_type} layer {layer_ind}")

    # Load data
    train_activations, test_activations, train_labels, test_labels, int_to_author, train_doc_ids, test_doc_ids = \
        retrieve_and_combine_author_data(author_filename_dict, path_to_data, from_token=from_token)

    all_results = {}

    # Run analysis for each author (one-vs-all)
    for author_ind, author in int_to_author.items():
        logger.info(f"\n{'#'*70}")
        logger.info(f"Running iterative analysis for author: {author}")
        logger.info(f"{'#'*70}")

        # Encode labels as one-vs-all
        train_labels_binary = (train_labels == author_ind).astype(int)
        test_labels_binary = (test_labels == author_ind).astype(int)

        logger.info(f"Class distribution - Train: {train_labels_binary.sum()}/{len(train_labels_binary)}, "
                   f"Test: {test_labels_binary.sum()}/{len(test_labels_binary)}")

        # Create FeaturesData object
        features_data = FeaturesData(
            train_data=train_activations,
            test_data=test_activations,
            train_labels=train_labels_binary,
            test_labels=test_labels_binary,
            train_doc_ids=train_doc_ids,
            test_doc_ids=test_doc_ids
        )

        # Run iterative variance threshold classification
        iterative_selector = IterativeVarianceThresholdClassification(
            features_data=features_data,
            min_features=min_features,
            target_iterations=target_iterations,
            use_adaptive_thresholds=use_adaptive_thresholds
        )

        results = iterative_selector.run_feature_selection()

        # Store results
        all_results[author] = {
            'layer_type': layer_type,
            'layer_ind': layer_ind,
            'iterations': results,
            'n_iterations': len(results),
            'initial_features': results[0]['n_features'] if results else 0,
            'final_features': results[-1]['n_features'] if results else 0
        }

        # Save individual author results
        author_results_path = output_path / f"iterative_variance_threshold__{layer_type}__{layer_ind}__{author}.json"
        with open(author_results_path, 'w') as f:
            json.dump(all_results[author], f, indent=2)
        logger.info(f"Saved results for {author} to {author_results_path}")

    # Save combined results
    combined_results_path = output_path / f"iterative_variance_threshold_combined__{layer_type}__{layer_ind}.json"
    with open(combined_results_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    logger.info(f"Saved combined results to {combined_results_path}")

    return all_results

def main():
    """Main entry point for feature importance analysis"""
    args = parse_arguments()

    # Configure logging
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Validate that either run_id or path_to_data is provided
    if not args.run_id and not args.path_to_data:
        raise ValueError("Either --run_id or --path_to_data must be provided")
    
    # Get data and output paths
    analysis_type = "feature_selection_shap_iterative" if args.run_iterative_analysis else "feature_selection_shap"
    data_path, output_path, activation_run_info = get_data_and_output_paths(
        run_id=args.run_id,
        data_path=args.path_to_data,
        analysis_type=analysis_type,
        run_name_override=None
    )
    
    # Register analysis run
    analysis_tracker = AnalysisRunTracker()
    activation_run_id = activation_run_info.get('id') if activation_run_info else None
    if activation_run_id:
        analysis_id = analysis_tracker.register_analysis(
            activation_run_id=activation_run_id,
            analysis_type=analysis_type,
            data_path=str(data_path),
            output_path=str(output_path)
        )
        logger.info(f"Registered analysis run with ID: {analysis_id}")

    activation_filenames_structured = ActivationFilenamesLoader(
        data_dir=Path(data_path),
        include_authors=args.include_authors,
        include_layer_types=args.include_layer_types,
        include_layer_inds=args.include_layer_inds,
        include_prompted=args.include_prompted
    ).get_structured_filenames()

    # Check if we should run iterative analysis
    if args.run_iterative_analysis:
        logger.info("Running ITERATIVE VARIANCE THRESHOLD ANALYSIS mode")
        logger.info(f"Parameters: min_features={args.min_features}, target_iterations={args.target_iterations}, "
                   f"use_adaptive_thresholds={args.use_adaptive_thresholds}")

        for layer_type, layer_ind_dict in activation_filenames_structured.items():
            for layer_ind, author_filename_dict in layer_ind_dict.items():
                run_iterative_variance_threshold_analysis(
                    author_filename_dict=author_filename_dict,
                    path_to_data=str(data_path),
                    output_path=output_path,
                    layer_type=layer_type,
                    layer_ind=layer_ind,
                    from_token=args.from_token,
                    min_features=args.min_features,
                    target_iterations=args.target_iterations,
                    use_adaptive_thresholds=args.use_adaptive_thresholds
                )

        logger.info("Iterative variance threshold analysis completed!")
        return

    # Otherwise run standard analysis
    logger.info("Running STANDARD ANALYSIS mode")
    classification_results = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(float))))))

    for layer_type, layer_ind_dict in activation_filenames_structured.items():
        for layer_ind, author_filename_dict in layer_ind_dict.items():


            """if os.path.exists(Path(args.path_to_outputs) / f"{args.run_name}" / f"classification_results.json"):
                with open(Path(args.path_to_outputs) / f"{args.run_name}" / f"classification_results.json", "r+") as f:
                    classification_results_loaded = json.load(f)

                if layer_type in classification_results_loaded and layer_ind in classification_results_loaded[layer_type]:
                    logger.info(f"Classification results already computed for {layer_type} {layer_ind}")
                    continue"""

            train_activations, test_activations, train_labels, test_labels, int_to_author, train_doc_ids, test_doc_ids = retrieve_and_combine_author_data(author_filename_dict, str(data_path), from_token=args.from_token)

            most_important_features = defaultdict(lambda: defaultdict(list))

            # Apply dfferent feature selection methods one vs all
            for author_ind, author in int_to_author.items():
                logger.info(f"Applying feature selection for {author}")

                # Get labels encoded one vs all
                labels_encoded_one_vs_all = train_labels.copy()
                labels_encoded_one_vs_all[train_labels == author_ind] = 1
                labels_encoded_one_vs_all[train_labels != author_ind] = 0

                labels_encoded_one_vs_all_test = test_labels.copy()
                labels_encoded_one_vs_all_test[test_labels == author_ind] = 1
                labels_encoded_one_vs_all_test[test_labels != author_ind] = 0

                logger.debug(f"Train data labels shape: {labels_encoded_one_vs_all.shape}")
                logger.debug(f"Train data labels sum: {labels_encoded_one_vs_all.sum()}")

                logger.debug(f"Test data labels shape: {labels_encoded_one_vs_all_test.shape}")
                logger.debug(f"Test data labels sum: {labels_encoded_one_vs_all_test.sum()}")

                features_data_initial = FeaturesData(
                    train_data=train_activations,
                    test_data=test_activations,
                    train_labels=labels_encoded_one_vs_all,
                    test_labels=labels_encoded_one_vs_all_test,
                    train_doc_ids=train_doc_ids,
                    test_doc_ids=test_doc_ids
                )

                features_data_variance_threshold, most_important_features_variance_threshold = VarianceThresholdFeatureSelection(features_data_initial).run_feature_selection()
                most_important_features[author]["variance_threshold"] = most_important_features_variance_threshold

                # Save updated metadata after variance threshold filtering
                """save_updated_metadata(
                    features_data_variance_threshold.train_metadata,
                    Path(args.path_to_outputs) / f"{args.run_name}" / f"train_metadata_variance_threshold__{layer_type}__{layer_ind}__{author}.json",
                    f"After VarianceThreshold for {author} {layer_type} {layer_ind}"
                )
                save_updated_metadata(
                    features_data_variance_threshold.test_metadata,
                    Path(args.path_to_outputs) / f"{args.run_name}" / f"test_metadata_variance_threshold__{layer_type}__{layer_ind}__{author}.json",
                    f"After VarianceThreshold for {author} {layer_type} {layer_ind}"
                )"""

                # Apply LogisticRegression on variance threshold filtered data
                logistic_regression = LogisticRegression(max_iter=1500)
                logger.info(f"Fitting LogisticRegression on variance threshold filtered data")
                logistic_regression.fit(features_data_variance_threshold.train_data, features_data_variance_threshold.train_labels)
                labels_predicted = logistic_regression.predict(features_data_variance_threshold.test_data)

                classification_report_logreg = classification_report(features_data_variance_threshold.test_labels, labels_predicted, output_dict=True)
                logger.info(f"LogisticRegression classification report: {classification_report_logreg}")
                classification_report_df = pd.DataFrame(classification_report_logreg)
                classification_report_df.to_csv(output_path / f"classification_report__logreg__{layer_type}__{layer_ind}__{author}__variance_threshold.csv")

                classification_results[layer_type][layer_ind][author]["variance_threshold"]["LogisticRegression"]["precision__class_1"] = classification_report_logreg["1"]["precision"]
                classification_results[layer_type][layer_ind][author]["variance_threshold"]["LogisticRegression"]["recall__class_1"] = classification_report_logreg["1"]["recall"]
                classification_results[layer_type][layer_ind][author]["variance_threshold"]["LogisticRegression"]["f1__class_1"] = classification_report_logreg["1"]["f1-score"]

                # Run SHAP analysis on logistic regression with variance threshold features
                shap_importance_df, shap_values = run_shap_analysis(
                    model=logistic_regression,
                    X_train=features_data_variance_threshold.train_data,
                    X_test=features_data_variance_threshold.test_data,
                    feature_names=most_important_features_variance_threshold,
                    output_path=output_path,
                    author=author,
                    layer_type=layer_type,
                    layer_ind=layer_ind,
                    model_name="LogisticRegression"
                )

                if shap_importance_df is not None:
                    most_important_features[author]["shap_variance_threshold"] = shap_importance_df['feature_name'].tolist()
                    most_important_features[author]["shap_importance_scores"] = shap_importance_df['importance'].tolist()

                """features_data_shap = get_features_data_shap(features_data_initial, shap_importance_df['feature_name'].tolist())

                # Run another round of logistic regression on shap features
                logistic_regression = LogisticRegression(max_iter=1500)
                logger.info(f"Fitting LogisticRegression on shap features")
                logistic_regression.fit(features_data_shap.train_data, features_data_shap.train_labels)
                labels_predicted = logistic_regression.predict(features_data_shap.test_data)


                classification_report_logreg = classification_report(features_data_shap.test_labels, labels_predicted, output_dict=True)
                logger.info(f"LogisticRegression classification report: {classification_report_logreg}")
                classification_report_df = pd.DataFrame(classification_report_logreg)
                classification_report_df.to_csv(Path(args.path_to_outputs) / f"{args.run_name}" / f"classification_report__logreg__{layer_type}__{layer_ind}__{author}__shap_variance_threshold.csv")

                classification_results[layer_type][layer_ind][author]["shap_variance_threshold"]["LogisticRegression"]["precision__class_1"] = classification_report_logreg["1"]["precision"]
                classification_results[layer_type][layer_ind][author]["shap_variance_threshold"]["LogisticRegression"]["recall__class_1"] = classification_report_logreg["1"]["recall"]
                classification_results[layer_type][layer_ind][author]["shap_variance_threshold"]["LogisticRegression"]["f1__class_1"] = classification_report_logreg["1"]["f1-score"]"""


                svm = SVC()
                logger.info(f"Fitting SVM")
                svm.fit(features_data_variance_threshold.train_data, features_data_variance_threshold.train_labels)
                labels_predicted = svm.predict(features_data_variance_threshold.test_data)
                classification_report_svm = classification_report(features_data_variance_threshold.test_labels, labels_predicted, output_dict=True)
                logger.info(f"SVM classification report: {classification_report_svm}")
                classification_report_df = pd.DataFrame(classification_report_svm)
                classification_report_df.to_csv(output_path / f"classification_report__svm__{layer_type}__{layer_ind}__{author}__variance_threshold.csv")

                classification_results[layer_type][layer_ind][author]["variance_threshold"]["SVM"]["precision__class_1"] = classification_report_svm["1"]["precision"]
                classification_results[layer_type][layer_ind][author]["variance_threshold"]["SVM"]["recall__class_1"] = classification_report_svm["1"]["recall"]
                classification_results[layer_type][layer_ind][author]["variance_threshold"]["SVM"]["f1__class_1"] = classification_report_svm["1"]["f1-score"]

                features_data_select_k_best, most_important_features_select_k_best = SelectKBestFeatureSelection(features_data_variance_threshold, most_important_features_variance_threshold, approach=f_classif, k=100).run_feature_selection()
                most_important_features[author]["select_k_best"] = most_important_features_select_k_best

                features_data_sequential_feature_selector, most_important_features_sequential_feature_selector = FeatureSelectionSequential(features_data_select_k_best, most_important_features_select_k_best, approach=LogisticRegression(), k=10).run_feature_selection()
                most_important_features[author]["sequential_feature_selector"] = most_important_features_sequential_feature_selector

                # Save metadata after sequential feature selection
                save_updated_metadata(
                    features_data_sequential_feature_selector.train_metadata,
                    output_path / f"train_metadata_sequential__{layer_type}__{layer_ind}__{author}.json",
                    f"After SequentialFeatureSelector for {author} {layer_type} {layer_ind}"
                )
                save_updated_metadata(
                    features_data_sequential_feature_selector.test_metadata,
                    output_path / f"test_metadata_sequential__{layer_type}__{layer_ind}__{author}.json",
                    f"After SequentialFeatureSelector for {author} {layer_type} {layer_ind}"
                )

                # Apply LogisticRegression (works with sparse matrices)
                logistic_regression = LogisticRegression(max_iter=1000)  # Increased max_iter for sparse data
                logger.info(f"Fitting LogisticRegression on sparse data")
                logistic_regression.fit(features_data_sequential_feature_selector.train_data, features_data_sequential_feature_selector.train_labels)
                labels_predicted = logistic_regression.predict(features_data_sequential_feature_selector.test_data)
                

                
                classification_report_logreg = classification_report(features_data_sequential_feature_selector.test_labels, labels_predicted, output_dict=True)
                logger.info(f"LogisticRegression classification report: {classification_report_logreg}")
                classification_report_df = pd.DataFrame(classification_report_logreg)
                classification_report_df.to_csv(output_path / f"classification_report__logreg__{layer_type}__{layer_ind}__{author}__sequential_feature_selector.csv")

                classification_results[layer_type][layer_ind][author]["sequential_feature_selector"]["LogisticRegression"]["precision__class_1"] = classification_report_logreg["1"]["precision"]
                classification_results[layer_type][layer_ind][author]["sequential_feature_selector"]["LogisticRegression"]["recall__class_1"] = classification_report_logreg["1"]["recall"]
                classification_results[layer_type][layer_ind][author]["sequential_feature_selector"]["LogisticRegression"]["f1__class_1"] = classification_report_logreg["1"]["f1-score"]
                
                # Apply SVM (works with sparse matrices)
                svm = SVC(kernel='linear')  # Linear kernel works better with sparse high-dimensional data
                logger.info(f"Fitting SVM on sparse data")
                svm.fit(features_data_sequential_feature_selector.train_data, features_data_sequential_feature_selector.train_labels)
                labels_predicted_svm = svm.predict(features_data_sequential_feature_selector.test_data)

                
                classification_report_svm = classification_report(features_data_sequential_feature_selector.test_labels, labels_predicted_svm, output_dict=True)
                logger.info(f"SVM classification report: {classification_report_svm}")
                classification_report_df = pd.DataFrame(classification_report_svm)
                classification_report_df.to_csv(output_path / f"classification_report__svm__{layer_type}__{layer_ind}__{author}__sequential_feature_selector.csv")

                classification_results[layer_type][layer_ind][author]["sequential_feature_selector"]["SVM"]["precision__class_1"] = classification_report_svm["1"]["precision"]
                classification_results[layer_type][layer_ind][author]["sequential_feature_selector"]["SVM"]["recall__class_1"] = classification_report_svm["1"]["recall"]
                classification_results[layer_type][layer_ind][author]["sequential_feature_selector"]["SVM"]["f1__class_1"] = classification_report_svm["1"]["f1-score"]


                """random_forest = RandomForestClassifier()
                logger.info(f"Fitting RandomForest")
                cv_classification_random_forest_score = cross_validate(random_forest, author_data_numerical_sequential_feature_selector_filtered, labels_encoded_one_vs_all_variance_threshold_filtered, cv=cv, scoring=scoring, n_jobs=-1)
                logger.info(f"RandomForest classification scores: {cv_classification_random_forest_score}")
                for score in scoring:
                    classification_results[layer_type][layer_ind][author]["RandomForest"][score] = np.mean(cv_classification_random_forest_score[f"test_{score}"])"""

            logger.info(f"Most important features: {most_important_features}")
            
            # Save most important features
            with open(output_path / f"most_important_features__{layer_type}__{layer_ind}.json", "w") as f:
                json.dump(most_important_features, f, indent=4)

                    
            logger.info(f"Classification results: {classification_results}")

            # Save classification results
            with open(output_path / f"classification_results.json", "w") as f:
                json.dump(classification_results, f, indent=4)
    


if __name__ == "__main__":
    main()
