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
import seaborn as sns
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
from sklearn.metrics import classification_report


"""from cuml.common import logger;
logger.set_level(logger.level_enum.info)"""

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
    
    args = parser.parse_args()

    return args

@dataclass
class FeaturesData:
    train_data: sp.csr_matrix  # Changed to sparse matrix
    test_data: sp.csr_matrix   # Changed to sparse matrix
    train_labels: np.ndarray
    test_labels: np.ndarray
    train_metadata: dict = None  # Added metadata tracking
    test_metadata: dict = None   # Added metadata tracking

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
        logger.info(f"Train data sparsity: {1 - features_data.train_data.nnz / (features_data.train_data.shape[0] * features_data.train_data.shape[1]):.4f}")
        logger.info(f"Train data non-zero elements: {features_data.train_data.nnz}")
        logger.info(f"Train data non-zero elements per row: {features_data.train_data.nnz / features_data.train_data.shape[0]:.4f}")
        logger.info(f"Train data non-zero elements per col: {features_data.train_data.nnz / features_data.train_data.shape[1]:.4f}")
        logger.info(f"Train data labels shape: {features_data.train_labels.shape}")
        logger.info(f"Train data labels sum: {features_data.train_labels.sum()}")

        # Get the indptr array
        indptr = features_data.train_data.indptr

        # Calculate the number of non-zeros per row
        num_nonzeros_per_row = np.diff(indptr)

        # Find the index of the all-zero row
        zero_row_indices = np.where(num_nonzeros_per_row == 0)[0]

        logger.info(f"All-zero row(s): {len(zero_row_indices)}")

class VarianceThresholdFeatureSelection(FeatureSelectionForClassification):
    def __init__(self, features_data: FeaturesData, threshold=0.01):
        self.threshold = threshold
        self.features_data = features_data

    def run_feature_selection(self):
        """Run feature selection on sparse matrices"""
        logger.info(f"Applying VarianceThreshold on sparse matrix with threshold {self.threshold}")
        
        # VarianceThreshold works with sparse matrices
        variance_threshold = VarianceThreshold(threshold=self.threshold)
        train_activations_reduced = variance_threshold.fit_transform(self.features_data.train_data)

        logger.info(f"VarianceThreshold selected {train_activations_reduced.shape[1]} features from {self.features_data.train_data.shape[1]}")
        most_important_features_author = list(variance_threshold.get_feature_names_out())
        logger.debug(f"VarianceThreshold features: {most_important_features_author[:10]}...") 

        # For sparse matrices, use different approach to find non-zero rows
        if sp.issparse(train_activations_reduced):
            non_zero_doctok_indices = np.diff(train_activations_reduced.indptr).nonzero()[0]
        else:
            non_zero_doctok_indices = np.argwhere(train_activations_reduced.sum(axis=1) > 0)[:, 0]
        
        train_activations_reduced = train_activations_reduced[non_zero_doctok_indices, :]
        train_labels_one_vs_all = self.features_data.train_labels[non_zero_doctok_indices]

        # Apply same transformation to test data
        test_activations_reduced = variance_threshold.transform(self.features_data.test_data)
        if sp.issparse(test_activations_reduced):
            non_zero_doctok_indices_test = np.diff(test_activations_reduced.indptr).nonzero()[0]
        else:
            non_zero_doctok_indices_test = np.argwhere(test_activations_reduced.sum(axis=1) > 0)[:, 0]
        
        test_activations_reduced = test_activations_reduced[non_zero_doctok_indices_test, :]
        test_labels_one_vs_all = self.features_data.test_labels[non_zero_doctok_indices_test]

        logger.info(f"After removing zero-activation rows: train {train_activations_reduced.shape}, test {test_activations_reduced.shape}")

        out = FeaturesData(
            train_data=train_activations_reduced, 
            test_data=test_activations_reduced, 
            train_labels=train_labels_one_vs_all, 
            test_labels=test_labels_one_vs_all, 
            train_metadata=self.features_data.train_metadata,
            test_metadata=self.features_data.test_metadata
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
        """Run feature selection on sparse matrices"""
        # Apply SelectKBest on features chosen with VarianceThreshold
        select_k_best = SelectKBest(self.approach, k=self.k)
        logger.info(f"Fitting SelectKBest with k={self.k} on sparse matrix")
        
        # SelectKBest works with sparse matrices
        train_data_reduced = select_k_best.fit_transform(self.features_data.train_data, self.features_data.train_labels)
        current_feature_names = select_k_best.get_feature_names_out()
        # Map back to original feature indices through the inherited feature list
        original_feature_names = [self.inherited_features_from_previous_step[int(i.replace("x", ""))] for i in current_feature_names]
        logger.debug(f"SelectKBest features: {original_feature_names}")

        test_data_reduced = select_k_best.transform(self.features_data.test_data)
        
        out = FeaturesData(
            train_data=train_data_reduced, 
            test_data=test_data_reduced, 
            train_labels=self.features_data.train_labels, 
            test_labels=self.features_data.test_labels, 
            train_metadata=self.features_data.train_metadata,
            test_metadata=self.features_data.test_metadata
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
        """Run feature selection on sparse matrices"""
        # Apply SequentialFeatureSelector - works with sparse matrices if the estimator does
        sequential_feature_selector = SequentialFeatureSelector(self.approach, n_features_to_select=self.k, n_jobs=-1)
        logger.info(f"Applying SequentialFeatureSelector with k={self.k} on sparse matrix")
        
        train_data_reduced = sequential_feature_selector.fit_transform(self.features_data.train_data, self.features_data.train_labels.ravel())

        current_feature_names = sequential_feature_selector.get_feature_names_out()
        # Map back to original feature indices through the inherited feature list
        original_feature_names = [self.inherited_features_from_previous_step[int(i.replace("x", ""))] for i in current_feature_names]
        logger.debug(f"SequentialFeatureSelector features: {original_feature_names}")

        test_data_reduced = sequential_feature_selector.transform(self.features_data.test_data)
        
        out = FeaturesData(
            train_data=train_data_reduced, 
            test_data=test_data_reduced, 
            train_labels=self.features_data.train_labels, 
            test_labels=self.features_data.test_labels, 
            train_metadata=self.features_data.train_metadata,
            test_metadata=self.features_data.test_metadata
        )
        self.print_stats_of_train_data(out)
        
        return out, original_feature_names

        

def save_histogram_of_data(data, save_path):
    """Save histogram of data - handles both sparse and dense matrices"""
    if sp.issparse(data):
        # For sparse matrices, get non-zero data
        non_zero_data = data.data
        zero_count = data.shape[0] * data.shape[1] - data.nnz
    else:
        # For dense matrices
        zero_count = np.sum(data == 0)
        non_zero_data = data[data > 0]
    
    plt.figure(figsize=(12, 8))
    if len(non_zero_data) > 0:
        plt.hist(non_zero_data, bins=100)
    plt.title(f"Histogram of raw activations data for {save_path.split('.')[0]}. {zero_count} zero activations")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

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

def retrieve_and_combine_author_data(author_filename_dict, path_to_data, binary=True, from_token=0):
    """Retrieve and combine author data using sparse matrices and proper metadata handling"""

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
        if hasattr(metadata, 'doc_lengths'):
            doc_lengths = metadata.doc_lengths
        else:
            raise ValueError(f"Doc lengths not found in metadata for {filename}. Possibly old files are used")

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

    
def visualize_results_with_heatmap(predicted_labels, true_labels, metadata, title, save_path):
    """Visualize results with heatmap"""
    
    # construct back to matrix doc x tok
    max_docs = max(doc_ind for doc_ind in metadata['doc_ids'])+1
    max_toks = max(tok_ind for tok_ind in metadata['tok_ids'])+1
    predictions = np.full((max_docs, max_toks), np.nan)

    for i, (doc_ind, tok_ind) in enumerate(zip(metadata['doc_ids'], metadata['tok_ids'])):
        predictions[doc_ind, tok_ind] = predicted_labels[i] == true_labels[i]

    # transform boolean to int binary
    predictions = predictions.astype(int)

    sns.heatmap(predictions, cmap=["red", "green"], 
                    mask=np.isnan(predictions),  # mask so the nan cells aren’t coloured
                    cbar=True)

    plt.title(title)
    plt.xlabel("Token")
    plt.ylabel("Document")

    plt.savefig(save_path)
    plt.close()


def main():
    """Main entry point for feature importance analysis"""
    args = parse_arguments()
    
    # Configure logging
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Get data and output paths
    data_path, output_path, activation_run_info = get_data_and_output_paths(
        run_id=args.run_id,
        data_path=args.path_to_data,
        analysis_type="feature_selection",
        run_name_override=None
    )
    
    # Register analysis run
    analysis_tracker = AnalysisRunTracker()
    activation_run_id = activation_run_info.get('id') if activation_run_info else None
    if activation_run_id:
        analysis_id = analysis_tracker.register_analysis(
            activation_run_id=activation_run_id,
            analysis_type="feature_selection",
            data_path=str(data_path),
            output_path=str(output_path)
        )
        logger.info(f"Registered analysis run with ID: {analysis_id}")
    
    path_to_data = str(data_path)
    path_to_outputs = str(output_path)
    run_name = "feature_selection_analysis"  # Keep for compatibility with existing code

    activation_filenames_structured = ActivationFilenamesLoader(
        data_dir=Path(path_to_data), 
        include_authors=args.include_authors, 
        include_layer_types=args.include_layer_types, 
        include_layer_inds=args.include_layer_inds, 
        include_prompted=args.include_prompted
    ).get_structured_filenames()

    classification_results = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(float))))))
    
    for layer_type, layer_ind_dict in activation_filenames_structured.items():
        for layer_ind, author_filename_dict in layer_ind_dict.items():


            if os.path.exists(Path(path_to_outputs) / f"{run_name}" / f"classification_results.json"):
                with open(Path(path_to_outputs) / f"{run_name}" / f"classification_results.json", "r+") as f:
                    classification_results_loaded = json.load(f)

                if layer_type in classification_results_loaded and layer_ind in classification_results_loaded[layer_type]:
                    logger.info(f"Classification results already computed for {layer_type} {layer_ind}")
                    continue

            train_activations, test_activations, train_labels, test_labels, int_to_author, train_metadata, test_metadata = retrieve_and_combine_author_data(author_filename_dict, path_to_data, binary=True, from_token=args.from_token)

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
                    train_metadata=train_metadata,
                    test_metadata=test_metadata
                )

                features_data_variance_threshold, most_important_features_variance_threshold = VarianceThresholdFeatureSelection(features_data_initial).run_feature_selection()
                most_important_features[author]["variance_threshold"] = most_important_features_variance_threshold
                
                # Save updated metadata after variance threshold filtering
                save_updated_metadata(
                    features_data_variance_threshold.train_metadata,
                    Path(path_to_outputs) / f"{run_name}" / f"train_metadata_variance_threshold__{layer_type}__{layer_ind}__{author}.json",
                    f"After VarianceThreshold for {author} {layer_type} {layer_ind}"
                )
                save_updated_metadata(
                    features_data_variance_threshold.test_metadata,
                    Path(path_to_outputs) / f"{run_name}" / f"test_metadata_variance_threshold__{layer_type}__{layer_ind}__{author}.json",
                    f"After VarianceThreshold for {author} {layer_type} {layer_ind}"
                )

                svm = SVC()
                logger.info(f"Fitting SVM")
                svm.fit(features_data_variance_threshold.train_data, features_data_variance_threshold.train_labels)
                labels_predicted = svm.predict(features_data_variance_threshold.test_data)
                visualize_results_with_heatmap(labels_predicted, features_data_variance_threshold.test_labels, features_data_variance_threshold.test_metadata, f"SVM classification results for {author} {layer_type} {layer_ind} Variance Threshold features", Path(path_to_outputs) / f"{run_name}" / f"svm_classification_results__variance_threshold__{layer_type}__{layer_ind}__{author}___{args.include_prompted}.png")
                classification_report_svm = classification_report(features_data_variance_threshold.test_labels, labels_predicted, output_dict=True)
                logger.info(f"SVM classification report: {classification_report_svm}")
                classification_report_df = pd.DataFrame(classification_report_svm)
                classification_report_df.to_csv(Path(path_to_outputs) / f"{run_name}" / f"classification_report__svm__{layer_type}__{layer_ind}__{author}__variance_threshold.csv")

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
                    Path(path_to_outputs) / f"{run_name}" / f"train_metadata_sequential__{layer_type}__{layer_ind}__{author}.json",
                    f"After SequentialFeatureSelector for {author} {layer_type} {layer_ind}"
                )
                save_updated_metadata(
                    features_data_sequential_feature_selector.test_metadata,
                    Path(path_to_outputs) / f"{run_name}" / f"test_metadata_sequential__{layer_type}__{layer_ind}__{author}.json",
                    f"After SequentialFeatureSelector for {author} {layer_type} {layer_ind}"
                )

                # Apply LogisticRegression (works with sparse matrices)
                logistic_regression = LogisticRegression(max_iter=1000)  # Increased max_iter for sparse data
                logger.info(f"Fitting LogisticRegression on sparse data")
                logistic_regression.fit(features_data_sequential_feature_selector.train_data, features_data_sequential_feature_selector.train_labels)
                labels_predicted = logistic_regression.predict(features_data_sequential_feature_selector.test_data)
                
                """visualize_results_with_heatmap(
                    labels_predicted, 
                    features_data_sequential_feature_selector.test_labels, 
                    features_data_sequential_feature_selector.test_token_indices, 
                    f"LogisticRegression classification results for {author} {layer_type} {layer_ind} SequentialFeatureSelector features", 
                    Path(args.path_to_outputs) / f"{args.run_name}" / f"logreg_classification_results__seq_selector__{layer_type}__{layer_ind}__{author}___{args.include_prompted}.png"
                )"""
                
                classification_report_logreg = classification_report(features_data_sequential_feature_selector.test_labels, labels_predicted, output_dict=True)
                logger.info(f"LogisticRegression classification report: {classification_report_logreg}")
                classification_report_df = pd.DataFrame(classification_report_logreg)
                classification_report_df.to_csv(Path(path_to_outputs) / f"{run_name}" / f"classification_report__logreg__{layer_type}__{layer_ind}__{author}__sequential_feature_selector.csv")

                classification_results[layer_type][layer_ind][author]["sequential_feature_selector"]["LogisticRegression"]["precision__class_1"] = classification_report_logreg["1"]["precision"]
                classification_results[layer_type][layer_ind][author]["sequential_feature_selector"]["LogisticRegression"]["recall__class_1"] = classification_report_logreg["1"]["recall"]
                classification_results[layer_type][layer_ind][author]["sequential_feature_selector"]["LogisticRegression"]["f1__class_1"] = classification_report_logreg["1"]["f1-score"]
                
                # Apply SVM (works with sparse matrices)
                svm = SVC(kernel='linear')  # Linear kernel works better with sparse high-dimensional data
                logger.info(f"Fitting SVM on sparse data")
                svm.fit(features_data_sequential_feature_selector.train_data, features_data_sequential_feature_selector.train_labels)
                labels_predicted_svm = svm.predict(features_data_sequential_feature_selector.test_data)
                
                """visualize_results_with_heatmap(
                    labels_predicted_svm, 
                    features_data_sequential_feature_selector.test_labels, 
                    features_data_sequential_feature_selector.test_token_indices, 
                    f"SVM classification results for {author} {layer_type} {layer_ind} SequentialFeatureSelector features", 
                    Path(args.path_to_outputs) / f"{args.run_name}" / f"svm_classification_results__seq_selector__{layer_type}__{layer_ind}__{author}___{args.include_prompted}.png"
                )"""
                
                classification_report_svm = classification_report(features_data_sequential_feature_selector.test_labels, labels_predicted_svm, output_dict=True)
                logger.info(f"SVM classification report: {classification_report_svm}")
                classification_report_df = pd.DataFrame(classification_report_svm)
                classification_report_df.to_csv(Path(path_to_outputs) / f"{run_name}" / f"classification_report__svm__{layer_type}__{layer_ind}__{author}__sequential_feature_selector.csv")

                classification_results[layer_type][layer_ind][author]["sequential_feature_selector"]["SVM"]["precision__class_1"] = classification_report_svm["1"]["precision"]
                classification_results[layer_type][layer_ind][author]["sequential_feature_selector"]["SVM"]["recall__class_1"] = classification_report_svm["1"]["recall"]
                classification_results[layer_type][layer_ind][author]["sequential_feature_selector"]["SVM"]["f1__class_1"] = classification_report_svm["1"]["f1-score"]


                """random_forest = RandomForestClassifier()
                logger.info(f"Fitting RandomForest")
                cv_classification_random_forest_score = cross_validate(random_forest, author_data_numerical_sequential_feature_selector_filtered, labels_encoded_one_vs_all_variance_threshold_filtered, cv=cv, scoring=scoring, n_jobs=-1)
                logger.info(f"RandomForest classification scores: {cv_classification_random_forest_score}")
                for score in scoring:
                    classification_results[layer_type][layer_ind][author]["RandomForest"][score] = np.mean(cv_classification_random_forest_score[f"test_{score}"])"""

            # Save most important features
            with open(Path(path_to_outputs) / f"{run_name}" / f"most_important_features__{layer_type}__{layer_ind}.json", "w") as f:
                json.dump(most_important_features, f, indent=4)

                    

            # Save classification results
            with open(Path(path_to_outputs) / f"{run_name}" / f"classification_results.json", "w") as f:
                json.dump(classification_results, f, indent=4)
    


if __name__ == "__main__":
    main()
