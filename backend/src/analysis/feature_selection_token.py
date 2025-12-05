"""
Token-level feature selection analysis.

This module implements feature selection and classification on token-level data.
"""

import argparse
import json
import logging
from collections import defaultdict
from pathlib import Path
import pandas as pd
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.feature_selection import f_classif

from backend.src.utils.shared_utilities import ActivationFilenamesLoader
from backend.src.analysis.analysis_run_tracking import (
    get_data_and_output_paths,
    AnalysisRunTracker
)
from backend.src.analysis.feature_selection_base import (
    FeaturesData,
    VarianceThresholdFeatureSelection,
    SelectKBestFeatureSelection,
    SequentialFeatureSelectorWrapper
)
from backend.src.analysis.feature_selection_data_loader import (
    retrieve_and_combine_author_data_token_level
)

logger = logging.getLogger(__name__)


def parse_arguments():
    """Parse arguments for token-level feature selection analysis."""
    parser = argparse.ArgumentParser(description="Token-level Feature Selection Analysis")
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
        "--variance_threshold",
        type=float,
        default=0.01,
        help="Variance threshold for feature selection"
    )
    parser.add_argument(
        "--select_k_best",
        type=int,
        default=100,
        help="Number of features to select with SelectKBest"
    )
    parser.add_argument(
        "--sequential_k",
        type=int,
        default=None,
        help="Number of features to select with SequentialFeatureSelector (if None, skip this step)"
    )
    parser.add_argument(
        "--binary",
        action="store_true",
        default=True,
        help="Apply binary threshold (>1) to activations"
    )
    
    return parser.parse_args()


def save_updated_metadata(metadata: dict, save_path: Path, description: str = ""):
    """Save updated metadata after filtering operations."""
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


def run_token_level_analysis(
    author_filename_dict: dict,
    path_to_data: str,
    output_path: Path,
    layer_type: str,
    layer_ind: int,
    from_token: int = 0,
    variance_threshold: float = 0.01,
    select_k_best: int = 100,
    sequential_k: int = None,
    binary: bool = True
):
    """
    Run token-level feature selection and classification analysis.
    
    Args:
        author_filename_dict: Dictionary mapping author names to filenames
        path_to_data: Path to data directory
        output_path: Path to save results
        layer_type: Layer type
        layer_ind: Layer index
        from_token: Starting token position
        variance_threshold: Variance threshold for feature selection
        select_k_best: Number of features for SelectKBest
        sequential_k: Number of features for SequentialFeatureSelector (None to skip)
        binary: Whether to apply binary threshold
    """
    logger.info(f"Running token-level analysis for {layer_type} layer {layer_ind}")
    
    # Load data
    train_activations, test_activations, train_labels, test_labels, int_to_author, train_metadata, test_metadata = \
        retrieve_and_combine_author_data_token_level(
            author_filename_dict, path_to_data, binary=binary, from_token=from_token
        )
    
    classification_results = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(float)))))
    most_important_features = defaultdict(lambda: defaultdict(list))
    
    # Apply feature selection methods one vs all
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
        
        # Create initial FeaturesData
        features_data_initial = FeaturesData(
            train_data=train_activations,
            test_data=test_activations,
            train_labels=labels_encoded_one_vs_all,
            test_labels=labels_encoded_one_vs_all_test,
            train_metadata=train_metadata,
            test_metadata=test_metadata
        )
        
        # Step 1: Variance Threshold
        variance_selector = VarianceThresholdFeatureSelection(
            features_data_initial, threshold=variance_threshold
        )
        features_data_variance_threshold, most_important_features_variance_threshold = \
            variance_selector.run_feature_selection()
        
        most_important_features[author]["variance_threshold"] = most_important_features_variance_threshold
        
        # Save metadata after variance threshold
        save_updated_metadata(
            features_data_variance_threshold.train_metadata,
            output_path / f"train_metadata_variance_threshold__{layer_type}__{layer_ind}__{author}.json",
            f"After VarianceThreshold for {author} {layer_type} {layer_ind}"
        )
        save_updated_metadata(
            features_data_variance_threshold.test_metadata,
            output_path / f"test_metadata_variance_threshold__{layer_type}__{layer_ind}__{author}.json",
            f"After VarianceThreshold for {author} {layer_type} {layer_ind}"
        )
        
        # Classification with Variance Threshold features
        svm = SVC()
        logger.info(f"Fitting SVM on variance threshold features")
        svm.fit(features_data_variance_threshold.train_data, features_data_variance_threshold.train_labels)
        labels_predicted = svm.predict(features_data_variance_threshold.test_data)
        classification_report_svm = classification_report(
            features_data_variance_threshold.test_labels, labels_predicted, output_dict=True
        )
        logger.info(f"SVM classification report: {classification_report_svm}")
        
        classification_report_df = pd.DataFrame(classification_report_svm)
        classification_report_df.to_csv(
            output_path / f"classification_report__svm__{layer_type}__{layer_ind}__{author}__variance_threshold.csv"
        )
        
        classification_results[layer_type][layer_ind][author]["variance_threshold"]["SVM"]["precision__class_1"] = \
            classification_report_svm["1"]["precision"]
        classification_results[layer_type][layer_ind][author]["variance_threshold"]["SVM"]["recall__class_1"] = \
            classification_report_svm["1"]["recall"]
        classification_results[layer_type][layer_ind][author]["variance_threshold"]["SVM"]["f1__class_1"] = \
            classification_report_svm["1"]["f1-score"]
        
        # Step 2: SelectKBest
        select_k_best_selector = SelectKBestFeatureSelection(
            features_data_variance_threshold,
            most_important_features_variance_threshold,
            approach=f_classif,
            k=select_k_best
        )
        features_data_select_k_best, most_important_features_select_k_best = \
            select_k_best_selector.run_feature_selection()
        
        most_important_features[author]["select_k_best"] = most_important_features_select_k_best
        
        # Step 3: SequentialFeatureSelector (optional)
        if sequential_k is not None:
            sequential_selector = SequentialFeatureSelectorWrapper(
                features_data_select_k_best,
                most_important_features_select_k_best,
                approach=LogisticRegression(),
                k=sequential_k
            )
            features_data_sequential, most_important_features_sequential = \
                sequential_selector.run_feature_selection()
            
            most_important_features[author]["sequential_feature_selector"] = most_important_features_sequential
            
            # Save metadata after sequential feature selection
            save_updated_metadata(
                features_data_sequential.train_metadata,
                output_path / f"train_metadata_sequential__{layer_type}__{layer_ind}__{author}.json",
                f"After SequentialFeatureSelector for {author} {layer_type} {layer_ind}"
            )
            save_updated_metadata(
                features_data_sequential.test_metadata,
                output_path / f"test_metadata_sequential__{layer_type}__{layer_ind}__{author}.json",
                f"After SequentialFeatureSelector for {author} {layer_type} {layer_ind}"
            )
            
            # Classification with SequentialFeatureSelector features
            logistic_regression = LogisticRegression(max_iter=1000)
            logger.info(f"Fitting LogisticRegression on sequential features")
            logistic_regression.fit(features_data_sequential.train_data, features_data_sequential.train_labels)
            labels_predicted = logistic_regression.predict(features_data_sequential.test_data)
            
            classification_report_logreg = classification_report(
                features_data_sequential.test_labels, labels_predicted, output_dict=True
            )
            logger.info(f"LogisticRegression classification report: {classification_report_logreg}")
            classification_report_df = pd.DataFrame(classification_report_logreg)
            classification_report_df.to_csv(
                output_path / f"classification_report__logreg__{layer_type}__{layer_ind}__{author}__sequential_feature_selector.csv"
            )
            
            classification_results[layer_type][layer_ind][author]["sequential_feature_selector"]["LogisticRegression"]["precision__class_1"] = \
                classification_report_logreg["1"]["precision"]
            classification_results[layer_type][layer_ind][author]["sequential_feature_selector"]["LogisticRegression"]["recall__class_1"] = \
                classification_report_logreg["1"]["recall"]
            classification_results[layer_type][layer_ind][author]["sequential_feature_selector"]["LogisticRegression"]["f1__class_1"] = \
                classification_report_logreg["1"]["f1-score"]
            
            # SVM on sequential features
            svm = SVC(kernel='linear')
            logger.info(f"Fitting SVM on sequential features")
            svm.fit(features_data_sequential.train_data, features_data_sequential.train_labels)
            labels_predicted_svm = svm.predict(features_data_sequential.test_data)
            
            classification_report_svm = classification_report(
                features_data_sequential.test_labels, labels_predicted_svm, output_dict=True
            )
            logger.info(f"SVM classification report: {classification_report_svm}")
            classification_report_df = pd.DataFrame(classification_report_svm)
            classification_report_df.to_csv(
                output_path / f"classification_report__svm__{layer_type}__{layer_ind}__{author}__sequential_feature_selector.csv"
            )
            
            classification_results[layer_type][layer_ind][author]["sequential_feature_selector"]["SVM"]["precision__class_1"] = \
                classification_report_svm["1"]["precision"]
            classification_results[layer_type][layer_ind][author]["sequential_feature_selector"]["SVM"]["recall__class_1"] = \
                classification_report_svm["1"]["recall"]
            classification_results[layer_type][layer_ind][author]["sequential_feature_selector"]["SVM"]["f1__class_1"] = \
                classification_report_svm["1"]["f1-score"]
    
    # Save results
    logger.info(f"Most important features: {most_important_features}")
    with open(output_path / f"most_important_features__{layer_type}__{layer_ind}.json", "w") as f:
        json.dump(most_important_features, f, indent=4)
    
    logger.info(f"Classification results: {classification_results}")
    with open(output_path / f"classification_results.json", "w") as f:
        json.dump(classification_results, f, indent=4)


def main():
    """Main entry point for token-level feature selection analysis."""
    args = parse_arguments()
    
    # Configure logging
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Validate arguments
    if not args.run_id and not args.path_to_data:
        raise ValueError("Either --run_id or --path_to_data must be provided")
    
    # Get data and output paths
    data_path, output_path, activation_run_info = get_data_and_output_paths(
        run_id=args.run_id,
        data_path=args.path_to_data,
        analysis_type="feature_selection_token",
        run_name_override=None
    )
    
    # Register analysis run
    analysis_tracker = AnalysisRunTracker()
    activation_run_id = activation_run_info.get('id') if activation_run_info else None
    if activation_run_id:
        analysis_id = analysis_tracker.register_analysis(
            activation_run_id=activation_run_id,
            analysis_type="feature_selection_token",
            data_path=str(data_path),
            output_path=str(output_path)
        )
        logger.info(f"Registered analysis run with ID: {analysis_id}")
    
    # Load activation filenames
    activation_filenames_structured = ActivationFilenamesLoader(
        data_dir=Path(data_path),
        include_authors=args.include_authors,
        include_layer_types=args.include_layer_types,
        include_layer_inds=args.include_layer_inds,
        include_prompted=args.include_prompted
    ).get_structured_filenames()
    
    # Run analysis for each layer
    for layer_type, layer_ind_dict in activation_filenames_structured.items():
        for layer_ind, author_filename_dict in layer_ind_dict.items():
            run_token_level_analysis(
                author_filename_dict=author_filename_dict,
                path_to_data=str(data_path),
                output_path=output_path,
                layer_type=layer_type,
                layer_ind=layer_ind,
                from_token=args.from_token,
                variance_threshold=args.variance_threshold,
                select_k_best=args.select_k_best,
                sequential_k=args.sequential_k,
                binary=args.binary
            )
    
    logger.info("Token-level feature selection analysis completed!")


if __name__ == "__main__":
    main()

