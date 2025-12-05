"""
Aggregated (document-level) feature selection analysis with SHAP.

This module implements feature selection and classification on per-document-aggregated data
with SHAP analysis for feature importance.
"""

import argparse
import json
import logging
from collections import defaultdict
from pathlib import Path
import pandas as pd
import numpy as np
import pickle
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

from backend.src.utils.shared_utilities import ActivationFilenamesLoader
from backend.src.analysis.analysis_run_tracking import (
    get_data_and_output_paths
)
from backend.src.analysis.feature_selection_base import (
    FeaturesData,
    VarianceThresholdFeatureSelection
)
from backend.src.analysis.feature_selection_data_loader import (
    retrieve_and_combine_author_data_aggregated
)
from backend.src.analysis.feature_selection_shap import run_shap_analysis

logger = logging.getLogger(__name__)


def parse_arguments():
    """Parse arguments for aggregated feature selection analysis."""
    parser = argparse.ArgumentParser(description="Aggregated Feature Selection Analysis with SHAP")
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
        default=["Sam Levine", "Paige Lavender", "Lee Moran", "Amanda Terkel"],
        help="The authors to include in the analysis"
    )
    parser.add_argument(
        "--include_layer_types",
        type=str,
        nargs="+",
        default=["res"],
        choices=["res", "mlp", "att"],
        help="The layer types to include in the analysis"
    )
    parser.add_argument(
        "--include_layer_inds",
        type=int,
        nargs="+",
        default=[15],
        help="The layers to include in the analysis"
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
        default=20,
        help="Number of features to select with SequentialFeatureSelector (if None, skip this step)"
    )


    parser.add_argument(
        "--include_prompted",
        type=str,
        default="baseline",
        choices=["prompted", "baseline"],
        help="The prompted to include in the analysis"
    )
    return parser.parse_args()


def run_aggregated_analysis(
    author_filename_dict: dict,
    path_to_data: str,
    output_path: Path,
    layer_type: str,
    layer_ind: int,
    from_token: int = 0,
    variance_threshold: float = 0.01,
    select_k_best: int = 100,
    sequential_k: int = 25,
    classification_results: dict = None
):
    """
    Run aggregated feature selection and classification analysis with SHAP.
    
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
        classification_results: Dictionary to accumulate results (will be created if None)
    
    Returns:
        classification_results: Updated nested dictionary with accumulated results.
    """
    logger.info(f"Running aggregated analysis for {layer_type} layer {layer_ind}")
    print(f"[prepare_features_for_steering] Running aggregated analysis for {layer_type=} {layer_ind=}")
    
    # Load data
    train_activations, test_activations, train_labels, test_labels, int_to_author, train_doc_ids, test_doc_ids = \
        retrieve_and_combine_author_data_aggregated(
            author_filename_dict, path_to_data, from_token=from_token
        )
    print(
        "[prepare_features_for_steering] Loaded activations with "
        f"{train_activations.shape=} {test_activations.shape=} "
        f"{train_labels.shape=} {test_labels.shape=}"
    )
    
    if classification_results is None:
        classification_results = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(float))))))
    most_important_features = defaultdict(lambda: defaultdict(list))
    
    # Apply feature selection methods one vs all
    for author_ind, author in int_to_author.items():
        logger.info(f"Applying feature selection for {author}")
        print(f"[prepare_features_for_steering] Applying feature selection for author='{author}' ({author_ind=})")
        
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
            train_doc_ids=train_doc_ids,
            test_doc_ids=test_doc_ids
        )
        
        # Step 1: Variance Threshold
        variance_selector = VarianceThresholdFeatureSelection(
            features_data_initial, threshold=variance_threshold
        )
        features_data_variance_threshold, most_important_features_variance_threshold = \
            variance_selector.run_feature_selection()
        
        most_important_features[author]["variance_threshold"] = most_important_features_variance_threshold
        
        # Classification with Variance Threshold features
        logistic_regression = LogisticRegression(max_iter=2000)
        logger.info(f"Fitting LogisticRegression on variance threshold filtered data")
        print("[prepare_features_for_steering] Fitting LogisticRegression with variance threshold features")
        logistic_regression.fit(features_data_variance_threshold.train_data, features_data_variance_threshold.train_labels)
        labels_predicted = logistic_regression.predict(features_data_variance_threshold.test_data)
        
        classification_report_logreg = classification_report(
            features_data_variance_threshold.test_labels, labels_predicted, output_dict=True
        )
        logger.info(f"LogisticRegression classification report: {classification_report_logreg}")
        print(
            "[prepare_features_for_steering] LogisticRegression (variance threshold) "
            f"precision={classification_report_logreg['1']['precision']:.4f} "
            f"recall={classification_report_logreg['1']['recall']:.4f} "
            f"f1={classification_report_logreg['1']['f1-score']:.4f}"
        )
        classification_report_df = pd.DataFrame(classification_report_logreg)
        classification_report_df.to_csv(
            output_path / f"classification_report__logreg__{layer_type}__{layer_ind}__{author}__variance_threshold.csv"
        )
        
        classification_results[layer_type][layer_ind][author]["variance_threshold"]["LogisticRegression"]["precision__class_1"] = \
            classification_report_logreg["1"]["precision"]
        classification_results[layer_type][layer_ind][author]["variance_threshold"]["LogisticRegression"]["recall__class_1"] = \
            classification_report_logreg["1"]["recall"]
        classification_results[layer_type][layer_ind][author]["variance_threshold"]["LogisticRegression"]["f1__class_1"] = \
            classification_report_logreg["1"]["f1-score"]
        
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
            most_important_features[author]["averaged_shap_values"] = shap_importance_df['averaged_shap_values'].tolist()
        


        for n_features in range(10, 25, 2):
            features_to_keep  = most_important_features[author]["shap_variance_threshold"][:n_features]
            features_to_keep_int = [int(feature.replace("x", "")) for feature in features_to_keep]
            print(f"[prepare_features_for_steering] Features to keep: {features_to_keep_int}")
            features_data_shap_best_k = FeaturesData(
                train_data=train_activations[:, np.array(features_to_keep_int)],
                test_data=test_activations[:, np.array(features_to_keep_int)],
                train_labels=labels_encoded_one_vs_all,
                test_labels=labels_encoded_one_vs_all_test,
                train_doc_ids=train_doc_ids,
                test_doc_ids=test_doc_ids
            )

            logistic_regression = LogisticRegression(max_iter=2000)
            logger.info(f"Fitting LogisticRegression on best k SHAP features")
            print(f"[prepare_features_for_steering] Fitting LogisticRegression on top-{n_features} SHAP features")
            logistic_regression.fit(features_data_shap_best_k.train_data, features_data_shap_best_k.train_labels)
            labels_predicted = logistic_regression.predict(features_data_shap_best_k.test_data)
            
            classification_report_logreg = classification_report(
                features_data_shap_best_k.test_labels, labels_predicted, output_dict=True
            )
            logger.info(f"LogisticRegression classification report: {classification_report_logreg}")
            print(
                "[prepare_features_for_steering] LogisticRegression (SHAP best k) "
                f"n_features={n_features} "
                f"precision={classification_report_logreg['1']['precision']:.4f} "
                f"recall={classification_report_logreg['1']['recall']:.4f} "
                f"f1={classification_report_logreg['1']['f1-score']:.4f}"
            )
            classification_report_df = pd.DataFrame(classification_report_logreg)
            classification_report_df.to_csv(
                output_path / f"classification_report__logreg__{layer_type}__{layer_ind}__{author}__shap_best_k__n_features_{n_features}.csv"
            )
            
            classification_results[layer_type][layer_ind][author][f"shap_best_{n_features}"]["LogisticRegression"]["precision__class_1"] = \
                classification_report_logreg["1"]["precision"]
            classification_results[layer_type][layer_ind][author][f"shap_best_{n_features}"]["LogisticRegression"]["recall__class_1"] = \
                classification_report_logreg["1"]["recall"]
            classification_results[layer_type][layer_ind][author][f"shap_best_{n_features}"]["LogisticRegression"]["f1__class_1"] = \
                classification_report_logreg["1"]["f1-score"]

            # Save the logreg model
            with open(output_path / f"logreg_model__{layer_type}__{layer_ind}__{author.replace(' ', '_')}__shap_best_{n_features}.pkl", "wb") as f:
                pickle.dump(logistic_regression, f)

            joblib.dump(logistic_regression, str(output_path / f"logreg_model__{layer_type}__{layer_ind}__{author.replace(' ', '_')}__shap_best_{n_features}.joblib")) 
            

    
    # Save results
    #logger.info(f"Most important features: {most_important_features}")
    with open(output_path / f"most_important_features__{layer_type}__{layer_ind}.json", "w") as f:
        json.dump(most_important_features, f, indent=4)
    
    logger.info(f"Classification results: {classification_results}")
    return classification_results


def create_classification_results_dict():
    """Create the nested defaultdict structure for classification results."""
    return defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(float))))))


def nested_defaultdict_to_dict(d):
    """Recursively convert nested defaultdicts to plain dicts."""
    if isinstance(d, defaultdict):
        d = {k: nested_defaultdict_to_dict(v) for k, v in d.items()}
    elif isinstance(d, dict):
        d = {k: nested_defaultdict_to_dict(v) for k, v in d.items()}
    return d


def main():
    """Main entry point for aggregated feature selection analysis."""
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
    data_path, output_path, _ = get_data_and_output_paths(
        run_id=args.run_id,
        data_path=args.path_to_data,
        analysis_type="feature_selection_aggregated",
        run_name_override="prepare_features_for_steering"
    )
    
    # Load activation filenames
    activation_filenames_structured = ActivationFilenamesLoader(
        data_dir=Path(data_path),
        include_authors=args.include_authors,
        include_layer_types=args.include_layer_types,
        include_layer_inds=args.include_layer_inds,
        include_prompted=args.include_prompted
    ).get_structured_filenames()
    
    # Prepare shared classification results container
    classification_results = create_classification_results_dict()

    # Run analysis for each layer
    for layer_type, layer_ind_dict in activation_filenames_structured.items():
        for layer_ind, author_filename_dict in layer_ind_dict.items():
            classification_results = run_aggregated_analysis(
                author_filename_dict=author_filename_dict,
                path_to_data=str(data_path),
                output_path=output_path,
                layer_type=layer_type,
                layer_ind=layer_ind,
                from_token=args.from_token,
                variance_threshold=args.variance_threshold,
                select_k_best=args.select_k_best,
                sequential_k=args.sequential_k,
                classification_results=classification_results
            )

    # Persist aggregated classification results once all layers are processed
    with open(output_path / "classification_results.json", "w") as f:
        json.dump(classification_results, f, indent=4)
    
    logger.info("Aggregated feature selection analysis completed!")


if __name__ == "__main__":
    main()

