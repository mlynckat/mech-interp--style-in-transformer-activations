"""
Iterative feature selection analysis.

This module implements iterative variance threshold feature selection with classification
at each step, and automatically visualizes results at the end.
"""

import os
import argparse
import json
import logging
from pathlib import Path
import numpy as np

from sklearn.feature_selection import VarianceThreshold

from backend.src.utils.shared_utilities import ActivationFilenamesLoader
from backend.src.analysis.analysis_run_tracking import (
    get_data_and_output_paths,
    AnalysisRunTracker
)
from backend.src.analysis.feature_selection_base import (
    FeaturesData,
    compute_adaptive_thresholds,
    run_classification_and_get_metrics
)
from backend.src.analysis.feature_selection_data_loader import (
    retrieve_and_combine_author_data_aggregated
)
from backend.src.analysis.visualize_iterative_results import main as visualize_main

logger = logging.getLogger(__name__)


class IterativeVarianceThresholdClassification:
    """
    Iteratively applies VarianceThreshold and classification until reaching min_features.
    At each step, runs classification and stores metrics.
    """
    
    def __init__(
        self,
        features_data: FeaturesData,
        min_features: int = 10,
        target_iterations: int = 100,
        use_adaptive_thresholds: bool = True,
        fixed_threshold: float = None
    ):
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
    
    def run_feature_selection(self) -> list:
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
            
            # Store results (without model object)
            result = {
                'iteration': iteration + 1,
                'n_features': int(current_train_data.shape[1]),
                'selected_features': [f"x{int(idx)}" for idx in remaining_features],
                'precision_class_1': metrics['precision'],
                'recall_class_1': metrics['recall'],
                'f1_score_class_1': metrics['f1_score']
            }
            self.results.append(result)
            
            logger.info(f"Metrics - Precision: {metrics['precision']:.4f}, "
                       f"Recall: {metrics['recall']:.4f}, F1: {metrics['f1_score']:.4f}")
            
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
                
                logger.info(f"Features after variance threshold: {new_train_data.shape[1]} "
                           f"(removed {current_train_data.shape[1] - new_train_data.shape[1]})")
                
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


def parse_arguments():
    """Parse arguments for iterative feature selection analysis."""
    parser = argparse.ArgumentParser(description="Iterative Feature Selection Analysis")
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
    parser.add_argument(
        "--fixed_threshold",
        type=float,
        default=0.01,
        help="Fixed threshold to use when adaptive thresholds are disabled"
    )
    
    return parser.parse_args()


def run_iterative_variance_threshold_analysis(
    author_filename_dict: dict,
    path_to_data: str,
    output_path: Path,
    layer_type: str,
    layer_ind: int,
    from_token: int = 0,
    min_features: int = 10,
    target_iterations: int = 100,
    use_adaptive_thresholds: bool = True,
    fixed_threshold: float = 0.01
) -> dict:
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
        fixed_threshold: Fixed threshold when adaptive is disabled
    
    Returns:
        Dictionary with results for all authors
    """
    logger.info(f"Starting iterative variance threshold analysis for {layer_type} layer {layer_ind}")

    combined_results_path = output_path / f"iterative_variance_threshold_combined__{layer_type}__{layer_ind}.json"
    if os.path.exists(combined_results_path):
        logger.info(f"Combined results already exist for {layer_type} layer {layer_ind}. Loading...")
        with open(combined_results_path, 'r') as f:
            all_results = json.load(f)
        return all_results
    
    # Load data
    train_activations, test_activations, train_labels, test_labels, int_to_author, train_doc_ids, test_doc_ids = \
        retrieve_and_combine_author_data_aggregated(
            author_filename_dict, path_to_data, from_token=from_token
        )
    
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
            use_adaptive_thresholds=use_adaptive_thresholds,
            fixed_threshold=fixed_threshold if not use_adaptive_thresholds else None
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
    
    with open(combined_results_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    logger.info(f"Saved combined results to {combined_results_path}")
    
    return all_results


def main():
    """Main entry point for iterative feature selection analysis."""
    args = parse_arguments()
    
    # Handle adaptive thresholds flag
    use_adaptive_thresholds = args.use_adaptive_thresholds and not args.no_adaptive_thresholds
    
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
        analysis_type="feature_selection_iterative",
        run_name_override=None
    )
    
    # Register analysis run
    analysis_tracker = AnalysisRunTracker()
    activation_run_id = activation_run_info.get('id') if activation_run_info else None
    if activation_run_id:
        analysis_id = analysis_tracker.register_analysis(
            activation_run_id=activation_run_id,
            analysis_type="feature_selection_iterative",
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


            run_iterative_variance_threshold_analysis(
                author_filename_dict=author_filename_dict,
                path_to_data=str(data_path),
                output_path=output_path,
                layer_type=layer_type,
                layer_ind=layer_ind,
                from_token=args.from_token,
                min_features=args.min_features,
                target_iterations=args.target_iterations,
                use_adaptive_thresholds=use_adaptive_thresholds,
                fixed_threshold=args.fixed_threshold
            )
    
    # Automatically run visualization after analysis
    logger.info("\n" + "="*70)
    logger.info("Starting visualization generation...")
    logger.info("="*70)
    
    # Create visualization arguments dict
    vis_args_dict = {
        'data_dir': str(output_path),
        'output_suffix': ''
    }
    
    # Run visualization
    try:
        visualize_main(vis_args_dict)
        logger.info("Visualization generation completed!")
    except Exception as e:
        logger.error(f"Error during visualization: {e}")
        logger.warning("Analysis completed but visualization failed. You can run visualization manually later.")
    
    logger.info("Iterative feature selection analysis completed!")


if __name__ == "__main__":
    main()

