"""
SHAP analysis utilities for feature selection.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
from pathlib import Path
import logging

from backend.src.analysis.feature_selection_base import FeaturesData

logger = logging.getLogger(__name__)


def run_shap_analysis(
    model,
    X_train: np.ndarray,
    X_test: np.ndarray,
    feature_names: list,
    output_path: Path,
    author: str,
    layer_type: str,
    layer_ind: int,
    model_name: str = "LogisticRegression"
) -> tuple[pd.DataFrame, np.ndarray]:
    """
    Run SHAP analysis on trained model and save results.
    
    Args:
        model: Trained model
        X_train: Training features
        X_test: Test features
        feature_names: List of feature names
        output_path: Path to save results
        author: Author name
        layer_type: Layer type
        layer_ind: Layer index
        model_name: Name of the model
    
    Returns:
        Tuple of (importance_df, shap_values) or (None, None) on error
    """
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
        averaged_shap_values = np.mean(shap_values, axis=0)
        
        # Create feature importance DataFrame
        importance_df = pd.DataFrame({
            'feature_name': feature_names,
            'importance': feature_importance,
            'averaged_shap_values': averaged_shap_values,
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


def get_features_data_shap(
    features_data,
    feature_names: list,
    top_features: int = 50
):
    """Get FeaturesData object with top features based on SHAP importance."""
    feature_indices = [int(name.replace("x", "")) for name in feature_names[:top_features]]
    return FeaturesData(
        train_data=features_data.train_data[:, feature_indices],
        test_data=features_data.test_data[:, feature_indices],
        train_labels=features_data.train_labels,
        test_labels=features_data.test_labels,
        train_doc_ids=features_data.train_doc_ids,
        test_doc_ids=features_data.test_doc_ids,
        train_metadata=features_data.train_metadata,
        test_metadata=features_data.test_metadata
    )

