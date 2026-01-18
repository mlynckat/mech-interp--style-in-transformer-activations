"""
Classification pipeline for generated texts.

This module provides functionality to classify generated texts using various
classification models and data transformers. It supports easy configuration
of models, transformers, and input paths.

Complete Pipeline Integration:
    The classification pipeline seamlessly connects:
    1. Trained classifiers (from main.py via ClassifierTrainingConfig)
    2. Generated texts (from steered_text_generation.py via RunConfig)
    
    Example - Full pipeline:
        # Step 1: Train classifiers (run main.py)
        python main.py --run-name modernbert_10epochs --data-subset all
        
        # Step 2: Generate texts (run steered_text_generation.py)
        # This creates a run with RunConfig
        
        # Step 3: Classify generated texts
        from backend.src.steering.classification.classify_generated import run_classification_from_configs
        
        results = run_classification_from_configs(
            generation_run_dir="data/steering/tests/my_generation_run",
            classifier_run_name="modernbert_10epochs"
        )
    
    Example - Using RunConfig only (with default classifiers):
        from backend.src.steering.run_config import RunConfig
        
        run_config = RunConfig.from_run_dir("data/steering/tests/my_run")
        classification_config = ClassificationConfig.from_run_config(run_config)
        run_classification_pipeline(classification_config)

Output Schema:
{
  "date": "2025-12-17T00:00:00Z",
  "run_id": "uuid",
  "config": {
    "data_transformer": "...",
    "classifier_name": "...",
    "steering_method": "...",
    "n_shap_features": 64
  },
  "results": {
    "per_target_classifier": {
      "author_A": {
        "metrics": {
          "NSS": float,  # Normalized Style Sensitivity (Cohen's d effect size)
          "CII": float   # Content Invariance Index (content leakage measure)
        },
        "per_desired_style_author": {
          "author_A": {
            "per_original_author": {
              "author_A": { "sum_predictions": int, "avg_probs": float, "n_samples": int },
              "author_B": { "sum_predictions": int, "avg_probs": float, "n_samples": int }
            }
          }
        }
      }
    }
  }
}

Metric Definitions:
-------------------
NSS (Normalized Style Sensitivity):
    Effect size measuring how well the classifier discriminates based on steered author.
    NSS = Δp / s_pooled, where Δp = mean(p|steered→target) - mean(p|steered→others)
    Interpretation: 0 = no style sensitivity, >0 = classifier responds to steering

CII (Content Invariance Index):
    Measures dependence of classifier outputs on original author (content leakage).
    CII = σ_μ / σ_p, where σ_μ = std of mean probabilities per original author
    Interpretation: close to 0 = content invariant (good), larger = content leakage (bad)
"""

from pathlib import Path
from typing import Tuple, Dict, Any, List, Optional, TYPE_CHECKING
from dataclasses import dataclass, field
from datetime import datetime
import uuid

import json
import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import seaborn as sns

from backend.src.steering.classification.data_transformers import (
    TFIDFTransformer,
    SentenceEmbeddingTransformer,
    ModernBertTransformer,
)
from backend.src.steering.classification.classification_models import (
    LogisticRegressionModel,
    ModernBertClassifierModel,
    RandomForestModel,
    SGDClassifierModel,
)
from backend.src.steering.classification.pipeline import MLPipeline

# Import RunConfig for type hints and integration
if TYPE_CHECKING:
    from backend.src.steering.run_config import RunConfig, ClassifierTrainingConfig


# ============================================================================
# CONFIGURATION
# ============================================================================

DEFAULT_AUTHORS = ["Sam Levine", "Paige Lavender", "Lee Moran", "Amanda Terkel"]
DEFAULT_CLASSIFIER_BASE_DIR = Path("backend/src/steering/classification/models")
DEFAULT_GENERATION_BASE_DIR = Path("data/steering/tests")


@dataclass
class ClassificationConfig:
    """
    Configuration class for classification pipeline.
    
    Can be created:
    1. Manually with explicit paths
    2. From a RunConfig (generation run)
    3. From both RunConfig and ClassifierTrainingConfig (full pipeline)
    
    Example - Full pipeline integration:
        config = ClassificationConfig.from_configs(
            generation_run_dir_name="my_run",
            classifier_run_name="modernbert_10epochs"
        )
        run_classification_pipeline(config)
    """
    
    # Model and transformer classes
    classification_model_class: type = ModernBertClassifierModel
    data_transformer_class: type = ModernBertTransformer
    
    # Paths
    path_to_models: Path = Path("backend/src/steering/classification/models")
    path_to_generated_texts_template: str = "data/steering/tests/generated_texts__sae_baseline__{}.json"
    output_dir: Path = Path("data/steering/tests/classification_results")
    
    # Author list
    author_list: List[str] = None
    
    # Steering configuration (for metadata/tracking)
    steering_method: Optional[str] = None  # e.g., "heuristic", "projected_gradient", None for baseline
    n_shap_features: Optional[int] = None
    
    # Run identification
    run_name: Optional[str] = None  # Generation run name
    classifier_run_name: Optional[str] = None  # Classifier training run name
    
    def __post_init__(self):
        """Initialize default author list if not provided."""
        if self.author_list is None:
            self.author_list = DEFAULT_AUTHORS.copy()
        
        # Ensure paths are Path objects
        self.path_to_models = Path(self.path_to_models)
        self.output_dir = Path(self.output_dir)
    
    @classmethod
    def from_run_config(
        cls,
        run_config: "RunConfig",
        classification_model_class: type = ModernBertClassifierModel,
        data_transformer_class: type = ModernBertTransformer,
        path_to_models: Optional[Path] = None
    ) -> "ClassificationConfig":
        """
        Create a ClassificationConfig from a RunConfig (generation run).
        
        Uses default classifier path unless specified.
        
        Args:
            run_config: The RunConfig from the generation run
            classification_model_class: Classification model to use
            data_transformer_class: Data transformer to use
            path_to_models: Path to classifier models (optional)
            
        Returns:
            ClassificationConfig configured to evaluate the generation run
        """
        template = run_config.get_generated_texts_template()
        base_output_dir = run_config.get_classification_dir()
        
        # Create subfolder based on model and transformer class names
        classifier_subfolder = f"{classification_model_class.__name__}__{data_transformer_class.__name__}"
        output_dir = base_output_dir / classifier_subfolder
        
        return cls(
            classification_model_class=classification_model_class,
            data_transformer_class=data_transformer_class,
            path_to_models=path_to_models or DEFAULT_CLASSIFIER_BASE_DIR,
            path_to_generated_texts_template=template,
            output_dir=output_dir,
            author_list=run_config.author_list.copy(),
            steering_method=run_config.steering_method,
            n_shap_features=run_config.n_shap_features,
            run_name=run_config.run_name
        )
    
    @classmethod
    def from_classifier_training_config(
        cls,
        classifier_config: "ClassifierTrainingConfig",
        path_to_generated_texts_template: str,
        output_dir: Path,
        author_list: Optional[List[str]] = None,
        steering_method: Optional[str] = None,
        n_shap_features: Optional[int] = None,
        run_name: Optional[str] = None
    ) -> "ClassificationConfig":
        """
        Create a ClassificationConfig from a ClassifierTrainingConfig.
        
        Automatically resolves the transformer and model classes from the training config.
        
        Args:
            classifier_config: The ClassifierTrainingConfig from training
            path_to_generated_texts_template: Template for generated texts paths
            output_dir: Directory for classification results
            author_list: List of authors (uses training config's list if None)
            steering_method: Steering method name (for metadata)
            n_shap_features: Number of SHAP features (for metadata)
            run_name: Generation run name (for metadata)
            
        Returns:
            ClassificationConfig configured to use the trained models
        """
        # Map names to classes
        transformer_map = {
            "tfidf": TFIDFTransformer,
            "sentence_embedding": SentenceEmbeddingTransformer,
            "modern_bert": ModernBertTransformer,
        }
        model_map = {
            "logistic_regression": LogisticRegressionModel,
            "random_forest": RandomForestModel,
            "sgd": SGDClassifierModel,
            "modern_bert": ModernBertClassifierModel,
        }
        
        transformer_class = transformer_map.get(classifier_config.transformer_name)
        model_class = model_map.get(classifier_config.model_name)
        
        if transformer_class is None:
            raise ValueError(f"Unknown transformer: {classifier_config.transformer_name}")
        if model_class is None:
            raise ValueError(f"Unknown model: {classifier_config.model_name}")
        
        return cls(
            classification_model_class=model_class,
            data_transformer_class=transformer_class,
            path_to_models=classifier_config.models_dir,
            path_to_generated_texts_template=path_to_generated_texts_template,
            output_dir=output_dir,
            author_list=author_list or classifier_config.author_list.copy(),
            steering_method=steering_method,
            n_shap_features=n_shap_features,
            run_name=run_name,
            classifier_run_name=classifier_config.run_name
        )
    
    @classmethod
    def from_configs(
        cls,
        generation_run_dir_name: str,
        classifier_run_name: str,
        classifier_base_dir: Optional[Path] = None
    ) -> "ClassificationConfig":
        """
        Create a ClassificationConfig from both generation and classifier configs.
        
        This is the recommended method for full pipeline integration.
        
        Args:
            generation_run_dir_name: Name of the generation run directory
            classifier_run_name: Name of the classifier training run
            classifier_base_dir: Base directory for classifiers (uses default if None)
            
        Returns:
            ClassificationConfig combining both configs
            
        Example:
            config = ClassificationConfig.from_configs(
                generation_run_dir_name="my_generation_run",
                classifier_run_name="modernbert_10epochs"
            )
            run_classification_pipeline(config)
        """
        from backend.src.steering.run_config import RunConfig, ClassifierTrainingConfig
        
        # Load generation run config
        run_config = RunConfig.from_run_dir(DEFAULT_GENERATION_BASE_DIR / generation_run_dir_name)
        
        # Load classifier training config
        base_dir = classifier_base_dir or DEFAULT_CLASSIFIER_BASE_DIR
        classifier_config = ClassifierTrainingConfig.from_run_name(
            classifier_run_name, 
            base_dir=base_dir
        )
        
        # Create output directory with classifier run name subfolder
        base_output_dir = run_config.get_classification_dir()
        output_dir = base_output_dir / classifier_run_name
        
        # Create combined config
        return cls.from_classifier_training_config(
            classifier_config=classifier_config,
            path_to_generated_texts_template=run_config.get_generated_texts_template(),
            output_dir=output_dir,
            author_list=run_config.author_list.copy(),
            steering_method=run_config.steering_method,
            n_shap_features=run_config.n_shap_features,
            run_name=run_config.run_name
        )
    
    @classmethod
    def from_run_dir(
        cls,
        run_dir: Path,
        classification_model_class: type = ModernBertClassifierModel,
        data_transformer_class: type = ModernBertTransformer,
        path_to_models: Optional[Path] = None
    ) -> "ClassificationConfig":
        """
        Create a ClassificationConfig by loading a RunConfig from a run directory.
        
        Args:
            run_dir: Path to the run directory containing run_config.json
            classification_model_class: Classification model to use
            data_transformer_class: Data transformer to use
            path_to_models: Path to classifier models (optional)
            
        Returns:
            ClassificationConfig configured to evaluate the generation run
        """
        from backend.src.steering.run_config import RunConfig
        
        run_config = RunConfig.from_run_dir(run_dir)
        return cls.from_run_config(
            run_config,
            classification_model_class=classification_model_class,
            data_transformer_class=data_transformer_class,
            path_to_models=path_to_models
        )


# ============================================================================
# DATA LOADING
# ============================================================================

def read_generated_texts(
    path_to_generated_texts: Path, 
    desired_style_author: str
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    Read generated texts from JSON file and split features and target.
    
    Args:
        path_to_generated_texts: Path to JSON file containing generated texts
        desired_style_author: Author whose style the texts were steered towards
        
    Returns:
        Tuple of (X, desired_style_authors, original_authors) as pandas Series
    """
    with open(path_to_generated_texts, "r", encoding="utf-8") as f:
        generated_texts = json.load(f)

    X = []
    desired_style_authors = []
    original_authors = []
    
    for doc in generated_texts:
        X.append(doc["generated_text"])
        desired_style_authors.append(desired_style_author)
        original_authors.append(doc["author"])

    return pd.Series(X), pd.Series(desired_style_authors), pd.Series(original_authors)


def aggregate_generated_texts(
    path_template: str,
    author_list: List[str]
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    Aggregate generated texts from multiple authors.
    
    Args:
        path_template: Template string for file paths with {} placeholder for author
        author_list: List of authors to process
        
    Returns:
        Tuple of (X_all, desired_style_authors_all, original_authors_all) as pandas Series
        - X_all: The generated texts
        - desired_style_authors_all: Which author's style the texts were steered towards
        - original_authors_all: The original author whose article generated the prompt
    """
    X_all = []
    desired_style_authors_all = []
    original_authors_all = []

    for desired_style_author in author_list:
        path_to_generated_texts = path_template.format(desired_style_author)
        
        # Check if file exists
        if not Path(path_to_generated_texts).exists():
            print(f"Warning: File not found, skipping: {path_to_generated_texts}")
            continue
        
        X, desired_style_authors, original_authors = read_generated_texts(
            Path(path_to_generated_texts), 
            desired_style_author
        )
        
        X_all.extend(X.tolist())
        desired_style_authors_all.extend(desired_style_authors.tolist())
        original_authors_all.extend(original_authors.tolist())

    return pd.Series(X_all), pd.Series(desired_style_authors_all), pd.Series(original_authors_all)


def load_texts_from_run_config(
    run_config: "RunConfig"
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    Load generated texts using a RunConfig.
    
    Convenience function that extracts the template from RunConfig.
    
    Args:
        run_config: RunConfig from the generation run
        
    Returns:
        Tuple of (X_all, desired_style_authors_all, original_authors_all) as pandas Series
    """
    template = run_config.get_generated_texts_template()
    return aggregate_generated_texts(template, run_config.author_list)


# ============================================================================
# FILENAME PARSING
# ============================================================================

def extract_data_type_from_path(path_template: str, author: str) -> str:
    """
    Extract data type from file path template.
    
    Pattern: generated_texts__{type}__{author}.json or generated_texts__{type}.json
    Examples:
        - generated_texts__sae_baseline__Sam Levine.json -> "sae_baseline"
        - generated_texts__steered__heuristic__Sam Levine.json -> "steered__heuristic"
        - generated_texts__baseline.json -> "baseline"
    
    Args:
        path_template: Template string for file paths
        author: Author name to substitute in template
        
    Returns:
        Extracted data type string
    """
    file_path = path_template.format(author)
    file_name = Path(file_path).stem  # Get filename without extension
    
    # Extract type: remove "generated_texts__" prefix and author suffix if present
    parts = file_name.split("__")
    
    if len(parts) >= 2:
        # If there are more than 2 parts, the last one is the author, so take parts[1:-1]
        # If there are exactly 2 parts, take parts[1] (just the type, no author)
        if len(parts) > 2:
            # Has author: generated_texts__type__author -> take parts[1:-1]
            type_parts = parts[1:-1]
        else:
            # No author: generated_texts__type -> take parts[1]
            type_parts = parts[1:]
        return "__".join(type_parts)  # Join in case of "steered__heuristic"
    else:
        return "unknown"


# ============================================================================
# METRICS CALCULATION
# ============================================================================

def calculate_metrics_for_group(
    y_pred: pd.Series,
    y_proba: pd.Series
) -> Dict[str, Any]:
    """
    Calculate metrics for a group of predictions.
    
    Args:
        y_pred: Binary predictions (0 or 1)
        y_proba: Probability of class 1
        
    Returns:
        Dictionary with sum_predictions and avg_probs
    """
    return {
        "sum_predictions": int(y_pred.sum()),
        "avg_probs": float(y_proba.mean()) if len(y_proba) > 0 else 0.0,
        "n_samples": int(len(y_pred))
    }


def compute_structured_metrics(
    X_all: pd.Series,
    desired_style_authors_all: pd.Series,
    original_authors_all: pd.Series,
    author_list: List[str],
    config: ClassificationConfig,
    file_suffix: str
) -> Dict[str, Any]:
    """
    Compute predictions and metrics in the structured format:
    per_target_classifier -> metrics + per_desired_style_author -> per_original_author
    
    Args:
        X_all: All generated texts
        desired_style_authors_all: Which author's style the texts were steered towards
        original_authors_all: Original author whose article generated the prompt
        author_list: List of authors
        config: Classification configuration
        file_suffix: Suffix for heatmap files
        
    Returns:
        Dictionary with structured metrics including NSS and CII per classifier
    """
    per_target_classifier = {}
    
    for target_classifier_author in author_list:
        print(f"\n{'='*60}")
        print(f"Processing classifier for: {target_classifier_author}")
        print(f"{'='*60}")
        
        # Initialize transformer and model for this classifier
        data_transformer = config.data_transformer_class(
            path_to_models=config.path_to_models, 
            author=target_classifier_author
        )
        classification_model = config.classification_model_class(
            path_to_models=config.path_to_models, 
            author=target_classifier_author, 
            data_transformer=data_transformer
        )
        
        # Get predictions and probabilities for ALL texts
        pipeline = MLPipeline(data_transformer, classification_model)
        y_pred_all = pipeline.predict_samples(X_all)
        y_proba_all = pipeline.predict_proba_samples(X_all)
        
        print(f"Total predictions: {len(y_pred_all)}, Sum of 1s: {y_pred_all.sum()}")
        
        # Calculate NSS (Normalized Style Sensitivity)
        nss = calculate_nss(
            y_proba_all,
            desired_style_authors_all,
            target_classifier_author
        )
        print(f"NSS (Normalized Style Sensitivity): {nss:.4f}")
        
        # Calculate CII (Content Invariance Index)
        cii = calculate_cii(
            y_proba_all,
            original_authors_all,
            author_list
        )
        print(f"CII (Content Invariance Index): {cii:.4f}")
        
        # Build per_desired_style_author structure
        per_desired_style_author = {}
        
        for desired_style_author in author_list:
            # Filter to texts steered towards this author's style
            desired_mask = desired_style_authors_all == desired_style_author
            
            per_original_author = {}
            
            for original_author in author_list:
                # Further filter to texts originally from this author
                combined_mask = desired_mask & (original_authors_all == original_author)
                
                if combined_mask.sum() == 0:
                    continue
                
                # Get predictions and probabilities for this subset
                y_pred_subset = y_pred_all[combined_mask]
                y_proba_subset = y_proba_all[combined_mask]
                
                per_original_author[original_author] = calculate_metrics_for_group(
                    y_pred_subset,
                    y_proba_subset
                )
            
            per_desired_style_author[desired_style_author] = {
                "per_original_author": per_original_author
            }
        
        # Store metrics and per_desired_style_author in the structure
        per_target_classifier[target_classifier_author] = {
            "metrics": {
                "NSS": nss,
                "CII": cii
            },
            "per_desired_style_author": per_desired_style_author
        }
        
        # Create heatmap for this classifier
        heatmap_data_per_author = prepare_heatmap_data_per_author(
            desired_style_authors_all, 
            y_pred_all,
            original_authors_all
        )
        create_heatmap_per_author(heatmap_data_per_author, target_classifier_author, config, file_suffix)
        
        # Create boxplot for this classifier
        create_boxplot_per_classifier(
            y_proba_all,
            desired_style_authors_all,
            original_authors_all,
            target_classifier_author,
            author_list,
            config,
            file_suffix
        )
    
    return per_target_classifier


def calculate_classification_metrics(
    y_true: pd.Series,
    y_pred: pd.Series,
    initial_authors: pd.Series
) -> Dict[str, Any]:
    """
    Calculate classification metrics per initial author and overall.
    (Legacy function - kept for compatibility)
    
    Args:
        y_true: True binary labels (1 for target_author, 0 otherwise)
        y_pred: Predicted binary labels
        initial_authors: Series of initial author names
        
    Returns:
        Dictionary with metrics per initial author and overall
    """
    metrics = {}
    
    # Get unique initial authors
    unique_initial_authors = initial_authors.unique()
    
    # Calculate metrics for each initial_author
    for initial_author in unique_initial_authors:
        # Filter predictions and labels for this initial_author
        mask = initial_authors == initial_author
        y_true_filtered = y_true[mask]
        y_pred_filtered = y_pred[mask]
        
        # Calculate metrics for class 1
        precision = precision_score(
            y_true_filtered, 
            y_pred_filtered, 
            pos_label=1, 
            average='binary', 
            zero_division=0
        )
        recall = recall_score(
            y_true_filtered, 
            y_pred_filtered, 
            pos_label=1, 
            average='binary', 
            zero_division=0
        )
        f1 = f1_score(
            y_true_filtered, 
            y_pred_filtered, 
            pos_label=1, 
            average='binary', 
            zero_division=0
        )
        
        metrics[initial_author] = {
            "precision": float(precision),
            "recall": float(recall),
            "f1score": float(f1)
        }
    
    # Calculate overall metrics
    precision_overall = precision_score(
        y_true, 
        y_pred, 
        pos_label=1, 
        average='binary', 
        zero_division=0
    )
    recall_overall = recall_score(
        y_true, 
        y_pred, 
        pos_label=1, 
        average='binary', 
        zero_division=0
    )
    f1_overall = f1_score(
        y_true, 
        y_pred, 
        pos_label=1, 
        average='binary', 
        zero_division=0
    )
    
    metrics["overall"] = {
        "precision": float(precision_overall),
        "recall": float(recall_overall),
        "f1score": float(f1_overall)
    }
    
    return metrics


def compute_all_predictions(
    X_all: pd.Series,
    y_all: pd.Series,
    initial_authors_all: pd.Series,
    author_list: List[str],
    config: ClassificationConfig,
    file_suffix: str
) -> Dict[str, Any]:
    """
    Compute predictions and metrics for all target authors.
    (Legacy function - kept for compatibility)
    
    Args:
        X_all: All generated texts
        y_all: All target labels (desired_style_authors)
        initial_authors_all: All initial author names (original_authors)
        author_list: List of target authors
        config: Classification configuration
        file_suffix: Suffix for output files
        
    Returns:
        Dictionary with metrics
    """
    all_metrics = {}
    
    for target_author in author_list:
        print(f"\nProcessing target author: {target_author}")
        
        # Initialize transformer and model
        data_transformer = config.data_transformer_class(
            path_to_models=config.path_to_models, 
            author=target_author
        )
        classification_model = config.classification_model_class(
            path_to_models=config.path_to_models, 
            author=target_author, 
            data_transformer=data_transformer
        )
        
        # Transform labels to binary (1 for current author, 0 otherwise)
        y_binary = (y_all == target_author).astype(int)
        
        # Predict
        pipeline = MLPipeline(data_transformer, classification_model)
        y_pred = pipeline.predict_samples(X_all)

        print(f"Predictions for {target_author}: {y_pred.head()}")
        print(f"Predictions for {target_author}: {y_pred.shape}")

        
        # Calculate metrics
        all_metrics[target_author] = calculate_classification_metrics(
            y_binary,
            y_pred,
            initial_authors_all
        )

        heatmap_data_per_author = prepare_heatmap_data_per_author(
            y_all, 
            y_pred,
            initial_authors_all
        )

        create_heatmap_per_author(heatmap_data_per_author, target_author, config, file_suffix)
    
    return all_metrics


# ============================================================================
# STYLE SENSITIVITY AND CONTENT INVARIANCE METRICS
# ============================================================================

def calculate_nss(
    y_proba: pd.Series,
    desired_style_authors: pd.Series,
    target_classifier_author: str
) -> float:
    """
    Calculate Normalized Style Sensitivity (NSS) - Cohen's d style effect size.
    
    NSS measures how well the classifier discriminates based on steered author labels.
    A higher NSS indicates stronger style sensitivity.
    
    Args:
        y_proba: Probability predictions from the classifier
        desired_style_authors: Which author's style the texts were steered towards
        target_classifier_author: The author this classifier is trained to detect
        
    Returns:
        NSS value (effect size). 0 = no style sensitivity, >0 = classifier gives
        higher probability to texts steered toward target author.
    """
    # S_A1: samples steered toward target author
    mask_steered_target = desired_style_authors == target_classifier_author
    # S_not_A1: samples steered toward other authors
    mask_steered_other = desired_style_authors != target_classifier_author
    
    proba_steered_target = y_proba[mask_steered_target]
    proba_steered_other = y_proba[mask_steered_other]
    
    n_target = len(proba_steered_target)
    n_other = len(proba_steered_other)
    
    if n_target < 2 or n_other < 2:
        return float('nan')
    
    # Mean probabilities
    mean_target = proba_steered_target.mean()
    mean_other = proba_steered_other.mean()
    delta_p = mean_target - mean_other
    
    # Sample variances
    var_target = proba_steered_target.var(ddof=1)
    var_other = proba_steered_other.var(ddof=1)
    
    # Pooled standard deviation
    numerator = (n_target - 1) * var_target + (n_other - 1) * var_other
    denominator = n_target + n_other - 2
    
    if denominator <= 0:
        return float('nan')
    
    s_pooled = np.sqrt(numerator / denominator)
    
    if s_pooled == 0:
        return float('inf') if delta_p > 0 else float('-inf') if delta_p < 0 else 0.0
    
    nss = delta_p / s_pooled
    
    return float(nss)


def calculate_cii(
    y_proba: pd.Series,
    original_authors: pd.Series,
    author_list: List[str]
) -> float:
    """
    Calculate Content Invariance Index (CII) - measures content leakage.
    
    CII_simple = σ_μ / σ_p where:
    - σ_μ is the std of mean probabilities per original author
    - σ_p is the overall std of probabilities
    
    A CII close to 0 indicates negligible dependence on original author (good).
    Larger values indicate evidence of content leakage (bad).
    
    Args:
        y_proba: Probability predictions from the classifier
        original_authors: The original author whose article generated the prompt
        author_list: List of all authors
        
    Returns:
        CII value. Close to 0 = content invariant, larger = content leakage.
    """
    # Calculate mean probability per original author
    mu_per_author = []
    for author in author_list:
        mask = original_authors == author
        if mask.sum() > 0:
            mu_a = y_proba[mask].mean()
            mu_per_author.append(mu_a)
    
    if len(mu_per_author) < 2:
        return float('nan')
    
    # σ_μ: std of the mean probabilities across authors
    sigma_mu = np.std(mu_per_author, ddof=1)
    
    # σ_p: overall std of all probabilities
    sigma_p = y_proba.std()
    
    if sigma_p == 0:
        return float('nan')
    
    cii = sigma_mu / sigma_p
    
    return float(cii)


# ============================================================================
# VISUALIZATION
# ============================================================================

def create_heatmap(
    heatmap_data: pd.DataFrame,
    author_list: List[str],
    data_type: str
) -> plt.Figure:
    """
    Create a heatmap visualization of classification predictions.
    
    Args:
        heatmap_data: DataFrame with initial_author column and predictions for each target author
        author_list: List of target authors (column names)
        data_type: Data type string for title
        
    Returns:
        Matplotlib figure object
    """

    # Calculate figure size
    width = 24
    height = 44
    figsize = (width, height)
        
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # Prepare data for heatmap - ensure integer type for proper colormap mapping
    heatmap_matrix = heatmap_data[author_list].values.astype(int)
    row_labels = heatmap_data['initial_author'].values
    
    # Create custom colormap: red for 0, green for 1
    colors = ['#FF0000', '#00FF00']  # Red, Green
    cmap = ListedColormap(colors)
    
    # Create heatmap
    sns.heatmap(
        heatmap_matrix,
        cmap=cmap,
        vmin=0,
        vmax=1,
        cbar=False,  # Remove color bar
        ax=ax,
        yticklabels=False,  # Don't show all row labels initially (too many)
        xticklabels=author_list,
        linewidths=0.5,  # Add thin lines between cells for clarity
        linecolor='gray'
    )
    
    # Format x-axis labels
    ax.set_xticklabels(author_list, rotation=45, ha='right', fontsize=24, fontweight='bold')
    
    # Add initial author names as y-axis labels at strategic positions
    y_positions = []
    y_labels = []
    prev_author = None
    for i, author in enumerate(row_labels):
        # Show label at the start of each author group
        if author != prev_author:
            y_positions.append(i + 0.5)  # Center of the row
            y_labels.append(author)
        prev_author = author
    
    # Set y-axis labels
    ax.set_yticks(y_positions)
    ax.set_yticklabels(y_labels, fontsize=24, fontweight='bold')
    
    # Set labels and title
    ax.set_xlabel('Target Authors / Style', fontsize=24, fontweight='bold')
    ax.set_ylabel('Documents (Original Authors) / Content', fontsize=24, fontweight='bold')
    ax.set_title(
        f'Classification Predictions Heatmap ({data_type})\n'
        f'(Red: Prediction=0, Green: Prediction=1)',
        fontsize=24,
        fontweight='bold',
        pad=20
    )
    
    plt.tight_layout()
    return fig


def create_heatmap_per_author(
    heatmap_data: pd.DataFrame,
    author: str,
    config: ClassificationConfig,
    file_suffix: str
) -> None:
    """
    Create a heatmap visualization of classification predictions per author.
    """
    # Ensure output directory exists
    config.output_dir.mkdir(parents=True, exist_ok=True)
    
    # save heatmap_data to csv
    heatmap_data.to_csv(config.output_dir / f"heatmap_data_per_author__{author}{file_suffix}.csv", index=False)
    
    # Calculate figure size
    width = 24
    height = 24
    figsize = (width, height)
        
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    author_list = heatmap_data.columns[heatmap_data.columns != 'initial_author']
    # Prepare data for heatmap - ensure integer type for proper colormap mapping
    heatmap_matrix = heatmap_data[author_list].values.astype(int)
    row_labels = heatmap_data['initial_author'].values
    
    # Create custom colormap: red for 0, green for 1
    colors = ['#FF0000', '#00FF00']  # Red, Green
    cmap = ListedColormap(colors)
    
    # Create heatmap
    sns.heatmap(
        heatmap_matrix,
        cmap=cmap,
        vmin=0,
        vmax=1,
        cbar=False,  # Remove color bar
        ax=ax,
        yticklabels=False,  # Don't show all row labels initially (too many)
        xticklabels=author_list,
        linewidths=0.5,  # Add thin lines between cells for clarity
        linecolor='gray'
    )
    
    # Format x-axis labels
    ax.set_xticklabels(author_list, rotation=45, ha='right', fontsize=24, fontweight='bold')
    
    # Add initial author names as y-axis labels at strategic positions
    y_positions = []
    y_labels = []
    prev_author = None
    for i, orig_author in enumerate(row_labels):
        # Show label at the start of each author group
        if orig_author != prev_author:
            y_positions.append(i + 0.5)  # Center of the row
            y_labels.append(orig_author)
        prev_author = orig_author
    
    # Set y-axis labels
    ax.set_yticks(y_positions)
    ax.set_yticklabels(y_labels, fontsize=24, fontweight='bold')
    
    # Set labels and title
    ax.set_xlabel('Target/Desired Authors', fontsize=24, fontweight='bold')
    ax.set_ylabel('Documents (Initial Authors)', fontsize=24, fontweight='bold')
    ax.set_title(
        f'Classification Predictions Heatmap for {author} \n'
        f'(Red: Prediction=0, Green: Prediction=1)',
        fontsize=24,
        fontweight='bold',
        pad=20
    )

    plt.tight_layout()

    fig.savefig(config.output_dir / f"heatmap_per_author__{author}{file_suffix}.png", dpi=300, bbox_inches='tight')

    plt.close(fig)

    return None

def prepare_heatmap_data(
    initial_authors: pd.Series,
    predictions: Dict[str, pd.Series],
    author_list: List[str]
) -> pd.DataFrame:
    """
    Prepare DataFrame for heatmap visualization.
    
    Args:
        initial_authors: Series of initial author names
        predictions: Dictionary mapping target authors to prediction series
        author_list: List of target authors
        
    Returns:
        DataFrame sorted by initial_author with predictions columns
    """

    heatmap_data = pd.DataFrame({'initial_author': initial_authors})
    
    # Add predictions for each target author with explicit index alignment
    for target_author in author_list:
        if target_author not in predictions.keys():
            continue
        # Explicitly align predictions by index to ensure correct sample ordering
        heatmap_data[target_author] = predictions[target_author].reindex(initial_authors.index).values
    
    
    # Sort by initial_author
    heatmap_data = heatmap_data.sort_values('initial_author').reset_index(drop=True)
    
    print(f"Heatmap data: {heatmap_data.head()}")
    print(f"Heatmap data: {heatmap_data.shape}")
    return heatmap_data


def prepare_heatmap_data_per_author(
    y_all: pd.Series,
    y_pred: pd.Series,
    initial_authors: pd.Series
) -> pd.DataFrame:
    """
    Prepare DataFrame for heatmap visualization per author.
    """
    dataframe = pd.DataFrame(columns=[unique_author for unique_author in y_all.unique()]+['initial_author']) 
    for unique_author in y_all.unique():
        indices_where_unique_author = y_all == unique_author
        dataframe[unique_author] = y_pred[indices_where_unique_author].values
        dataframe[f'initial_author_{unique_author}'] = initial_authors[indices_where_unique_author].values
    
    for unique_author in y_all.unique():
        for unique_author_2 in y_all.unique():
            assert dataframe[f'initial_author_{unique_author}'].equals(dataframe[f'initial_author_{unique_author_2}']), f"Initial authors for {unique_author} and {unique_author_2} are not the same"

    dataframe['initial_author'] = dataframe[f'initial_author_{y_all.unique()[0]}']
    # drop all other initial_author columns
    dataframe = dataframe.drop(columns=[f'initial_author_{unique_author}' for unique_author in y_all.unique()])
    dataframe = dataframe.sort_values('initial_author').reset_index(drop=True)
    
    return dataframe


def create_boxplot_per_classifier(
    y_proba: pd.Series,
    desired_style_authors: pd.Series,
    original_authors: pd.Series,
    target_classifier_author: str,
    author_list: List[str],
    config: ClassificationConfig,
    file_suffix: str
) -> None:
    """
    Create boxplot visualization showing probability distributions.
    
    Y-axis: probabilities
    X-axis: original authors
    Subplots: one per steered author
    
    This visualization helps assess:
    - Style Sensitivity: Boxplots for steered→target should be higher than others
    - Content Invariance: Within each subplot, boxplots should be similar across original authors
    
    Args:
        y_proba: Probability predictions from the classifier
        desired_style_authors: Which author's style the texts were steered towards
        original_authors: The original author whose article generated the prompt
        target_classifier_author: The author this classifier is trained to detect
        author_list: List of all authors
        config: Classification configuration
        file_suffix: Suffix for output files
    """
    # Ensure output directory exists
    config.output_dir.mkdir(parents=True, exist_ok=True)
    
    n_steered = len(author_list)
    fig, axes = plt.subplots(1, n_steered, figsize=(6 * n_steered, 6), sharey=True)
    
    if n_steered == 1:
        axes = [axes]
    
    # Color palette for original authors
    colors = sns.color_palette("husl", len(author_list))
    
    for idx, steered_author in enumerate(author_list):
        ax = axes[idx]
        
        # Filter to texts steered toward this author
        mask_steered = desired_style_authors == steered_author
        
        # Prepare data for boxplot
        boxplot_data = []
        positions = []
        box_colors = []
        
        for orig_idx, orig_author in enumerate(author_list):
            mask_orig = original_authors == orig_author
            combined_mask = mask_steered & mask_orig
            
            if combined_mask.sum() > 0:
                proba_subset = y_proba[combined_mask].values
                boxplot_data.append(proba_subset)
                positions.append(orig_idx)
                box_colors.append(colors[orig_idx])
        
        if boxplot_data:
            bp = ax.boxplot(
                boxplot_data,
                positions=positions,
                widths=0.6,
                patch_artist=True,
                showfliers=True,
                flierprops=dict(marker='o', markersize=4, alpha=0.5)
            )
            
            # Color the boxes
            for patch, color in zip(bp['boxes'], box_colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)
        
        # Highlight if this is the steered author matching the classifier target
        if steered_author == target_classifier_author:
            ax.set_facecolor('#e8f5e9')  # Light green background
            ax.set_title(f'Steered → {steered_author}\n(TARGET)', fontsize=12, fontweight='bold')
        else:
            ax.set_title(f'Steered → {steered_author}', fontsize=12)
        
        ax.set_xticks(range(len(author_list)))
        ax.set_xticklabels([a.split()[-1] for a in author_list], rotation=45, ha='right', fontsize=10)
        ax.set_xlabel('Original Author', fontsize=11)
        
        if idx == 0:
            ax.set_ylabel('Probability (P(target author))', fontsize=11)
        
        ax.set_ylim(-0.05, 1.05)
        ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, linewidth=1)
        ax.grid(axis='y', alpha=0.3)
    
    # Overall title
    fig.suptitle(
        f'Classifier: {target_classifier_author}\n'
        f'Probability Distributions by Steered Author and Original Author',
        fontsize=14,
        fontweight='bold',
        y=1.02
    )
    
    plt.tight_layout()
    
    # Save figure
    output_path = config.output_dir / f"boxplot__{target_classifier_author}{file_suffix}.png"
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    print(f"Saved boxplot to: {output_path}")



# ============================================================================
# RESULTS SAVING
# ============================================================================

def create_structured_output(
    per_target_classifier: Dict[str, Any],
    config: ClassificationConfig
) -> Dict[str, Any]:
    """
    Create the full structured output with metadata.
    
    Args:
        per_target_classifier: The structured metrics
        config: Classification configuration
        
    Returns:
        Complete output dictionary matching the desired schema
    """
    return {
        "date": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
        "run_id": str(uuid.uuid4()),
        "config": {
            "data_transformer": config.data_transformer_class.__name__,
            "classifier_name": config.classification_model_class.__name__,
            "classifier_run_name": config.classifier_run_name,
            "path_to_models": str(config.path_to_models),
            "steering_method": config.steering_method,
            "n_shap_features": config.n_shap_features,
            "generation_run_name": config.run_name
        },
        "results": {
            "per_target_classifier": per_target_classifier
        }
    }


def save_structured_results(
    structured_output: Dict[str, Any],
    config: ClassificationConfig,
    file_suffix: str = None
) -> Dict[str, Path]:
    """
    Save structured classification results to files.
    
    Args:
        structured_output: Complete structured output dictionary
        config: Classification configuration
        file_suffix: Suffix for filename
        
    Returns:
        Dictionary mapping file type to saved path
    """
    # Create output directory
    config.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save paths
    saved_paths = {}
    
    # Save JSON metrics
    json_path = config.output_dir / f"structured_results{file_suffix}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(structured_output, f, indent=2)
    saved_paths['json'] = json_path
    
    print(f"Saved structured results to: {json_path}")
    
    return saved_paths


def save_results(
    metrics: Dict[str, Any],
    config: ClassificationConfig,
    file_suffix: str = None
) -> Dict[str, Path]:
    """
    Save classification results to files.
    (Legacy function - kept for compatibility)
    
    Args:
        metrics: Dictionary of metrics for each target author
        config: Classification configuration
        file_suffix: Suffix for filename
    Returns:
        Dictionary mapping file type to saved path
    """
    # Create output directory
    config.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save paths
    saved_paths = {}
    
    # Save JSON metrics
    json_path = config.output_dir / f"classification_metrics{file_suffix}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    saved_paths['json'] = json_path
    
    return saved_paths



# ============================================================================
# MAIN PIPELINE
# ============================================================================

def run_classification_pipeline(config: Optional[ClassificationConfig] = None):
    """
    Run the complete classification pipeline with structured output.
    
    Output schema:
    {
        "date": "2025-12-17T00:00:00Z",
        "run_id": "uuid",
        "config": {...},
        "results": {
            "per_target_classifier": {
                "author_A": {
                    "metrics": {
                        "NSS": float,  # Normalized Style Sensitivity
                        "CII": float   # Content Invariance Index
                    },
                    "per_desired_style_author": {
                        "author_A": {
                            "per_original_author": {
                                "author_A": {"sum_predictions": int, "avg_probs": float, "n_samples": int},
                                ...
                            }
                        }
                    }
                }
            }
        }
    }
    
    Args:
        config: Optional[ClassificationConfig]. If None, uses default config.
    """
    # Use default config if not provided
    if config is None:
        config = ClassificationConfig()
    
    print("=" * 80)
    print("Classification Pipeline for Generated Texts (Structured Output)")
    print("=" * 80)
    print(f"Classifier Model: {config.classification_model_class.__name__}")
    print(f"Transformer: {config.data_transformer_class.__name__}")
    print(f"Path to Models: {config.path_to_models}")
    print(f"Classifier Run: {config.classifier_run_name or 'default'}")
    print("-" * 40)
    print(f"Generation Run: {config.run_name}")
    print(f"Steering Method: {config.steering_method}")
    print(f"N SHAP Features: {config.n_shap_features}")
    print(f"Authors: {config.author_list}")
    print(f"Input template: {config.path_to_generated_texts_template}")
    print(f"Output dir: {config.output_dir}")
    print("=" * 80)
    
    # Step 1: Extract data type from path
    data_type = extract_data_type_from_path(
        config.path_to_generated_texts_template,
        config.author_list[0]
    )
    print(f"\nDetected data type from filename: {data_type}")
    
    # Auto-detect steering method from path if not specified
    if config.steering_method is None:
        if "steered" in data_type:
            # Extract steering method (e.g., "steered__heuristic" -> "heuristic")
            parts = data_type.split("__")
            if len(parts) > 1:
                config.steering_method = parts[-1]
            else:
                config.steering_method = "unknown"
        else:
            config.steering_method = None  # baseline
    
    # Step 2: Aggregate all texts
    print("\nAggregating generated texts...")
    X_all, desired_style_authors_all, original_authors_all = aggregate_generated_texts(
        config.path_to_generated_texts_template,
        config.author_list
    )
    print(
        f"Aggregated data: X shape={X_all.shape}, "
        f"desired_style_authors shape={desired_style_authors_all.shape}, "
        f"original_authors shape={original_authors_all.shape}"
    )

    file_suffix = (
        f"__{data_type}__"
        f"{config.classification_model_class.__name__}__"
        f"{config.data_transformer_class.__name__}"
    )
    
    # Step 3: Compute structured metrics
    print("\nComputing structured metrics...")
    per_target_classifier = compute_structured_metrics(
        X_all, 
        desired_style_authors_all, 
        original_authors_all, 
        config.author_list, 
        config, 
        file_suffix
    )
    
    # Step 4: Create structured output
    structured_output = create_structured_output(per_target_classifier, config)
    
    # Step 5: Save structured results
    print("\nSaving structured results...")
    saved_paths = save_structured_results(
        structured_output,
        config,
        file_suffix
    )
    
    print("\n" + "=" * 80)
    print("Results saved to:")
    print(f"  JSON: {saved_paths['json']}")
    print("=" * 80)
    
    return structured_output


def run_classification_from_run_dir(
    run_dir: Path,
    classification_model_class: type = ModernBertClassifierModel,
    data_transformer_class: type = ModernBertTransformer,
    path_to_models: Optional[Path] = None
) -> Dict[str, Any]:
    """
    Convenience function to run classification on a generation run directory.
    
    Uses default classifier model/transformer classes. For custom classifiers
    trained with main.py, use run_classification_from_configs() instead.
    
    Args:
        run_dir: Path to the run directory containing run_config.json
        classification_model_class: Classification model to use
        data_transformer_class: Data transformer to use
        path_to_models: Path to classifier models (optional)
        
    Returns:
        Structured classification results
        
    Example:
        results = run_classification_from_run_dir("data/steering/tests/my_run")
    """
    config = ClassificationConfig.from_run_dir(
        Path(run_dir),
        classification_model_class=classification_model_class,
        data_transformer_class=data_transformer_class,
        path_to_models=path_to_models
    )
    return run_classification_pipeline(config)


def run_classification_from_configs(
    generation_run_dir_name: str,
    classifier_run_name: str,
    classifier_base_dir: Optional[Path] = None
) -> Dict[str, Any]:
    """
    Run classification using both generation and classifier training configs.
    
    This is the RECOMMENDED method for full pipeline integration. It automatically:
    1. Loads the generation run config (RunConfig)
    2. Loads the classifier training config (ClassifierTrainingConfig)
    3. Configures the classification pipeline with the correct models and paths
    
    Args:
        generation_run_dir_name: Name of the generation run directory containing run_config.json
        classifier_run_name: Name of the classifier training run (from main.py)
        classifier_base_dir: Base directory for classifiers (uses default if None)
        
    Returns:
        Structured classification results
        
    Example:
        # After training classifiers with:
        #   python main.py --run-name modernbert_10epochs --data-subset all
        #
        # And generating texts that created:
        #   data/steering/tests/my_generation_run/
        #
        # Run classification:
        results = run_classification_from_configs(
            generation_run_dir="data/steering/tests/my_generation_run",
            classifier_run_name="modernbert_10epochs"
        )
    """
    config = ClassificationConfig.from_configs(
        generation_run_dir_name=generation_run_dir_name,
        classifier_run_name=classifier_run_name,
        classifier_base_dir=classifier_base_dir
    )
    return run_classification_pipeline(config)


def run_classification_pipeline_legacy(config: Optional[ClassificationConfig] = None):
    """
    Run the complete classification pipeline (legacy format).
    Kept for backwards compatibility.
    
    Args:
        config: Optional[ClassificationConfig]. If None, uses default config.
    """
    # Use default config if not provided
    if config is None:
        config = ClassificationConfig()
    
    print("=" * 80)
    print("Classification Pipeline for Generated Texts (Legacy)")
    print("=" * 80)
    print(f"Model: {config.classification_model_class.__name__}")
    print(f"Transformer: {config.data_transformer_class.__name__}")
    print(f"Authors: {config.author_list}")
    print(f"Input template: {config.path_to_generated_texts_template}")
    print("=" * 80)
    
    # Step 1: Extract data type from path
    data_type = extract_data_type_from_path(
        config.path_to_generated_texts_template,
        config.author_list[0]
    )
    print(f"\nDetected data type from filename: {data_type}")
    
    # Step 2: Aggregate all texts
    print("\nAggregating generated texts...")
    X_all, y_all, initial_authors_all = aggregate_generated_texts(
        config.path_to_generated_texts_template,
        config.author_list
    )
    print(
        f"Aggregated data: X shape={X_all.shape}, "
        f"y shape={y_all.shape}, "
        f"initial_authors shape={initial_authors_all.shape}"
    )

    file_suffix = (
        f"__{data_type}__"
        f"{config.classification_model_class.__name__}__"
        f"{config.data_transformer_class.__name__}"
    )
    
    # Compute predictions and metrics for all target authors
    all_metrics = compute_all_predictions(
        X_all, y_all, initial_authors_all, config.author_list, config, file_suffix
    )
    
    # Save results
    print("\nSaving results...")
    saved_paths = save_results(
        all_metrics,
        config,
        file_suffix
    )
    
    print("\n" + "=" * 80)
    print("Results saved to:")
    print(f"  JSON: {saved_paths['json']}")
    print("=" * 80)

# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    # =========================================================================
    # OPTION 1 (RECOMMENDED): Full pipeline with trained classifiers
    # =========================================================================
    # Use this when you have:
    # - Trained classifiers with main.py (creates ClassifierTrainingConfig)
    # - Generated texts with steered_text_generation.py (creates RunConfig)


    for classifier_run_name in ["tfidf_3gram_max_features5000_tfidf_logistic_regression", "tfidf_3gram_max_features5000_tfidf_sgd", "tfidf_3gram_max_features5000_tfidf_random_forest", "tfidf_3gram_tfidf_logistic_regression", "tfidf_3gram_tfidf_sgd", "tfidf_3gram_tfidf_random_forest", "modern_bert_all", "modern_bert_one_author"]:
        for generation_run_dir in ["projected_gradient_test_run"]:
            print(f"Running classification for {classifier_run_name} and {generation_run_dir}")
            results = run_classification_from_configs(
                generation_run_dir_name=generation_run_dir,
                classifier_run_name=classifier_run_name
            )
    
    # =========================================================================
    # OPTION 2: Use default classifiers with a generation run
    # =========================================================================
    # Use this when you have a run_config.json but want to use default classifiers
    
    # results = run_classification_from_run_dir(
    #    "data/steering/tests/run_with_32_features",
    #    classification_model_class=ModernBertClassifierModel,
    #    data_transformer_class=ModernBertTransformer
    #)
    
    # =========================================================================
    # OPTION 3: Manual configuration (legacy approach, still supported)
    # =========================================================================
    
    # Configuration 1: ModernBERT with ModernBERT transformer - BASELINE
     #config1_baseline = ClassificationConfig(
     #   classification_model_class=ModernBertClassifierModel,
     #   data_transformer_class=ModernBertTransformer,
     #   path_to_generated_texts_template="data/steering/tests/generated_texts__sae_baseline__{}.json",
     #   steering_method=None,  # baseline has no steering
     #   n_shap_features=16,
     #   run_name="baseline_manual"
     #)
    
    # Configuration 2: ModernBERT with ModernBERT transformer - STEERED
    #   config1_steered = ClassificationConfig(
    #    classification_model_class=ModernBertClassifierModel,
    #    data_transformer_class=ModernBertTransformer,
    #    path_to_generated_texts_template="data/steering/tests/run_after_hooks_cache_fix/generated_texts__steered__heuristic__{}.json",
    #    steering_method="heuristic",
    #    n_shap_features=16,
    #    run_name="heuristic_steered_manual"
    #)
    
    # Run pipelines (uncomment as needed)
    # run_classification_pipeline(config1_baseline)
    # run_classification_pipeline(config1_steered)
