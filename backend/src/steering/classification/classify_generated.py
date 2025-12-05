"""
Classification pipeline for generated texts.

This module provides functionality to classify generated texts using various
classification models and data transformers. It supports easy configuration
of models, transformers, and input paths.
"""

from pathlib import Path
from typing import Tuple, Dict, Any, List, Optional
from dataclasses import dataclass

import json
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


# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class ClassificationConfig:
    """Configuration class for classification pipeline."""
    
    # Model and transformer classes
    classification_model_class: type = ModernBertClassifierModel
    data_transformer_class: type = ModernBertTransformer
    
    # Paths
    path_to_models: Path = Path("backend/src/steering/classification/models")
    path_to_generated_texts_template: str = "data/steering/tests/generated_texts__sae_baseline__{}.json"
    output_dir: Path = Path("data/steering/tests/classification_results")
    
    # Author list
    author_list: List[str] = None
    
    def __post_init__(self):
        """Initialize default author list if not provided."""
        if self.author_list is None:
            self.author_list = ["Sam Levine", "Paige Lavender", "Lee Moran", "Amanda Terkel"]


# ============================================================================
# DATA LOADING
# ============================================================================

def read_generated_texts(
    path_to_generated_texts: Path, 
    target_author: str
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    Read generated texts from JSON file and split features and target.
    
    Args:
        path_to_generated_texts: Path to JSON file containing generated texts
        target_author: Target author name to use as label
        
    Returns:
        Tuple of (X, y, initial_authors) as pandas Series
    """
    with open(path_to_generated_texts, "r", encoding="utf-8") as f:
        generated_texts = json.load(f)

    X = []
    y = []
    initial_authors = []
    
    for doc in generated_texts:
        X.append(doc["generated_text"])
        y.append(target_author)
        initial_authors.append(doc["author"])

    return pd.Series(X), pd.Series(y), pd.Series(initial_authors)


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
        Tuple of (X_all, y_all, initial_authors_all) as pandas Series
    """
    X_all = []
    y_all = []
    initial_authors_all = []

    for target_author in author_list:
        path_to_generated_texts = path_template.format(target_author)
        X, y, initial_authors = read_generated_texts(
            Path(path_to_generated_texts), 
            target_author
        )
        X_all.extend(X.tolist())
        y_all.extend(y.tolist())
        initial_authors_all.extend(initial_authors.tolist())

    return pd.Series(X_all), pd.Series(y_all), pd.Series(initial_authors_all)


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

def calculate_classification_metrics(
    y_true: pd.Series,
    y_pred: pd.Series,
    initial_authors: pd.Series,
    target_author: str
) -> Dict[str, Any]:
    """
    Calculate classification metrics per initial author and overall.
    
    Args:
        y_true: True binary labels (1 for target_author, 0 otherwise)
        y_pred: Predicted binary labels
        initial_authors: Series of initial author names
        target_author: Target author name for this classification task
        
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
    config: ClassificationConfig
) -> Tuple[Dict[str, pd.Series], Dict[str, Dict[str, Any]]]:
    """
    Compute predictions and metrics for all target authors.
    
    Args:
        X_all: All generated texts
        y_all: All target labels
        initial_authors_all: All initial author names
        author_list: List of target authors
        config: Classification configuration
        
    Returns:
        Tuple of (all_predictions dict, metrics dict)
    """
    all_predictions = {}
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
        
        # Store predictions
        all_predictions[target_author] = y_pred
        
        # Calculate metrics
        all_metrics[target_author] = calculate_classification_metrics(
            y_binary,
            y_pred,
            initial_authors_all,
            target_author
        )
    
    return all_predictions, all_metrics


# ============================================================================
# VISUALIZATION
# ============================================================================

def create_heatmap(
    heatmap_data: pd.DataFrame,
    author_list: List[str],
    data_type: str,
    figsize: Optional[Tuple[int, int]] = None
) -> plt.Figure:
    """
    Create a heatmap visualization of classification predictions.
    
    Args:
        heatmap_data: DataFrame with initial_author column and predictions for each target author
        author_list: List of target authors (column names)
        data_type: Data type string for title
        figsize: Optional figure size tuple
        
    Returns:
        Matplotlib figure object
    """
    # Prepare data for heatmap
    heatmap_matrix = heatmap_data[author_list].values
    row_labels = heatmap_data['initial_author'].values
    
    # Calculate figure size
    if figsize is None:
        width = max(16, len(author_list) * 3)
        height = max(10, len(heatmap_data) * 0.1)
        figsize = (width, height)
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
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
    ax.set_xticklabels(author_list, rotation=45, ha='right', fontsize=14, fontweight='bold')
    
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
    ax.set_yticklabels(y_labels, fontsize=12, fontweight='bold')
    
    # Set labels and title
    ax.set_xlabel('Target Authors', fontsize=14, fontweight='bold')
    ax.set_ylabel('Documents (Initial Authors)', fontsize=14, fontweight='bold')
    ax.set_title(
        f'Classification Predictions Heatmap ({data_type})\n'
        f'(Red=0/Prediction=0, Green=1/Prediction=1)',
        fontsize=16,
        fontweight='bold',
        pad=20
    )
    
    plt.tight_layout()
    return fig


def prepare_heatmap_data(
    initial_authors: pd.Series,
    all_predictions: Dict[str, pd.Series],
    author_list: List[str]
) -> pd.DataFrame:
    """
    Prepare DataFrame for heatmap visualization.
    
    Args:
        initial_authors: Series of initial author names
        all_predictions: Dictionary mapping target authors to prediction series
        author_list: List of target authors
        
    Returns:
        DataFrame sorted by initial_author with predictions columns
    """
    heatmap_data = pd.DataFrame({'initial_author': initial_authors})
    
    # Add predictions for each target author
    for target_author in author_list:
        heatmap_data[target_author] = all_predictions[target_author]
    
    # Sort by initial_author
    heatmap_data = heatmap_data.sort_values('initial_author').reset_index(drop=True)
    
    return heatmap_data


# ============================================================================
# RESULTS SAVING
# ============================================================================

def save_results(
    metrics: Dict[str, Any],
    heatmap_data: pd.DataFrame,
    heatmap_fig: plt.Figure,
    data_type: str,
    config: ClassificationConfig
) -> Dict[str, Path]:
    """
    Save classification results to files.
    
    Args:
        metrics: Dictionary of metrics for each target author
        heatmap_data: DataFrame with predictions for heatmap
        heatmap_fig: Matplotlib figure object
        data_type: Data type string for filename
        config: Classification configuration
        
    Returns:
        Dictionary mapping file type to saved path
    """
    # Create output directory
    config.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create file suffix
    file_suffix = (
        f"__{data_type}__"
        f"{config.classification_model_class.__name__}__"
        f"{config.data_transformer_class.__name__}"
    )
    
    # Save paths
    saved_paths = {}
    
    # Save JSON metrics
    json_path = config.output_dir / f"classification_metrics{file_suffix}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    saved_paths['json'] = json_path
    
    # Save CSV predictions
    csv_path = config.output_dir / f"classification_predictions{file_suffix}.csv"
    heatmap_data.to_csv(csv_path, index=False)
    saved_paths['csv'] = csv_path
    
    # Save heatmap
    heatmap_path = config.output_dir / f"classification_heatmap{file_suffix}.png"
    heatmap_fig.savefig(heatmap_path, dpi=300, bbox_inches='tight')
    plt.close(heatmap_fig)
    saved_paths['heatmap'] = heatmap_path
    
    return saved_paths


# ============================================================================
# MAIN PIPELINE
# ============================================================================

def run_classification_pipeline(config: Optional[ClassificationConfig] = None) -> Dict[str, Any]:
    """
    Run the complete classification pipeline.
    
    Args:
        config: Optional[ClassificationConfig]. If None, uses default config.
        
    Returns:
        Dictionary containing results and saved file paths
    """
    # Use default config if not provided
    if config is None:
        config = ClassificationConfig()
    
    print("=" * 80)
    print("Classification Pipeline for Generated Texts")
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
    
    # Step 3: Compute predictions and metrics for all target authors
    all_predictions, all_metrics = compute_all_predictions(
        X_all, y_all, initial_authors_all, config.author_list, config
    )
    
    # Step 5: Prepare heatmap data
    print("\nPreparing heatmap data...")
    heatmap_data = prepare_heatmap_data(
        initial_authors_all,
        all_predictions,
        config.author_list
    )
    
    # Step 6: Create heatmap
    print("Creating heatmap visualization...")
    heatmap_fig = create_heatmap(heatmap_data, config.author_list, data_type)
    
    # Step 7: Save results
    print("\nSaving results...")
    saved_paths = save_results(
        all_metrics,
        heatmap_data,
        heatmap_fig,
        data_type,
        config
    )
    
    print("\n" + "=" * 80)
    print("Results saved to:")
    print(f"  JSON: {saved_paths['json']}")
    print(f"  CSV: {saved_paths['csv']}")
    print(f"  Heatmap: {saved_paths['heatmap']}")
    print("=" * 80)
    
    return {
        'metrics': all_metrics,
        'predictions': all_predictions,
        'heatmap_data': heatmap_data,
        'saved_paths': saved_paths,
    }


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    # Example configurations - uncomment and modify as needed
    
    # Configuration 1: ModernBERT with ModernBERT transformer (default)
    config = ClassificationConfig(
        classification_model_class=ModernBertClassifierModel,
        data_transformer_class=ModernBertTransformer,
        path_to_generated_texts_template="data/steering/tests/generated_texts__sae_baseline__{}.json",
    )
    
    # Configuration 2: Logistic Regression with TF-IDF
    # config = ClassificationConfig(
    #     classification_model_class=LogisticRegressionModel,
    #     data_transformer_class=TFIDFTransformer,
    #     path_to_generated_texts_template="data/steering/tests/generated_texts__baseline__{}.json",
    # )
    
    # Configuration 3: Random Forest with Sentence Embeddings
    # config = ClassificationConfig(
    #     classification_model_class=RandomForestModel,
    #     data_transformer_class=SentenceEmbeddingTransformer,
    #     path_to_generated_texts_template="data/steering/tests/generated_texts__steered__heuristic__{}.json",
    # )
    
    # Run pipeline
    results = run_classification_pipeline(config)
