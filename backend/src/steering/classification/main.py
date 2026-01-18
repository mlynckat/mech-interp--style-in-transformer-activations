# main.py
"""
Main execution script for ML classification pipeline.

This module trains author classification models and saves them with a 
ClassifierTrainingConfig for seamless integration with the classification pipeline.

Usage:
    # Train with default settings
    python main.py --run-name my_experiment
    
    # Train ModernBERT for 10 epochs on all data
    python main.py --run-name modernbert_10epochs --transformers modern_bert --models modern_bert --data-subset all
    
    # Train on one-author subset
    python main.py --run-name modernbert_one_author --data-subset one_author

Integration with classify_generated.py:
    # After training, use the trained models for classification:
    from backend.src.steering.classification.classify_generated import run_classification_from_configs
    
    results = run_classification_from_configs(
        generation_run_dir="data/steering/tests/my_generation_run",
        classifier_run_name="my_experiment"
    )
"""

import json
import argparse
from typing import Dict, Any, List, Optional
from pathlib import Path
from collections import defaultdict
from datetime import datetime

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

from backend.src.steering.classification.read_data import DataReader
from backend.src.steering.classification.data_transformers import (
    TFIDFTransformer, 
    SentenceEmbeddingTransformer, 
    ModernBertTransformer
)
from backend.src.steering.classification.classification_models import (
    LogisticRegressionModel, 
    RandomForestModel, 
    SGDClassifierModel, 
    ModernBertClassifierModel
)
from backend.src.steering.classification.pipeline import MLPipeline
from backend.src.steering.run_config import ClassifierTrainingConfig, DEFAULT_AUTHORS


# ============================================================================
# CONFIGURATION MAPPINGS
# ============================================================================

AVAILABLE_TRANSFORMERS = {
    "tfidf": TFIDFTransformer,
    "sentence_embedding": SentenceEmbeddingTransformer,
    "modern_bert": ModernBertTransformer,
}

AVAILABLE_MODELS = {
    "logistic_regression": LogisticRegressionModel,
    "random_forest": RandomForestModel,
    "sgd": SGDClassifierModel,
    "modern_bert": ModernBertClassifierModel,
}

# Valid combinations (some transformers may only work with certain models)
VALID_COMBINATIONS = {
    "tfidf": ["logistic_regression", "random_forest", "sgd"],
    "sentence_embedding": ["logistic_regression", "random_forest", "sgd"],
    "modern_bert": ["modern_bert"],
}


# ============================================================================
# RESULT SAVING UTILITIES
# ============================================================================

def save_results(
    results_dir: Path, 
    results: Dict[str, Any], 
    config: ClassifierTrainingConfig
) -> Path:
    """
    Save training results to JSON file and create aggregated results dataframe.
    
    Args:
        results_dir: Directory to save results
        results: Dictionary of results per transformer/model/author
        config: Training configuration
        
    Returns:
        Path to saved results file
    """
    datetime_suffix = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save raw results
    filepath = results_dir / f"results_{datetime_suffix}_{config.run_name}.json"
    with open(filepath, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"Results saved to: {filepath}")

    # Create aggregated results dataframe
    aggregated_results = {
        "value": [],
        "score": [],
        "data_transformation": [],
        "classification_model": [],
        "author": [],
    }

    for data_transformer_class in results.keys():
        for classification_model_class in results[data_transformer_class].keys():
            for author in results[data_transformer_class][classification_model_class].keys():
                test_results = results[data_transformer_class][classification_model_class][author]["test_metrics"]
                for score, value in test_results.items():
                    aggregated_results["value"].append(value)
                    aggregated_results["score"].append(score)
                    aggregated_results["data_transformation"].append(data_transformer_class)
                    aggregated_results["classification_model"].append(classification_model_class)
                    aggregated_results["author"].append(author)

    results_df = pd.DataFrame(aggregated_results)
    csv_path = results_dir / f"results_aggregated_{datetime_suffix}_{config.run_name}.csv"
    results_df.to_csv(csv_path, index=False)
    print(f"Aggregated results saved to: {csv_path}")
    
    return filepath


def get_confusion_matrix(predicted_labels: list, true_labels: list) -> pd.DataFrame:
    """
    Create confusion matrix as DataFrame.
    
    Args:
        predicted_labels: List of predicted labels
        true_labels: List of true labels
        
    Returns:
        DataFrame with confusion matrix
    """
    true_labels_unique = list(set(true_labels))
    predicted_labels_unique = list(set(predicted_labels))

    df_conf_matrix = pd.DataFrame(0, index=true_labels_unique, columns=predicted_labels_unique)
    
    for i in range(len(true_labels)):
        df_conf_matrix.loc[true_labels[i], predicted_labels[i]] += 1  

    print(f"Confusion matrix: {df_conf_matrix}")
    return df_conf_matrix


# ============================================================================
# TRAINING FUNCTIONS
# ============================================================================

def run_single_experiment(
    config: ClassifierTrainingConfig,
    data_transformer_class,
    classification_model_class,
    data_transformer_params: dict = None,
    classification_model_params: dict = None,
) -> Dict[str, Any]:
    """
    Run a single training experiment for all authors.
    
    Args:
        config: Training configuration
        data_transformer_class: Data transformer class to use
        classification_model_class: Classification model class to use
        data_transformer_params: Optional transformer parameters
        classification_model_params: Optional model parameters
        
    Returns:
        Dictionary of results per author
    """
    results = {}
    
    # 1. Read data
    X, y = DataReader.read_news_json_data()
    
    if config.data_subset == "all":
        print(f"Data shape: X: {X.shape}, y: {y.shape}")
        print(f"Data unique authors: {y.unique()}")
        print(f"Data author distribution: {y.value_counts()}")

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=config.test_size,
            random_state=config.random_state,
            stratify=y,
        )
    
    elif config.data_subset == "one_author":
        X_train, X_test, y_train, y_test = DataReader.read_news_generated_data()
        print(f"Data shape: X_train: {X_train.shape}, y_train: {y_train.shape}")
        print(f"Data shape: X_test: {X_test.shape}, y_test: {y_test.shape}")
        print(f"Data unique authors: {y_train.unique()}")
        print(f"Data author distribution: {y_train.value_counts()}")
    else:
        raise ValueError(f"Data subset {config.data_subset} is not supported.")

    # Train binary classifier for each author
    for author in y.unique():
        print(f"\n{'='*60}")
        print(f"Training classifier for author: {author}")
        print(f"{'='*60}")

        if config.data_subset == "all":
            X_train_subset = X_train
            X_test_subset = X_test
            y_train_subset = y_train
            y_test_subset = y_test
            y_train_author_subset = (y_train_subset == author).astype(int)
            y_test_author_subset = (y_test_subset == author).astype(int)
        elif config.data_subset == "one_author":
            author_mask_train = (y_train == author) | (y_train == f"{author} generated")
            author_mask_test = (y_test == author) | (y_test == f"{author} generated")
            X_train_subset = X_train[author_mask_train]
            X_test_subset = X_test[author_mask_test]
            y_train_subset = y_train[author_mask_train]
            y_test_subset = y_test[author_mask_test]
            y_train_author_subset = (y_train_subset == author).astype(int)
            y_test_author_subset = (y_test_subset == author).astype(int)

        print(f"Train data shape: X_train: {X_train_subset.shape}, y_train: {y_train_subset.shape}")
        print(f"Test data shape: X_test: {X_test_subset.shape}, y_test: {y_test_subset.shape}")
        print(f"Train data author distribution: {y_train_author_subset.value_counts()}")
        print(f"Test data author distribution: {y_test_author_subset.value_counts()}")

        results[author] = {}

        # 2. Initialize data transformation and model
        data_transformer = data_transformer_class(
            path_to_models=config.models_dir, 
            author=author, 
            **(data_transformer_params or {})
        )
        classification_model = classification_model_class(
            path_to_models=config.models_dir, 
            author=author, 
            data_transformer=data_transformer, 
            **(classification_model_params or {})
        )

        # 3. Create and run pipeline
        pipeline = MLPipeline(data_transformer, classification_model)
        results_per_author = pipeline.run(
            X_train_subset, 
            y_train_author_subset, 
            X_test_subset, 
            y_test_author_subset
        )

        # 4. Create and save confusion matrix
        predicted_labels = results_per_author["predicted_labels"].map({1: author, 0: "other"}).values.tolist()
        conf_matrix = get_confusion_matrix(predicted_labels, y_test_subset.values.tolist())
        
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(conf_matrix, annot=True, fmt='d', ax=ax)
        ax.set_title(f"Confusion Matrix for {author}\n{data_transformer_class.__name__} + {classification_model_class.__name__}")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        cm_path = config.results_dir / f"confusion_matrix_{author}_{config.run_name}.png"
        fig.savefig(cm_path)
        plt.close(fig)
        print(f"Saved confusion matrix to: {cm_path}")

        results[author].update(results_per_author)
      
    return results


def run_training(config: ClassifierTrainingConfig) -> Dict[str, Any]:
    """
    Run classifier training with the given configuration.
    
    This is the main entry point for training classifiers.
    
    Args:
        config: ClassifierTrainingConfig with all training settings
        
    Returns:
        Dictionary of training results
    """
    print("=" * 80)
    print("Classifier Training Pipeline")
    print("=" * 80)
    print(f"Run name: {config.run_name}")
    print(f"Description: {config.description}")
    print(f"Transformer: {config.transformer_name}")
    print(f"Model: {config.model_name}")
    print(f"Data subset: {config.data_subset}")
    print(f"Test size: {config.test_size}")
    print(f"Random state: {config.random_state}")
    print(f"Models directory: {config.models_dir}")
    print("=" * 80)
    
    # Ensure directories exist
    config.ensure_directories_exist()
    
    # Save configuration
    config.save_config()
    
    # Get classes
    data_transformer_class = AVAILABLE_TRANSFORMERS[config.transformer_name]
    classification_model_class = AVAILABLE_MODELS[config.model_name]
    
    # Run training
    results = defaultdict(lambda: defaultdict(dict))
    
    try:
        results[data_transformer_class.__name__][classification_model_class.__name__] = run_single_experiment(
            config=config,
            data_transformer_class=data_transformer_class,
            classification_model_class=classification_model_class,
            data_transformer_params=config.transformer_params,
            classification_model_params=config.model_params,
        )
    except Exception as e:
        print(f"Error during training: {e}")
        import traceback
        traceback.print_exc()
        raise
    
    # Save results
    save_results(config.results_dir, dict(results), config)
    
    print("\n" + "=" * 80)
    print("Training completed!")
    print(f"Models saved to: {config.models_dir}")
    print(f"Config saved to: {config.models_dir / 'training_config.json'}")
    print("=" * 80)
    
    return dict(results)


# ============================================================================
# COMMAND LINE INTERFACE
# ============================================================================

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Train ML classification models with configurable settings",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
    Examples:
    # Train ModernBERT with default settings
    python main.py --run-name my_experiment
    
    # Train with specific transformer and model
    python main.py --run-name tfidf_logreg --transformers tfidf --models logistic_regression
    
    # Train on one-author data subset
    python main.py --run-name modernbert_one_author --data-subset one_author
    
    # Train with custom description
    python main.py --run-name experiment_v2 --description "Testing with 10 epochs"

    After training, use the models in classification:
    from backend.src.steering.classification.classify_generated import run_classification_from_configs
    results = run_classification_from_configs(
        generation_run_dir="data/steering/tests/my_run",
        classifier_run_name="my_experiment"
    )
            """
    )
    
    parser.add_argument(
        "--run-name",
        type=str,
        required=True,
        help="Name for this training run. Models will be saved to models/{run_name}/"
    )
    
    parser.add_argument(
        "--description",
        type=str,
        default="",
        help="Description of this training run"
    )
    
    parser.add_argument(
        "--transformers",
        nargs="+",
        choices=list(AVAILABLE_TRANSFORMERS.keys()),
        default=["modern_bert"],
        help="Transformer to use (default: modern_bert)"
    )
    
    parser.add_argument(
        "--models",
        nargs="+",
        choices=list(AVAILABLE_MODELS.keys()),
        default=["modern_bert"],
        help="Model to use (default: modern_bert)"
    )
    
    parser.add_argument(
        "--data-subset",
        type=str,
        default="one_author",
        choices=["all", "one_author"],
        help="Data subset to use for training (default: one_author)"
    )
    
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Proportion of data to use for testing (default: 0.2)"
    )
    
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)"
    )
    
    parser.add_argument(
        "--base-dir",
        type=str,
        default="backend/src/steering/classification/models",
        help="Base directory for saving models (default: backend/src/steering/classification/models)"
    )
    
    return parser.parse_args()


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    args = parse_args()
    
    # Multiple combinations of transformer/model should use multiple runs
    if len(args.transformers) > 1 or len(args.models) > 1:
        print("Warning: Multiple transformers/models specified. Running each as separate experiment.")
        for transformer in args.transformers:
            for model in args.models:
                # Check if valid combination
                if model not in VALID_COMBINATIONS.get(transformer, []):
                    print(f"Skipping invalid combination: {transformer} + {model}")
                    continue
                
                run_name = f"{args.run_name}_{transformer}_{model}"
                config = ClassifierTrainingConfig(
                    run_name=run_name,
                    description=args.description,
                    base_dir=Path(args.base_dir),
                    transformer_name=transformer,
                    transformer_params={"ngram_range": (1, 3), "max_features": 5000},
                    model_name=model,
                    data_subset=args.data_subset,
                    test_size=args.test_size,
                    random_state=args.random_state,
                )
                run_training(config)
    else:

        # Validate combination
        valid_models = VALID_COMBINATIONS.get(args.transformers[0], [])
        if args.models[0] not in valid_models:
            raise ValueError(
                f"Invalid combination: {args.transformers[0]} + {args.models[0]}. "
                f"Valid models for {args.transformers[0]}: {valid_models}"
            )
        # Single transformer/model - use run_name as-is
        config = ClassifierTrainingConfig(
            run_name=args.run_name,
            description=args.description,
            base_dir=Path(args.base_dir),
            transformer_name=args.transformers[0],
            transformer_params={"ngram_range": (1, 3)},
            model_name=args.models[0],
            data_subset=args.data_subset,
            test_size=args.test_size,
            random_state=args.random_state,
        )
        run_training(config)
