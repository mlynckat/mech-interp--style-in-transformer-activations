# main.py
"""
Main execution script for ML pipeline
Run multiple experiments with different transformers and models
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
from backend.src.steering.classification.data_transformers import TFIDFTransformer, SentenceEmbeddingTransformer, ModernBertTransformer
from backend.src.steering.classification.classification_models import LogisticRegressionModel, RandomForestModel, SGDClassifierModel, ModernBertClassifierModel
from backend.src.steering.classification.pipeline import MLPipeline

# Configuration: Map string names to classes
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



def save_results(base_dir: Path, results: Dict[str, Any], suffix: str = None):
    """Save results to JSON file and create aggregated results dataframe test results"""
    datetime_suffix = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    filepath = base_dir / f"results_{datetime_suffix}_{suffix}.json"
    with open(filepath, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"Results saved to: {filepath}")

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
    results_df.to_csv(base_dir / f"results_aggregated_{datetime_suffix}_{suffix}.csv", index=False)


def get_confusion_matrix(predicted_labels: list, true_labels: list):
    """Get confusion matrix"""

    true_labels_unique = list(set(true_labels))
    predicted_labels_unique = list(set(predicted_labels))

    df_conf_matrix = pd.DataFrame(0, index=true_labels_unique, columns=predicted_labels_unique)
    print(df_conf_matrix)
    for i in range(len(true_labels)):
        df_conf_matrix.loc[true_labels[i], predicted_labels[i]] += 1  

    return df_conf_matrix


def run_single_experiment(
    path_to_models: Path,
    path_to_results: Path,
    data_transformer_class,
    classification_model_class,
    data_transformer_params: dict = None,
    classification_model_params: dict = None,
    test_size: float = 0.2,
    random_state: int = 42
):
    """Run a single pipeline experiment"""

    results = {}
    
    # 1. Read data
    X, y = DataReader.read_news_json_data()

    print(f"Data shape: X: {X.shape}, y: {y.shape}")
    print(f"Data unique authors: {y.unique()}")
    print(f"Data author distribution: {y.value_counts()}")

    X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=test_size,
            random_state=random_state,
            stratify=y,
        )

    # Transform y to binary classification
    for author in y.unique():

        y_train_author = (y_train == author).astype(int)
        y_test_author = (y_test == author).astype(int)

        print(f"Train data shape: X_train: {X_train.shape}, y_train: {y_train.shape}")
        print(f"Test data shape: X_test: {X_test.shape}, y_test: {y_test.shape}")
        print(f"Train data author distribution: {y_train_author.value_counts()}")
        print(f"Test data author distribution: {y_test_author.value_counts()}")

        results[author] = {}

        # 2. Initialize data transformation and model
        data_transformer = data_transformer_class(path_to_models=path_to_models, author=author, **(data_transformer_params or {}))
        classification_model = classification_model_class(path_to_models=path_to_models, author=author, data_transformer=data_transformer, **(classification_model_params or {}))

        # 3. Create and run pipeline
        pipeline = MLPipeline(data_transformer, classification_model)
        results_per_author = pipeline.run(X_train, y_train_author, X_test, y_test_author)

        predicted_labels = results_per_author["predicted_labels"].map({1: author, 0: "other"}).values.tolist()

        confusion_matrix = get_confusion_matrix(predicted_labels, y_test.values.tolist())
        sns.heatmap(confusion_matrix, annot=True, fmt='d')
        plt.title(f"Confusion Matrix for {author} {data_transformer_class.__name__} {classification_model_class.__name__}")
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.xticks(rotation=45)
        plt.savefig(path_to_results / f"confusion_matrix_{author}_{data_transformer_class.__name__}_{classification_model_class.__name__}.png")
        plt.tight_layout()
        plt.close()

        results[author].update(results_per_author)
      
    return results


def run_experiments(
    transformers: Optional[List[str]] = None,
    models: Optional[List[str]] = None,
    test_size: float = 0.2,
    random_state: int = 42,
    data_transformer_params: Optional[Dict[str, Dict[str, Any]]] = None,
    classification_model_params: Optional[Dict[str, Dict[str, Any]]] = None,
    suffix: Optional[str] = None
):
    """
    Run multiple experiments with different configurations
    
    Args:
        transformers: List of transformer names to use. If None, uses all valid combinations.
        models: List of model names to use. If None, uses all valid combinations.
        test_size: Proportion of data to use for testing
        random_state: Random seed for reproducibility
        data_transformer_params: Dict mapping transformer names to their parameter dicts
        classification_model_params: Dict mapping model names to their parameter dicts
        suffix: Optional suffix for result files
    """
    # Create base directory for results in the current directory
    results_dir = Path(__file__).parent / "results"
    models_dir = Path(__file__).parent / "models"

    results_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    # Determine which combinations to run
    if transformers is None:
        transformers = list(AVAILABLE_TRANSFORMERS.keys())
    if models is None:
        models = list(AVAILABLE_MODELS.keys())
    
    # Validate transformer names
    invalid_transformers = [t for t in transformers if t not in AVAILABLE_TRANSFORMERS]
    if invalid_transformers:
        raise ValueError(f"Invalid transformer names: {invalid_transformers}. Available: {list(AVAILABLE_TRANSFORMERS.keys())}")
    
    # Validate model names
    invalid_models = [m for m in models if m not in AVAILABLE_MODELS]
    if invalid_models:
        raise ValueError(f"Invalid model names: {invalid_models}. Available: {list(AVAILABLE_MODELS.keys())}")
    
    # Build combinations to run
    combinations_to_run = []
    for transformer_name in transformers:
        valid_models_for_transformer = VALID_COMBINATIONS.get(transformer_name, models)
        for model_name in models:
            if model_name in valid_models_for_transformer:
                combinations_to_run.append((transformer_name, model_name))
            else:
                print(f"Warning: Skipping invalid combination {transformer_name} + {model_name}")
    
    if not combinations_to_run:
        raise ValueError("No valid combinations found. Check transformer and model selections.")
    
    print(f"Running {len(combinations_to_run)} experiment(s):")
    for trans, mod in combinations_to_run:
        print(f"  - {trans} + {mod}")
    
    results = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(float)))))
    
    # Run experiments
    for transformer_name, model_name in combinations_to_run:
        data_transformer_class = AVAILABLE_TRANSFORMERS[transformer_name]
        classification_model_class = AVAILABLE_MODELS[model_name]
        
        transformer_params = (data_transformer_params or {}).get(transformer_name, {})
        model_params = (classification_model_params or {}).get(model_name, {})
        
        print(f"\n{'='*60}")
        print(f"Running: {transformer_name} + {model_name}")
        print(f"{'='*60}")
        
        try:
            results[data_transformer_class.__name__][classification_model_class.__name__] = run_single_experiment(
                path_to_models=models_dir,
                path_to_results=results_dir,
                data_transformer_class=data_transformer_class,
                classification_model_class=classification_model_class,
                data_transformer_params=transformer_params,
                classification_model_params=model_params,
                test_size=test_size,
                random_state=random_state,
            )
        except Exception as e:
            print(f"Error running {transformer_name} + {model_name}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Generate suffix if not provided
    if suffix is None:
        suffix = "_".join(sorted(set([t for t, _ in combinations_to_run])))
    
    save_results(results_dir, results, suffix=suffix)
    print(f"\n{'='*60}")
    print(f"All experiments completed! Results saved with suffix: {suffix}")
    print(f"{'='*60}")


def parse_args():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(
        description="Run ML classification experiments with different transformers and models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run all valid combinations
  python main.py
  
  # Run specific transformer with all compatible models
  python main.py --transformers modern_bert
  
  # Run specific transformer and model
  python main.py --transformers tfidf --models logistic_regression
  
  # Run multiple transformers and models
  python main.py --transformers tfidf sentence_embedding --models logistic_regression random_forest
  
  # Run with custom test size and random state
  python main.py --test-size 0.3 --random-state 123
        """
    )
    
    parser.add_argument(
        "--transformers",
        nargs="+",
        choices=list(AVAILABLE_TRANSFORMERS.keys()),
        default=None,
        help="List of transformers to use. If not specified, runs all valid combinations."
    )
    
    parser.add_argument(
        "--models",
        nargs="+",
        choices=list(AVAILABLE_MODELS.keys()),
        default=None,
        help="List of models to use. If not specified, runs all valid combinations."
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
        "--suffix",
        type=str,
        default=None,
        help="Optional suffix for result files"
    )
    
    return parser.parse_args()
  


if __name__ == "__main__":
    args = parse_args()
    run_experiments(
        transformers=args.transformers,
        models=args.models,
        test_size=args.test_size,
        random_state=args.random_state,
        suffix=args.suffix
    )