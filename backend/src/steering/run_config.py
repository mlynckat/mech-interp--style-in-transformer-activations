"""
Shared configuration for the complete ML pipeline.

This module provides a unified configuration system for:
1. Classifier training (main.py)
2. Text generation - baseline and steered (steered_text_generation.py)
3. Classification and evaluation of generated texts (classify_generated.py)

Configuration Classes:
- ClassifierTrainingConfig: Training settings for classifiers
- RunConfig: Text generation run settings
- GeneratedTextsPaths: Path utilities for generated texts

The pipeline flow:
    1. Train classifiers (main.py) → saves ClassifierTrainingConfig
    2. Generate texts (steered_text_generation.py) → saves RunConfig
    3. Classify generated texts (classify_generated.py) → loads both configs

Example:
    # Train classifiers with custom name
    training_config = ClassifierTrainingConfig(
        run_name="modernbert_10epochs_all_data",
        transformer_name="modern_bert",
        model_name="modern_bert",
        epochs=10,
        data_subset="all"
    )
    
    # Later, use in classification
    classification_config = ClassificationConfig.from_configs(
        generation_run_dir="data/steering/tests/my_generation_run",
        classifier_run_name="modernbert_10epochs_all_data"
    )
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Dict, Any
from datetime import datetime
import json
import os


# ============================================================================
# CONSTANTS
# ============================================================================

DEFAULT_BASE_DIR = Path("data/steering/tests")
DEFAULT_MODEL_NAME = "google/gemma-2-9b-it"
DEFAULT_SAE_RELEASE = "gemma-scope-9b-pt-res"
DEFAULT_SAE_ID = "layer_15/width_16k/average_l0_131"
DEFAULT_AUTHORS = ["Sam Levine", "Paige Lavender", "Lee Moran", "Amanda Terkel"]

# Default base directory for classifier models
DEFAULT_CLASSIFIER_BASE_DIR = Path("backend/src/steering/classification/models")

# Default path for SAE diffs
DEFAULT_SAE_DIFFS_DIR = Path("data/steering/sae_diffs")


# ============================================================================
# FILE NAMING UTILITIES
# ============================================================================

class GeneratedTextsPaths:
    """
    Utility class for generating consistent file paths for generated texts.
    
    File naming convention:
    - Baseline: generated_texts__sae_baseline__{author}.json
    - Steered: generated_texts__steered__{method}__{author}.json
    
    Directory structure:
    - {base_dir}/{run_name}/generated_texts__...json
    - {base_dir}/{run_name}/classification_results/...
    """
    
    def __init__(
        self,
        base_dir: Path,
        run_name: str,
        steering_method: Optional[str] = None,
        is_baseline: bool = False
    ):
        """
        Initialize paths generator.
        
        Args:
            base_dir: Base directory for all runs (e.g., data/steering/tests)
            run_name: Name of this run (used as subdirectory)
            steering_method: Steering method name (e.g., "heuristic", "projected_gradient")
            is_baseline: Whether this is a baseline run (no steering)
        """
        self.base_dir = Path(base_dir)
        self.run_name = run_name
        self.steering_method = steering_method
        self.is_baseline = is_baseline
        
        # Run output directory
        self.run_dir = self.base_dir / run_name
        
        # Classification results subdirectory
        self.classification_dir = self.run_dir / "classification_results"
    
    def get_output_dir(self) -> Path:
        """Get the output directory for this run."""
        return self.run_dir
    
    def get_classification_dir(self) -> Path:
        """Get the classification results directory for this run."""
        return self.classification_dir
    
    def get_generated_text_filename(self, author: str) -> str:
        """
        Get the filename for generated texts for a specific author.
        
        Args:
            author: Author name
            
        Returns:
            Filename string (without path)
        """
        if self.is_baseline:
            return f"generated_texts__sae_baseline__{author}.json"
        else:
            return f"generated_texts__steered__{self.steering_method}__{author}.json"
    
    def get_generated_text_path(self, author: str) -> Path:
        """
        Get full path to generated texts file for a specific author.
        
        Args:
            author: Author name
            
        Returns:
            Full Path object
        """
        return self.run_dir / self.get_generated_text_filename(author)
    
    def get_generated_texts_template(self) -> str:
        """
        Get template string for generated texts paths (with {} placeholder for author).
        
        This is compatible with classify_generated.py's path_to_generated_texts_template.
        
        Returns:
            Template string like "data/steering/tests/run_name/generated_texts__steered__heuristic__{}.json"
        """
        if self.is_baseline:
            return str(self.run_dir / "generated_texts__sae_baseline__{}.json")
        else:
            return str(self.run_dir / f"generated_texts__steered__{self.steering_method}__{{}}.json")
    
    def ensure_directories_exist(self):
        """Create all necessary directories."""
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.classification_dir.mkdir(parents=True, exist_ok=True)


# ============================================================================
# CLASSIFIER TRAINING CONFIGURATION
# ============================================================================

@dataclass
class ClassifierTrainingConfig:
    """
    Configuration for classifier training runs (main.py).
    
    This config captures all settings used during classifier training, allowing
    for reproducibility and seamless integration with the classification pipeline.
    
    Directory structure:
        {base_dir}/{run_name}/
            ├── training_config.json      # This config
            ├── {transformer}_{model}_{author}.pkl  # Trained models
            └── results/                  # Training results
    
    Example:
        config = ClassifierTrainingConfig(
            run_name="modernbert_10epochs",
            transformer_name="modern_bert",
            model_name="modern_bert",
            epochs=10,
            data_subset="all",
            description="ModernBERT trained for 10 epochs on all data"
        )
        
        # Train models...
        config.save_config()
        
        # Later, load for classification
        loaded_config = ClassifierTrainingConfig.load_config(
            "backend/src/steering/classification/models/modernbert_10epochs"
        )
    """
    
    # Run identification
    run_name: str = "default_classifier_run"
    description: str = ""
    
    # Base directory for all classifier runs
    base_dir: Path = field(default_factory=lambda: DEFAULT_CLASSIFIER_BASE_DIR)
    
    # Transformer and model settings
    transformer_name: str = "modern_bert"  # "tfidf", "sentence_embedding", "modern_bert"
    model_name: str = "modern_bert"  # "logistic_regression", "random_forest", "sgd", "modern_bert"
    
    # Training parameters
    data_subset: str = "all"  # "all" or "one_author"
    test_size: float = 0.2
    random_state: int = 42
    
    # Model-specific parameters (optional)
    epochs: Optional[int] = None  # For neural network models
    learning_rate: Optional[float] = None
    batch_size: Optional[int] = None
    
    # Additional parameters stored as dict for flexibility
    transformer_params: Dict[str, Any] = field(default_factory=dict)
    model_params: Dict[str, Any] = field(default_factory=dict)
    
    # Author list
    author_list: List[str] = field(default_factory=lambda: DEFAULT_AUTHORS.copy())
    
    def __post_init__(self):
        """Validate and process configuration after initialization."""
        self.base_dir = Path(self.base_dir)
    
    @property
    def models_dir(self) -> Path:
        """Get the directory where trained models are saved."""
        return self.base_dir / self.run_name
    
    @property
    def results_dir(self) -> Path:
        """Get the directory where training results are saved."""
        return self.models_dir / "results"
    
    def get_model_path(self, author: str, extension: str = ".pkl") -> Path:
        """
        Get the path for a specific author's trained model.
        
        Args:
            author: Author name
            extension: File extension (default: .pkl)
            
        Returns:
            Path to the model file
        """
        # Use a consistent naming convention
        safe_author = author.replace(" ", "_")
        return self.models_dir / f"{self.transformer_name}_{self.model_name}_{safe_author}{extension}"
    
    def ensure_directories_exist(self):
        """Create all necessary directories."""
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir.mkdir(parents=True, exist_ok=True)
    
    def save_config(self, path: Optional[Path] = None):
        """
        Save this configuration to a JSON file.
        
        Args:
            path: Path to save to. If None, saves to models_dir/training_config.json
        """
        if path is None:
            self.ensure_directories_exist()
            path = self.models_dir / "training_config.json"
        
        config_dict = {
            "run_name": self.run_name,
            "description": self.description,
            "base_dir": str(self.base_dir),
            "transformer_name": self.transformer_name,
            "model_name": self.model_name,
            "data_subset": self.data_subset,
            "test_size": self.test_size,
            "random_state": self.random_state,
            "epochs": self.epochs,
            "learning_rate": self.learning_rate,
            "batch_size": self.batch_size,
            "transformer_params": self.transformer_params,
            "model_params": self.model_params,
            "author_list": self.author_list,
            "saved_at": datetime.utcnow().isoformat() + "Z"
        }
        
        with open(path, "w") as f:
            json.dump(config_dict, f, indent=2)
        
        print(f"Saved classifier training config to: {path}")
    
    @classmethod
    def load_config(cls, path_or_dir: Path) -> "ClassifierTrainingConfig":
        """
        Load configuration from a JSON file or directory.
        
        Args:
            path_or_dir: Path to config file or directory containing training_config.json
            
        Returns:
            ClassifierTrainingConfig instance
        """
        path = Path(path_or_dir)
        if path.is_dir():
            path = path / "training_config.json"
        
        with open(path, "r") as f:
            config_dict = json.load(f)
        
        # Remove metadata fields
        config_dict.pop("saved_at", None)
        
        # Convert base_dir back to Path
        if config_dict.get("base_dir"):
            config_dict["base_dir"] = Path(config_dict["base_dir"])
        
        return cls(**config_dict)
    
    @classmethod
    def from_run_name(cls, run_name: str, base_dir: Optional[Path] = None) -> "ClassifierTrainingConfig":
        """
        Load configuration by run name.
        
        Args:
            run_name: Name of the training run
            base_dir: Base directory (uses default if None)
            
        Returns:
            ClassifierTrainingConfig instance
        """
        base = base_dir or DEFAULT_CLASSIFIER_BASE_DIR
        return cls.load_config(base / run_name)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary for logging/display."""
        return {
            "run_name": self.run_name,
            "description": self.description,
            "transformer_name": self.transformer_name,
            "model_name": self.model_name,
            "data_subset": self.data_subset,
            "test_size": self.test_size,
            "random_state": self.random_state,
            "epochs": self.epochs,
            "learning_rate": self.learning_rate,
            "batch_size": self.batch_size,
            "models_dir": str(self.models_dir),
        }


# ============================================================================
# RUN CONFIGURATION (TEXT GENERATION)
# ============================================================================

@dataclass
class RunConfig:
    """
    Complete configuration for a text generation and classification run.
    
    This config is used by both:
    - steered_text_generation.py: For generating steered/baseline texts
    - classify_generated.py: For classifying and evaluating generated texts
    
    Example usage:
        # Create config for a steered run
        config = RunConfig(
            run_name="experiment_v1",
            steering_method="heuristic",
            n_shap_features=24
        )
        
        # Use in generation
        paths = config.get_paths()
        output_path = paths.get_generated_text_path("Sam Levine")
        
        # Use in classification
        template = paths.get_generated_texts_template()
    """
    
    # Run identification
    run_name: str = "default_run"
    description: str = ""
    
    # Base directory for all runs
    base_dir: Path = field(default_factory=lambda: DEFAULT_BASE_DIR)
    
    # Model settings
    model_name: str = DEFAULT_MODEL_NAME
    sae_release: str = DEFAULT_SAE_RELEASE
    sae_id: str = DEFAULT_SAE_ID
    
    # Steering configuration
    is_baseline: bool = False  # If True, no steering is applied
    steering_method: Optional[str] = None  # "heuristic" or "projected_gradient"
    n_shap_features: int = 24
    
    # Steering hyperparameters
    target_layer: int = 15
    alpha: float = 0.1
    lambda_rec: float = 10.0
    mu_norm: float = 0.01
    max_iterations: int = 100
    target_confidence: float = 0.7
    num_sae_features: int = 16384
    
    # Early stopping for projected gradient optimization
    early_stop_patience: int = 5  # Stop if no improvement for N iterations
    early_stop_min_delta: float = 1e-5  # Minimum change to count as improvement
    
    # Generation parameters
    max_new_tokens: int = 500
    temperature: float = 0.7
    top_k: Optional[int] = None  # If set, only sample from top-k most likely tokens
    top_p: Optional[float] = None  # If set, use nucleus sampling with this probability
    
    # Memory/Quality tradeoff for prompt steering
    # Options: "skip" (don't steer prompt, most memory efficient),
    #          "last" (steer last position only), 
    #          "all" (steer all positions, may OOM on long prompts),
    #          "chunk_N" (steer last N positions, e.g., "chunk_32")
    prompt_steering_mode: str = "skip"
    
    # Author list
    author_list: List[str] = field(default_factory=lambda: DEFAULT_AUTHORS.copy())
    
    # Input paths
    input_prompts_file: Optional[Path] = None
    path_to_logreg_models: Optional[Path] = None
    path_to_most_important_features: Optional[Path] = None
    path_to_sae_diffs: Optional[Path] = None  # For SAE-diff steering
    
    # Cached paths object
    _paths: Optional[GeneratedTextsPaths] = field(default=None, repr=False)
    
    def __post_init__(self):
        """Validate and process configuration after initialization."""
        self.base_dir = Path(self.base_dir)
        
        # Validate steering configuration
        if not self.is_baseline and self.steering_method is None:
            raise ValueError("steering_method must be set when is_baseline=False")
        
        if self.is_baseline and self.steering_method is not None:
            # Baseline runs shouldn't have a steering method
            self.steering_method = None
        
        # Set default paths if not provided
        if self.input_prompts_file is None:
            self.input_prompts_file = self.base_dir / "prompts_test_data__detailed.json"
        
        if self.path_to_logreg_models is None:
            self.path_to_logreg_models = Path(
                "data/output_data/news/politics/google_gemma-2-9b-it/"
                "prepare_features_for_steering/feature_selection_aggregated"
            )
        
        if self.path_to_most_important_features is None:
            self.path_to_most_important_features = (
                self.path_to_logreg_models / 
                f"most_important_features__res__{self.target_layer}.json"
            )
        
        if self.path_to_sae_diffs is None:
            self.path_to_sae_diffs = DEFAULT_SAE_DIFFS_DIR
    
    def get_paths(self) -> GeneratedTextsPaths:
        """
        Get the paths utility for this run.
        
        Returns:
            GeneratedTextsPaths instance configured for this run
        """
        if self._paths is None:
            self._paths = GeneratedTextsPaths(
                base_dir=self.base_dir,
                run_name=self.run_name,
                steering_method=self.steering_method,
                is_baseline=self.is_baseline
            )
        return self._paths
    
    def get_generated_texts_template(self) -> str:
        """
        Convenience method to get the template for generated texts paths.
        
        Returns:
            Template string compatible with classify_generated.py
        """
        return self.get_paths().get_generated_texts_template()
    
    def get_output_dir(self) -> Path:
        """
        Convenience method to get the output directory.
        
        Returns:
            Path to output directory
        """
        return self.get_paths().get_output_dir()
    
    def get_classification_dir(self) -> Path:
        """
        Convenience method to get the classification results directory.
        
        Returns:
            Path to classification results directory
        """
        return self.get_paths().get_classification_dir()
    
    def save_config(self, path: Optional[Path] = None):
        """
        Save this configuration to a JSON file.
        
        Args:
            path: Path to save to. If None, saves to run_dir/run_config.json
        """
        if path is None:
            paths = self.get_paths()
            paths.ensure_directories_exist()
            path = paths.run_dir / "run_config.json"
        
        config_dict = {
            "run_name": self.run_name,
            "description": self.description,
            "base_dir": str(self.base_dir),
            "model_name": self.model_name,
            "sae_release": self.sae_release,
            "sae_id": self.sae_id,
            "is_baseline": self.is_baseline,
            "steering_method": self.steering_method,
            "n_shap_features": self.n_shap_features,
            "target_layer": self.target_layer,
            "alpha": self.alpha,
            "lambda_rec": self.lambda_rec,
            "mu_norm": self.mu_norm,
            "max_iterations": self.max_iterations,
            "target_confidence": self.target_confidence,
            "num_sae_features": self.num_sae_features,
            "early_stop_patience": self.early_stop_patience,
            "early_stop_min_delta": self.early_stop_min_delta,
            "max_new_tokens": self.max_new_tokens,
            "temperature": self.temperature,
            "top_k": self.top_k,
            "top_p": self.top_p,
            "author_list": self.author_list,
            "input_prompts_file": str(self.input_prompts_file) if self.input_prompts_file else None,
            "path_to_logreg_models": str(self.path_to_logreg_models) if self.path_to_logreg_models else None,
            "path_to_most_important_features": str(self.path_to_most_important_features) if self.path_to_most_important_features else None,
            "path_to_sae_diffs": str(self.path_to_sae_diffs) if self.path_to_sae_diffs else None,
            "saved_at": datetime.utcnow().isoformat() + "Z"
        }
        
        with open(path, "w") as f:
            json.dump(config_dict, f, indent=2)
        
        print(f"Saved run config to: {path}")
    
    @classmethod
    def load_config(cls, path: Path) -> "RunConfig":
        """
        Load configuration from a JSON file.
        
        Args:
            path: Path to config file
            
        Returns:
            RunConfig instance
        """
        with open(path, "r") as f:
            config_dict = json.load(f)
        
        # Remove metadata fields
        config_dict.pop("saved_at", None)
        
        # Convert paths back to Path objects
        if config_dict.get("base_dir"):
            config_dict["base_dir"] = Path(config_dict["base_dir"])
        if config_dict.get("input_prompts_file"):
            config_dict["input_prompts_file"] = Path(config_dict["input_prompts_file"])
        if config_dict.get("path_to_logreg_models"):
            config_dict["path_to_logreg_models"] = Path(config_dict["path_to_logreg_models"])
        if config_dict.get("path_to_most_important_features"):
            config_dict["path_to_most_important_features"] = Path(config_dict["path_to_most_important_features"])
        if config_dict.get("path_to_sae_diffs"):
            config_dict["path_to_sae_diffs"] = Path(config_dict["path_to_sae_diffs"])
        
        return cls(**config_dict)
    
    @classmethod
    def from_run_dir(cls, run_dir: Path) -> "RunConfig":
        """
        Load configuration from a run directory.
        
        Args:
            run_dir: Path to run directory containing run_config.json
            
        Returns:
            RunConfig instance
        """
        config_path = Path(run_dir) / "run_config.json"
        return cls.load_config(config_path)


# ============================================================================
# PREDEFINED CONFIGURATIONS
# ============================================================================

def create_baseline_config(
    run_name: str = "baseline",
    author_list: Optional[List[str]] = None,
    **kwargs
) -> RunConfig:
    """
    Create a configuration for baseline (no steering) generation.
    
    Args:
        run_name: Name for this run
        author_list: List of authors (uses defaults if None)
        **kwargs: Additional RunConfig parameters
        
    Returns:
        RunConfig for baseline generation
    """
    return RunConfig(
        run_name=run_name,
        is_baseline=True,
        steering_method=None,
        author_list=author_list or DEFAULT_AUTHORS.copy(),
        **kwargs
    )


def create_heuristic_steering_config(
    run_name: str,
    n_shap_features: int = 24,
    alpha: float = 0.1,
    author_list: Optional[List[str]] = None,
    **kwargs
) -> RunConfig:
    """
    Create a configuration for heuristic steering generation.
    
    Args:
        run_name: Name for this run
        n_shap_features: Number of SHAP features to use
        alpha: Steering strength
        author_list: List of authors (uses defaults if None)
        **kwargs: Additional RunConfig parameters
        
    Returns:
        RunConfig for heuristic steering
    """
    return RunConfig(
        run_name=run_name,
        is_baseline=False,
        steering_method="heuristic",
        n_shap_features=n_shap_features,
        alpha=alpha,
        author_list=author_list or DEFAULT_AUTHORS.copy(),
        **kwargs
    )


def create_projected_gradient_config(
    run_name: str,
    n_shap_features: int = 24,
    lambda_rec: float = 10.0,
    mu_norm: float = 0.01,
    author_list: Optional[List[str]] = None,
    **kwargs
) -> RunConfig:
    """
    Create a configuration for projected gradient steering generation.
    
    Args:
        run_name: Name for this run
        n_shap_features: Number of SHAP features to use
        lambda_rec: Reconstruction loss weight
        mu_norm: Norm regularization weight
        author_list: List of authors (uses defaults if None)
        **kwargs: Additional RunConfig parameters
        
    Returns:
        RunConfig for projected gradient steering
    """
    return RunConfig(
        run_name=run_name,
        is_baseline=False,
        steering_method="projected_gradient",
        n_shap_features=n_shap_features,
        lambda_rec=lambda_rec,
        mu_norm=mu_norm,
        author_list=author_list or DEFAULT_AUTHORS.copy(),
        **kwargs
    )


def create_sae_diff_steering_config(
    run_name: str,
    alpha: float = 1.0,
    path_to_sae_diffs: Optional[Path] = None,
    author_list: Optional[List[str]] = None,
    **kwargs
) -> RunConfig:
    """
    Create a configuration for SAE-diff based steering generation.
    
    SAE-diff steering uses pre-computed differences between original and 
    baseline-generated text features. The diff is added to activations
    after the prompt to steer generation towards the target author's style.
    
    Args:
        run_name: Name for this run
        alpha: Steering strength multiplier for the diff vector
        path_to_sae_diffs: Path to directory containing pre-computed SAE diffs
                          (uses default if None)
        author_list: List of authors (uses defaults if None)
        **kwargs: Additional RunConfig parameters
        
    Returns:
        RunConfig for SAE-diff steering
        
    Example:
        config = create_sae_diff_steering_config(
            run_name="sae_diff_experiment",
            alpha=0.5,
            description="SAE-diff steering with alpha=0.5"
        )
    """
    return RunConfig(
        run_name=run_name,
        is_baseline=False,
        steering_method="sae_diff",
        alpha=alpha,
        path_to_sae_diffs=path_to_sae_diffs or DEFAULT_SAE_DIFFS_DIR,
        author_list=author_list or DEFAULT_AUTHORS.copy(),
        **kwargs
    )


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    # Example: Create and save a heuristic steering config
    config = create_heuristic_steering_config(
        run_name="experiment_24_features",
        n_shap_features=24,
        description="Testing heuristic steering with 24 SHAP features"
    )
    
    print("Run Configuration:")
    print(f"  Run name: {config.run_name}")
    print(f"  Output dir: {config.get_output_dir()}")
    print(f"  Generated texts template: {config.get_generated_texts_template()}")
    print(f"  Authors: {config.author_list}")
    print(f"  Steering method: {config.steering_method}")
    print(f"  N SHAP features: {config.n_shap_features}")
    
    # Example paths for each author
    paths = config.get_paths()
    for author in config.author_list:
        print(f"  {author}: {paths.get_generated_text_path(author)}")

