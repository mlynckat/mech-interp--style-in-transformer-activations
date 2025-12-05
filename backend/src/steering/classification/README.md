# Classification Module

This module provides functionality for training author classification models and applying them to generated (and potentially steered) texts. The module supports multiple data transformation methods and classification algorithms.

## Main Entry Points

### 1. `main.py` - Model Training

The main entry point for training classification models on news article data.

**Purpose**: Trains binary classification models to identify whether a text was written by a specific author. The script supports multiple combinations of data transformers and classification models.

**Usage**:
```bash
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
```

**Features**:
- Supports multiple data transformers: TF-IDF, Sentence Embeddings, ModernBERT
- Supports multiple classification models: Logistic Regression, Random Forest, SGD, ModernBERT
- Performs train/test split with stratification
- Trains binary classifiers for each author (one-vs-rest)
- Generates confusion matrices and saves them as images
- Saves trained models to `models/` directory
- Saves results (metrics, aggregated CSV) to `results/` directory

**Output**:
- Trained models saved in `models/` directory
- Results JSON files with metrics per author
- Aggregated CSV files with all metrics
- Confusion matrix visualizations

### 2. `classify_generated.py` - Classification on Generated Texts

The main entry point for classifying generated and steered texts using pre-trained models.

**Purpose**: Applies trained classification models to generated texts (baseline or steered) to evaluate how well the models can identify the target author in generated content.

**Usage**:
```python
from classify_generated import ClassificationConfig, run_classification_pipeline

# Default configuration (ModernBERT)
config = ClassificationConfig(
    path_to_generated_texts_template="data/steering/tests/generated_texts__sae_baseline__{}.json",
)

# Or use different model/transformer
config = ClassificationConfig(
    classification_model_class=LogisticRegressionModel,
    data_transformer_class=TFIDFTransformer,
    path_to_generated_texts_template="data/steering/tests/generated_texts__baseline__{}.json",
)

results = run_classification_pipeline(config)
```

**Features**:
- Loads pre-trained models from `models/` directory
- Processes generated texts from JSON files
- Computes predictions for all target authors
- Calculates metrics (precision, recall, F1) per initial author and overall
- Generates heatmap visualizations showing predictions across authors
- Automatically extracts data type from filename (e.g., "sae_baseline", "steered__heuristic")

**Output**:
- JSON file with classification metrics per author
- CSV file with all predictions
- Heatmap visualization (PNG) showing prediction patterns

## Supporting Scripts

### `read_data.py`
Handles data loading from news JSON files. The `DataReader` class reads news articles from JSON format, filters by category (currently supports "politics"), applies length and document count constraints, and returns features (text) and targets (author labels) as pandas Series.

**Key Features**:
- Reads from `data/news_authormix_datasets/`
- Filters documents by minimum length
- Limits documents per author
- Returns data for specified author list

### `data_transformers.py`
Provides data transformation classes that convert raw text into features suitable for classification models.

**Available Transformers**:
- **`TFIDFTransformer`**: Converts text to TF-IDF vectors using scikit-learn's TfidfVectorizer
- **`SentenceEmbeddingTransformer`**: Converts text to sentence embeddings using SentenceTransformer models (default: "all-MiniLM-L6-v2")
- **`ModernBertTransformer`**: Tokenizes text using ModernBERT tokenizer, returning input_ids, attention_mask, and token_type_ids

All transformers inherit from `BaseDataTransformer` and implement `fit()` and `transform()` methods. They can save and load their fitted state for reuse.

### `classification_models.py`
Provides classification model wrappers that implement a consistent interface for training and prediction.

**Available Models**:
- **`LogisticRegressionModel`**: Wrapper around scikit-learn's LogisticRegression
- **`RandomForestModel`**: Wrapper around scikit-learn's RandomForestClassifier
- **`SGDClassifierModel`**: Wrapper around scikit-learn's SGDClassifier
- **`ModernBertClassifierModel`**: Fine-tunes ModernBERT for sequence classification using Hugging Face Transformers

All models inherit from `BaseModel` and implement `fit()`, `predict()`, and `get_metrics()` methods. They save trained models to disk and can load them for inference.

### `pipeline.py`
Orchestrates the machine learning pipeline by coordinating data transformation and model training/prediction.

**`MLPipeline` Class**:
- **`run()`**: Executes the full training pipeline (transform → fit → evaluate)
- **`predict_samples()`**: Uses pre-trained models to make predictions on new data

The pipeline handles the workflow of transforming data, training models, and computing metrics, ensuring that transformers are fitted before models are trained and that saved models are loaded correctly during inference.

## Workflow

1. **Training Phase** (`main.py`):
   - Load news data using `DataReader`
   - Split into train/test sets
   - For each author, train binary classifiers using selected transformer/model combinations
   - Save models and evaluation metrics

2. **Inference Phase** (`classify_generated.py`):
   - Load pre-trained models
   - Read generated texts from JSON files
   - Transform texts using the same transformer used during training
   - Make predictions using the trained models
   - Compute metrics and generate visualizations

## Directory Structure

```
classification/
├── main.py                    # Training entry point
├── classify_generated.py     # Inference entry point
├── read_data.py              # Data loading utilities
├── data_transformers.py      # Data transformation classes
├── classification_models.py  # Classification model classes
├── pipeline.py               # Pipeline orchestration
├── models/                   # Saved trained models
└── results/                  # Training results and metrics
```

## Dependencies

- pandas
- scikit-learn
- transformers (Hugging Face)
- sentence-transformers
- torch
- matplotlib
- seaborn

