# SAE Feature Extraction Script

## Overview

The `get_sae_features.py` script is a comprehensive tool for extracting Sparse Autoencoder (SAE) features from transformer models using the AuthorMix or news datasets. It performs parallel processing across multiple GPUs. The script supports both dense and sparse storage formats for memory-efficient activation storage.

## 1. Goal of the Script

The primary goal of this script is to:

- **Extract SAE features** from transformer models (Gemma-2-2b, Gemma-2-9b, or Gemma-2-9b-it) across multiple layers
- **Analyze author-specific patterns** in text data using the AuthorMix or news datasets
- **Compute linguistic metrics** including cross-entropy loss and entropy for each token
- **Enable parallel processing** across multiple GPUs for efficient large-scale analysis
- **Store comprehensive results** including activations, metadata, and text data for downstream analysis
- **Support multiple storage formats** (dense or sparse) for memory-efficient activation storage

The script serves as a foundation for mechanistic interpretability research by providing detailed feature activations that can be used to understand how transformer models process different types of text and author styles.

## 2. Sequential Execution Flow

### Entry Point: `main()` Function

The script execution follows this sequential flow:

#### 2.1 Initialization Phase
```python
def main(parsed_args):
    # Construct save directory path
    model_name_safe = parsed_args.model.replace('/', '_')
    if parsed_args.category_name:
        save_dir = Path(parsed_args.output_dir) / parsed_args.dataset_name / parsed_args.category_name / model_name_safe / parsed_args.run_name
    else:
        save_dir = Path(parsed_args.output_dir) / parsed_args.dataset_name / model_name_safe / parsed_args.run_name
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Load and organize dataset by author
    if parsed_args.dataset_name == "AuthorMix":
        author_to_docs, dataset_config = load_AuthorMix_data(...)
    elif parsed_args.dataset_name == "news":
        author_to_docs, dataset_config = load_news_data(...)
```

**What happens:**
1. **Directory Setup**: Creates the output directory structure based on dataset, category (if applicable), model, and run name
2. **Data Loading**: Loads the specified dataset (AuthorMix or news) and organizes documents by author
3. **Author Assignment**: Determines the number of authors and documents per author
4. **GPU Detection**: Identifies available GPUs and their specifications
5. **Author Filtering**: Filters authors to only include those with sufficient documents

#### 2.2 Multi-Process Spawning
```python
# Distribute authors across GPUs
valid_authors = [author for author in dataset_config.author_list 
                 if len(author_to_docs[author]) >= dataset_config.max_n_docs_per_author]
author_subsets = [valid_authors[i::world_size] for i in range(world_size)]

mp.spawn(
    run_inference_on_gpu,
    args=(author_subsets, parsed_args, author_to_docs, dataset_config, batch_size, save_dir),
    nprocs=world_size,
    join=True
)
```

**What happens:**
1. **Author Filtering**: Filters authors to only include those with sufficient documents
2. **Author Distribution**: Distributes authors across GPUs using round-robin slicing
3. **Process Creation**: Spawns one process per available GPU
4. **Argument Distribution**: Passes configuration and data to each process
5. **Parallel Execution**: Each process runs `run_inference_on_gpu()` independently
6. **Synchronization**: Waits for all processes to complete
7. **Run Registration**: Registers the run in the tracking system after completion

#### 2.3 Per-Process Execution (`run_inference_on_gpu()`)

Each GPU process follows this sequence:

##### 2.3.1 Model and SAE Loading
```python
# Load transformer model
model_config = ModelConfig(model_name=parsed_args.model, layer_indices=parsed_args.layers)
model = HookedSAETransformer.from_pretrained(
    model_config.model_name,
    fold_ln=True,
    center_writing_weights=False,
    center_unembed=False,
    device=device
)
model.eval()

# Load SAEs for different layer types
saes, saes_ids = load_saes(model_config, parsed_args.layers, parsed_args.sae_features_width, device)
```

**What happens:**
1. **Model Loading**: Loads the specified transformer model (Gemma-2-2b, Gemma-2-9b, or Gemma-2-9b-it) using `HookedSAETransformer`
2. **SAE Loading**: Downloads and loads canonical pre-trained SAEs for all layer types (residual, MLP, attention) specified in the model config
3. **Device Assignment**: Moves all models to the assigned GPU device
4. **Memory Optimization**: Sets models to evaluation mode
5. **Storage Format**: Determines storage format (dense or sparse) from command-line arguments

##### 2.3.2 Author Assignment and Data Preparation
```python
# Assign authors to this GPU process
authors_for_this_gpu = author_subsets[rank]

# Process each author assigned to this GPU
for author in authors_for_this_gpu:
    docs = author_to_docs[author]
    
    # Initialize activation storage for each layer and SAE
    activations = {}
    for layer_type in saes_ids:
        activations[layer_type] = {}
        for sae_name, sae in zip(saes_ids[layer_type], saes[layer_type]):
            activations[layer_type][sae_name] = ActivationStorage(
                storage_format=storage_format,
                n_docs=dataset_config.max_n_docs_per_author,
                max_seq_len=dataset_config.max_sequence_length - 1,
                n_features=sae.cfg.d_sae
            )
    
    # Create dataset and dataloader
    dataset = MyDataset(docs, model, dataset_config.max_sequence_length, parsed_args.setting)
    loader = DataLoader(dataset, batch_size=batch_size, ...)
```

**What happens:**
1. **Author Assignment**: Each GPU process is assigned one or more authors from the pre-distributed author subsets
2. **Document Selection**: Retrieves all documents for each assigned author
3. **Storage Initialization**: Creates `ActivationStorage` objects for each layer type and SAE, supporting both dense and sparse formats
4. **Dataset Creation**: Creates a PyTorch dataset with tokenization and padding, supporting both "baseline" and "prompted" settings
5. **DataLoader Setup**: Configures batch processing with custom collation function

##### 2.3.3 Batch Processing Loop
```python
for batch_idx, batch in enumerate(tqdm(loader, desc=f"Processing {author}")):
    full_texts_batch, input_ids_batch, input_tokens_batch, prompt_length_batch = batch
    prompt_length_doc = prompt_length_batch[0]
    
    input_ids = input_ids_batch['input_ids'].to(device)
    attention_mask = input_ids_batch['attention_mask'].to(device)
    actual_lengths = (attention_mask.sum(dim=1) - prompt_length_doc).tolist()
    
    # Forward pass through model with SAEs
    full_list_saes = []
    for layer_type in model_config.layer_types:
        full_list_saes.extend(saes[layer_type])
    
    logits, cache = model.run_with_cache_with_saes(input_ids, saes=full_list_saes)
    
    # Compute metrics
    ce_loss = lm_cross_entropy_loss(logits, input_ids, attention_mask=attention_mask, per_token=True)
    entropy = compute_entropy(logits)
    
    # Extract SAE features
    for layer_type in model_config.layer_types:
        for layeri, sae_id, sae in zip(model_config.layer_indices, saes_ids[layer_type], saes[layer_type]):
            hook_name = f'{sae.cfg.metadata.hook_name}.hook_sae_acts_post'
            sae_acts = cache[hook_name][:, prompt_length_doc:].to(device)
            
            activations[layer_type][sae_id].add_batch(
                sae_acts,
                batch_doc_indices,
                actual_lengths
            )
```

**What happens for each batch:**
1. **Batch Unpacking**: Extracts text, token IDs, tokens, and prompt length information
2. **Device Transfer**: Moves tensors to the assigned GPU device
3. **Forward Pass**: Runs the model with SAEs integrated using `run_with_cache_with_saes()` to capture SAE activations directly
4. **Metric Computation**: Calculates cross-entropy loss and entropy for each token, accounting for prompt length
5. **SAE Processing**: Extracts SAE activations from cache using hook names, skipping prompt tokens
6. **Storage**: Adds batch activations to `ActivationStorage` objects (handles both dense and sparse formats)
7. **Memory Management**: Cleans up GPU memory every 5 batches

##### 2.3.4 Results Storage
```python
# Save entropy and cross-entropy loss
np.save(save_dir / f"sae_{parsed_args.setting}__{model_name_safe}__entropy__{author}.npy", entropy_author)
np.save(save_dir / f"sae_{parsed_args.setting}__{model_name_safe}__cross_entropy_loss__{author}.npy", cross_entropy_loss_author)

# Save tokens and full texts
json.dump(tokens_per_author, ...)
json.dump(full_texts_per_author, ...)

# Save activations
for layer_type in model_config.layer_types:
    for layeri, sae_id in zip(model_config.layer_indices, saes_ids[layer_type]):
        activations[layer_type][sae_id].save(
            filepath=filepath,
            author_id=author,
            doc_lengths=doc_lengths,
            layer_type=layer_type,
            layer_index=layeri,
            sae_id=sae_id,
            model_name=model_name_safe
        )
```

**What happens:**
1. **Metric Storage**: Saves entropy and cross-entropy loss as NumPy arrays
2. **Text Storage**: Saves tokenized text and full texts as JSON files
3. **Activation Storage**: Saves activations using `ActivationStorage.save()` which:
   - Creates `ActivationMetadata` objects with comprehensive information
   - Saves data in dense (`.npz`) or sparse (`.sparse.npz`) format
   - Saves metadata as `.meta.pkl` files
   - Logs file sizes and save times
4. **Memory Cleanup**: Clears GPU memory and Python variables after each author

## 3. Command Line Arguments

### Required Arguments
- **`--layers`** (list of integers): Layer indices to analyze
  - Example: `--layers 5 10 15 20`
  - Determines which transformer layers to extract features from

### Optional Arguments

#### Model Configuration
- **`--model`** (string): Transformer model to use
  - Default: `"google/gemma-2-9b-it"`
  - Choices: `["google/gemma-2-2b", "google/gemma-2-9b", "google/gemma-2-9b-it"]`
  - Determines model size and corresponding SAE repositories

- **`--sae_features_width`** (string): SAE feature count
  - Default: `"16k"`
  - Format: `"16k"`, `"32k"`, `"65k"`, `"131k"`
  - Determines the size of SAE feature spaces (converted to integer width internally)

#### Data Configuration
- **`--dataset_name`** (string): Dataset to use
  - Default: `"news"`
  - Choices: `["AuthorMix", "news"]`
  - Determines which dataset loader to use

- **`--category_name`** (string): Category for news dataset
  - Default: `"politics"`
  - Only used when `--dataset_name` is `"news"`
  - Determines which news category to process

- **`--n_docs`** (integer): Total documents to process
  - Default: `4000`
  - Limits the total number of documents across all authors

- **`--n_docs_per_author`** (integer): Documents per author
  - Default: `500`
  - Controls how many documents to process for each author
  - Authors with fewer documents are automatically filtered out

- **`--min_length_doc`** (integer): Minimum document length
  - Default: `35`
  - Filters out very short documents

- **`--exclude_authors`** (list of strings): Authors to skip
  - Default: `None`
  - Example: `--exclude_authors "author1" "author2"`

#### Processing Configuration
- **`--batch_size`** (integer): Batch size for processing
  - Default: `2`
  - Affects memory usage and processing speed
  - Should be adjusted based on GPU memory

- **`--setting`** (string): Feature extraction setting
  - Default: `"baseline"`
  - Choices: `["prompted", "baseline"]`
  - "baseline": Process text as-is
  - "prompted": Prepend style prompt to text (e.g., "The text in style of {style}: \n")

#### Storage Configuration
- **`--storage_format`** (string): Storage format for activations
  - Default: `"sparse"`
  - Choices: `["dense", "sparse"]`
  - "dense": Store full activation arrays (larger files, faster access)
  - "sparse": Store only non-zero activations (smaller files, memory-efficient)

- **`--output_dir`** (string): Base output directory
  - Default: `"data/raw_features"`
  - Base directory where results will be saved

- **`--run_name`** (string): Name of the run
  - Default: `"news_500_politics"`
  - Creates a subdirectory with this name
  - Final path: `{output_dir}/{dataset_name}/{category_name}/{model_name}/{run_name}/` (or without category_name for AuthorMix)

### Example Usage

#### AuthorMix Dataset
```bash
python get_sae_features.py \
    --model "google/gemma-2-9b-it" \
    --layers 5 10 15 20 \
    --sae_features_width "16k" \
    --dataset_name "AuthorMix" \
    --n_docs_per_author 250 \
    --batch_size 4 \
    --setting "baseline" \
    --storage_format "sparse" \
    --output_dir "data/raw_features" \
    --run_name "author_mix_experiment_1"
```

#### News Dataset
```bash
python get_sae_features.py \
    --model "google/gemma-2-9b-it" \
    --layers 5 10 15 20 \
    --sae_features_width "16k" \
    --dataset_name "news" \
    --category_name "politics" \
    --n_docs 4000 \
    --n_docs_per_author 500 \
    --batch_size 2 \
    --setting "prompted" \
    --storage_format "dense" \
    --output_dir "data/raw_features" \
    --run_name "news_politics_500"
```

## 4. Outputs: What, Where, and Shape

### Output Directory Structure
```
{output_dir}/{dataset_name}/{category_name}/{model_name}/{run_name}/
├── sae_{setting}__{model_name}__entropy__{author}.npy
├── sae_{setting}__{model_name}__cross_entropy_loss__{author}.npy
├── sae_{setting}__{model_name}__tokens__{author}.json
├── sae_{setting}__{model_name}__full_texts__{author}.json
├── sae_{setting}__{model_name}__res__activations__{author}__layer_{N}.npz (or .sparse.npz)
├── sae_{setting}__{model_name}__res__activations__{author}__layer_{N}.meta.pkl
├── sae_{setting}__{model_name}__mlp__activations__{author}__layer_{N}.npz (or .sparse.npz)
├── sae_{setting}__{model_name}__mlp__activations__{author}__layer_{N}.meta.pkl
├── sae_{setting}__{model_name}__att__activations__{author}__layer_{N}.npz (or .sparse.npz)
└── sae_{setting}__{model_name}__att__activations__{author}__layer_{N}.meta.pkl
```

**Note**: For AuthorMix dataset, the `{category_name}` level is omitted from the path.

### Output Files Description

#### 4.1 Entropy and Loss Files
- **File**: `sae_{setting}__{model_name}__entropy__{author}.npy`
- **Shape**: `(max_docs_per_author, max_seq_len-1)`
- **Data Type**: `float32`
- **Content**: Entropy values for each token position in each document (excluding prompt tokens)
- **Purpose**: Measures uncertainty in model predictions

- **File**: `sae_{setting}__{model_name}__cross_entropy_loss__{author}.npy`
- **Shape**: `(max_docs_per_author, max_seq_len-1)`
- **Data Type**: `float32`
- **Content**: Cross-entropy loss for each token position (excluding prompt tokens)
- **Purpose**: Measures prediction accuracy

#### 4.2 Text Data Files
- **File**: `sae_{setting}__{model_name}__tokens__{author}.json`
- **Format**: JSON array of token lists
- **Content**: Tokenized text for each document (excluding prompt tokens)
- **Purpose**: Provides token-level text data for analysis

- **File**: `sae_{setting}__{model_name}__full_texts__{author}.json`
- **Format**: JSON array of strings
- **Content**: Original full text of each document (without prompts)
- **Purpose**: Provides original text for reference and analysis

#### 4.3 SAE Activation Files

##### Dense Format
- **File**: `sae_{setting}__{model_name}__{layer_type}__activations__{author}__layer_{N}.npz`
- **Shape**: `(max_docs_per_author, max_seq_len-1, n_sae_features)`
- **Data Type**: `float32`
- **Content**: Full activation values for each SAE feature at each token position

##### Sparse Format
- **File**: `sae_{setting}__{model_name}__{layer_type}__activations__{author}__layer_{N}.sparse.npz`
- **Format**: Compressed Sparse Row (CSR) matrix
- **Shape**: `(max_docs_per_author * max_seq_len-1, n_sae_features)`
- **Data Type**: `float32`
- **Content**: Only non-zero activation values (memory-efficient for sparse activations)

##### Metadata Files
- **File**: `sae_{setting}__{model_name}__{layer_type}__activations__{author}__layer_{N}.meta.pkl`
- **Format**: Pickled `ActivationMetadata` object
- **Content**: Comprehensive metadata including:
  - `doc_ids`: Document index for each position
  - `tok_ids`: Token index within each document
  - `author_id`: Author identifier
  - `doc_lengths`: Actual length of each document
  - `valid_mask`: Boolean mask indicating valid (non-padding) tokens
  - `original_shape`: Original shape of activation array
  - `n_features`: Number of SAE features
  - `storage_format`: Storage format used ("dense" or "sparse")
  - `layer_type`: Type of layer (res, mlp, att)
  - `layer_index`: Layer number
  - `sae_id`: SAE identifier
  - `model_name`: Model name

**Layer Types**: `res` (residual), `mlp`, `att` (attention)

### Data Shapes and Dimensions

#### Dimension Explanations
- **`max_docs_per_author`**: Maximum number of documents processed for any author (determined at runtime)
- **`max_seq_len-1`**: Sequence length minus 1 (first token excluded from analysis)
- **`n_sae_features`**: Number of features in the SAE (e.g., 16,384 for "16k")

#### Memory Considerations
- **Dense Activation Storage**: Each dense activation file can be large (e.g., 500 docs × 359 tokens × 16k features ≈ 2.8GB)
- **Sparse Activation Storage**: Sparse files are typically much smaller, storing only non-zero values (size depends on sparsity)
- **Compression**: Files are saved with NPZ compression to reduce disk usage
- **Data Types**: Uses `float32` for both metrics and activations
- **Metadata**: Separate metadata files (`.meta.pkl`) contain indexing and validation information

## 5. ActivationStorage Class

The script uses an `ActivationStorage` class to manage memory-efficient storage of activations in either dense or sparse format.

### 5.1 Storage Formats

#### Dense Format
- Stores full activation arrays as NumPy arrays
- Shape: `(n_docs, max_seq_len, n_features)`
- Faster access for downstream analysis
- Larger memory footprint

#### Sparse Format
- Stores only non-zero activation values
- Uses Compressed Sparse Row (CSR) matrix format
- Memory-efficient for sparse activations (typical for SAEs)
- Requires conversion to dense format for some operations

### 5.2 Key Methods

- **`__init__()`**: Initializes storage based on format (dense or sparse)
- **`add_batch()`**: Accumulates activations from a batch, handling padding masks
- **`save()`**: Saves activations and metadata to disk, returns save time and file size
- **`_save_dense()`**: Saves dense format as `.npz` with metadata as `.meta.pkl`
- **`_save_sparse()`**: Saves sparse format as `.sparse.npz` with metadata as `.meta.pkl`

### 5.3 Metadata Management

The `ActivationMetadata` class stores comprehensive information about the activations:
- Document and token indices for each position
- Valid mask indicating non-padding tokens
- Layer and SAE information
- Storage format and shape information

## 6. Run Tracking System

After successful completion, the script automatically registers the run in a tracking system:

```python
tracker = get_tracker()
run_id = tracker.register_run(
    model=parsed_args.model,
    dataset=parsed_args.dataset_name,
    run_name=parsed_args.run_name,
    layers=parsed_args.layers,
    activation_path=str(save_dir),
    category=parsed_args.category_name,
    authors=dataset_config.author_list,
    storage_format=parsed_args.storage_format,
    n_docs_per_author=parsed_args.n_docs_per_author,
    min_length_doc=parsed_args.min_length_doc,
    setting=parsed_args.setting
)
```

This allows for:
- **Run Management**: Track and organize different experimental runs
- **Reproducibility**: Record all parameters used for each run
- **Data Discovery**: Find runs by model, dataset, or other criteria

## 7. Multi-GPU Utilization and Parallelization

### 5.1 Parallelization Strategy

The script uses **data parallelism** with **author-based distribution**:

```python
# Filter authors with sufficient documents
valid_authors = [author for author in dataset_config.author_list
                 if len(author_to_docs[author]) >= dataset_config.max_n_docs_per_author]

# Distribute authors across GPUs using round-robin slicing
author_subsets = [valid_authors[i::world_size] for i in range(world_size)]

# Each GPU process receives its subset
authors_for_this_gpu = author_subsets[rank]
```

#### Distribution Logic
- **Author Filtering**: Only authors with at least `n_docs_per_author` documents are included
- **Process Assignment**: Each GPU runs one process (rank 0, 1, 2, ...)
- **Author Assignment**: Authors are distributed using round-robin slicing (`[i::world_size]`)
- **Load Balancing**: If `n_gpus > n_authors`, some GPUs will process fewer authors
- **Multiple Authors per GPU**: Each GPU can process multiple authors sequentially

### 7.2 GPU Utilization Details

#### 7.2.1 Model and SAE Loading per GPU
```python
# Each GPU loads its own copy of:
device = torch.device(f"cuda:{rank}")
model = HookedSAETransformer.from_pretrained(model_config.model_name, ...).to(device)
saes, saes_ids = load_saes(model_config, layers, width, device)
# saes is a dict: {'res': [sae1, sae2, ...], 'mlp': [...], 'att': [...]}
```

**What's on each GPU:**
- **Complete Model**: Full transformer model (2B, 9B, or 9B-it parameters)
- **All SAEs**: Residual, MLP, and attention SAEs for specified layers (loaded once, reused for all authors)
- **Author Data**: Documents for assigned author(s)
- **Processing Memory**: Batch processing and activation storage (CPU memory for activations, GPU for computation)

#### 7.2.2 Memory Management per GPU
```python
# Memory monitoring and cleanup
log_gpu_memory_usage()  # Logs memory usage for current GPU
torch.cuda.empty_cache()  # Clears unused GPU memory
gc.collect()  # Python garbage collection
```

**Memory Optimization:**
- **Periodic Cleanup**: Memory cleared every 5 batches
- **Batch Processing**: Small batch sizes to fit in GPU memory
- **Activation Storage**: Stored in CPU memory using `ActivationStorage` class, transferred to GPU only for computation
- **Sparse Storage Option**: Can use sparse format to reduce memory usage for activation storage

### 7.3 Parallelization Architecture

#### 7.3.1 Process Spawning
```python
mp.spawn(
    run_inference_on_gpu,   # Function to run on each process
    args=(author_subsets, parsed_args, author_to_docs, dataset_config, batch_size, save_dir),
    nprocs=world_size,      # Number of processes = number of GPUs
    join=True                # Wait for all processes to complete
)
```

**Process Management:**
- **Independent Processes**: Each GPU runs in a separate Python process
- **No Inter-Process Communication**: Processes don't share memory or data
- **Synchronization**: `join=True` ensures all processes complete before main exits

#### 7.3.2 Data Distribution
```python
# Data is distributed at the author level
author_to_docs = {
    'author1': [(doc_idx, doc), ...],
    'author2': [(doc_idx, doc), ...],
    ...
}
```

**Distribution Strategy:**
- **Author-Level Parallelism**: Each process handles one or more complete authors
- **No Data Sharing**: Each process loads only its assigned author's data
- **Independent Output**: Each process saves its own results

### 7.4 GPU Assignment Examples

#### Example 1: 4 GPUs, 3 Authors
```
Valid authors: [author1, author2, author3]
author_subsets[0] = [author1]      # i=0, step=4: [0]
author_subsets[1] = [author2]      # i=1, step=4: [1]
author_subsets[2] = [author3]      # i=2, step=4: [2]
author_subsets[3] = []             # i=3, step=4: [3] (out of range)

GPU 0 (rank 0): author1
GPU 1 (rank 1): author2
GPU 2 (rank 2): author3
GPU 3 (rank 3): (no authors assigned)
```

#### Example 2: 2 GPUs, 5 Authors
```
Valid authors: [author1, author2, author3, author4, author5]
author_subsets[0] = [author1, author3, author5]  # i=0, step=2: [0, 2, 4]
author_subsets[1] = [author2, author4]            # i=1, step=2: [1, 3]

GPU 0 (rank 0): author1, author3, author5
GPU 1 (rank 1): author2, author4
```

### 7.5 Performance Considerations

#### 7.5.1 Load Balancing
- **Uneven Distribution**: If `n_gpus` doesn't divide evenly into `n_authors`, some GPUs will process more authors
- **Memory Usage**: Each GPU must have enough memory for the model, SAEs, and batch processing
- **I/O Bottlenecks**: All processes read from the same dataset simultaneously

#### 7.5.2 Scalability
- **Linear Scaling**: Processing time scales inversely with number of GPUs (for equal author distribution)
- **Memory Requirements**: Each GPU needs sufficient VRAM for model + SAEs + batch data
- **Storage I/O**: Multiple processes writing simultaneously may create I/O contention

#### 7.5.3 Optimization Strategies
- **Batch Size Tuning**: Adjust `--batch_size` based on GPU memory
- **Author Distribution**: Ensure `n_authors >= n_gpus` for optimal utilization
- **Memory Monitoring**: Use `log_gpu_memory_usage()` to track memory usage
- **Periodic Cleanup**: Regular `torch.cuda.empty_cache()` prevents memory leaks

### 7.6 Error Handling and Recovery

#### 7.6.1 Process Isolation
```python
try:
    # Main processing loop
    for batch_idx, batch in enumerate(loader):
        # Process batch...
except Exception as e:
    print(f"Error processing author {author}: {e}")
    traceback.print_exc()
    # Clean up and re-raise
    torch.cuda.empty_cache()
    gc.collect()
    raise
```

**Error Handling:**
- **Process Isolation**: Errors in one GPU don't affect others
- **Memory Cleanup**: Failed processes clean up GPU memory
- **Error Propagation**: Exceptions are logged and re-raised for debugging

#### 7.6.2 Resource Management
- **Automatic Cleanup**: GPU memory is cleared on process exit
- **File Safety**: Output files are saved incrementally to prevent data loss
- **Process Monitoring**: Each process logs its status and memory usage

This multi-GPU architecture enables efficient processing of large-scale feature extraction tasks while maintaining data integrity and providing robust error handling.
