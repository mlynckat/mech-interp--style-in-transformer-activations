# SAE Feature Extraction Script

## Overview

The `get_sae_features.py` script is a comprehensive tool for extracting Sparse Autoencoder (SAE) features from transformer models using the AuthorMix dataset. It performs parallel processing across multiple GPUs to efficiently analyze text data and extract meaningful features from different model components.

## 1. Goal of the Script

The primary goal of this script is to:

- **Extract SAE features** from transformer models (Gemma-2-2b and Gemma-2-9b) across multiple layers
- **Analyze author-specific patterns** in text data using the AuthorMix dataset
- **Compute linguistic metrics** including cross-entropy loss and entropy for each token
- **Enable parallel processing** across multiple GPUs for efficient large-scale analysis
- **Store comprehensive results** including activations, metadata, and text data for downstream analysis

The script serves as a foundation for mechanistic interpretability research by providing detailed feature activations that can be used to understand how transformer models process different types of text and author styles.

## 2. Sequential Execution Flow

### Entry Point: `main()` Function

The script execution follows this sequential flow:

#### 2.1 Initialization Phase
```python
def main(parsed_args):
    # Create output directory
    os.makedirs(parsed_args.save_dir, exist_ok=True)
    
    # Load and organize dataset by author
    author_to_docs, author_list = load_AuthorMix_data(...)
```

**What happens:**
1. **Directory Setup**: Creates the specified output directory for saving results
2. **Data Loading**: Loads the AuthorMix dataset and organizes documents by author
3. **Author Assignment**: Determines the number of authors and documents per author
4. **GPU Detection**: Identifies available GPUs and their specifications

#### 2.2 Multi-Process Spawning
```python
mp.spawn(
    run_inference,
    args=(parsed_args, author_list, author_to_docs, batch_size),
    nprocs=world_size,
    join=True
)
```

**What happens:**
1. **Process Creation**: Spawns one process per available GPU
2. **Argument Distribution**: Passes configuration and data to each process
3. **Parallel Execution**: Each process runs `run_inference()` independently
4. **Synchronization**: Waits for all processes to complete

#### 2.3 Per-Process Execution (`run_inference()`)

Each GPU process follows this sequence:

##### 2.3.1 Model and SAE Loading
```python
# Load transformer model
model = HookedTransformer.from_pretrained(parsed_args.model).to(device)

# Load SAEs for different layer types
res_saes, res_saes_names = load_saes(rank, "res", model, ...)
mlp_saes, mlp_saes_names = load_saes(rank, "mlp", model, ...)
att_saes, att_saes_names = load_saes(rank, "att", model, ...)
```

**What happens:**
1. **Model Loading**: Loads the specified transformer model (Gemma-2-2b or Gemma-2-9b)
2. **SAE Loading**: Downloads and loads pre-trained SAEs for residual, MLP, and attention layers
3. **Device Assignment**: Moves all models to the assigned GPU device
4. **Memory Optimization**: Sets models to evaluation mode

##### 2.3.2 Author Assignment and Data Preparation
```python
# Assign author to this GPU process
author_idx = rank % len(author_list)
author = author_list[author_idx]
docs = author_to_docs[author]

# Create dataset and dataloader
dataset = MyDataset(docs, model, max_seq_len)
loader = DataLoader(dataset, batch_size=batch_size, ...)
```

**What happens:**
1. **Author Assignment**: Each GPU process is assigned one or more authors using modulo operation
2. **Document Selection**: Retrieves all documents for the assigned author(s)
3. **Dataset Creation**: Creates a PyTorch dataset with tokenization and padding
4. **DataLoader Setup**: Configures batch processing with custom collation

##### 2.3.3 Batch Processing Loop
```python
for batch_idx, batch in enumerate(tqdm(loader, desc=f"Processing {author}")):
    # Forward pass through model
    logits, cache = model.run_with_cache(input_ids)
    
    # Compute metrics
    cross_entropy_loss_temp = lm_cross_entropy_loss(logits, input_ids, ...)
    entropy_temp = compute_entropy(logits)
    
    # Extract SAE features
    for layeri, sae_name, sae in zip(layers, sae_names, saes):
        target_acts = cache[f'hook_name']
        sae_acts = sae.encode(target_acts) > 1
        # Store activations...
```

**What happens for each batch:**
1. **Tokenization**: Converts text to token IDs with attention masks
2. **Forward Pass**: Runs the model with caching to capture intermediate activations
3. **Metric Computation**: Calculates cross-entropy loss and entropy for each token
4. **SAE Processing**: Extracts features from residual, MLP, and attention layers
5. **Memory Management**: Cleans up GPU memory periodically

##### 2.3.4 Results Storage
```python
# Save all results for this author
safe_save_npz(output_file, activations=activations, metadata=metadata)
```

**What happens:**
1. **Data Compilation**: Organizes all computed features and metrics
2. **File Writing**: Saves results in compressed NPZ format
3. **Metadata Storage**: Includes author, layer, and processing information
4. **Memory Cleanup**: Clears GPU memory and Python variables

## 3. Command Line Arguments

### Required Arguments
- **`--layers`** (list of integers): Layer indices to analyze
  - Example: `--layers 5 10 15 20`
  - Determines which transformer layers to extract features from

### Optional Arguments

#### Model Configuration
- **`--model`** (string): Transformer model to use
  - Default: `"google/gemma-2-2b"`
  - Choices: `["google/gemma-2-2b", "google/gemma-2-9b"]`
  - Determines model size and corresponding SAE repositories

- **`--sae_features`** (string): SAE feature count
  - Default: `"16k"`
  - Format: `"16k"`, `"32k"`, `"65k"`, `"131k"`
  - Determines the size of SAE feature spaces

- **`--target_l0`** (integer): Target L0 sparsity
  - Default: `50`
  - Range: Typically 10-100
  - Selects SAEs with closest L0 sparsity to target

#### Data Configuration
- **`--n_docs`** (integer): Total documents to process
  - Default: `None` (process all available)
  - Limits the total number of documents across all authors

- **`--n_docs_per_author`** (integer): Documents per author
  - Default: `250`
  - Controls how many documents to process for each author

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

- **`--save_dir`** (string): Output directory
  - Default: `"data/raw_features/AuthorMix"`
  - Where all results will be saved

### Example Usage
```bash
python get_sae_features.py \
    --model "google/gemma-2-2b" \
    --layers 5 10 15 20 \
    --sae_features "16k" \
    --target_l0 50 \
    --n_docs_per_author 100 \
    --batch_size 4 \
    --save_dir "results/experiment_1"
```

## 4. Outputs: What, Where, and Shape

### Output Directory Structure
```
save_dir/
├── sae__google_gemma-2-2b__entropy_loss__{author}.npy
├── sae__google_gemma-2-2b__cross_entropy_loss__{author}.npy
├── sae__google_gemma-2-2b__tokens__{author}.json
├── sae__google_gemma-2-2b__full_texts__{author}.json
├── sae__google_gemma-2-2b__res__activations__{author}__layer_{N}.npz
├── sae__google_gemma-2-2b__mlp__activations__{author}__layer_{N}.npz
└── sae__google_gemma-2-2b__att__activations__{author}__layer_{N}.npz
```

### Output Files Description

#### 4.1 Entropy and Loss Files
- **File**: `sae__{model}__entropy_loss__{author}.npy`
- **Shape**: `(max_docs_per_author, max_seq_len-1)`
- **Data Type**: `float16`
- **Content**: Entropy values for each token position in each document
- **Purpose**: Measures uncertainty in model predictions

- **File**: `sae__{model}__cross_entropy_loss__{author}.npy`
- **Shape**: `(max_docs_per_author, max_seq_len-1)`
- **Data Type**: `float16`
- **Content**: Cross-entropy loss for each token position
- **Purpose**: Measures prediction accuracy

#### 4.2 Text Data Files
- **File**: `sae__{model}__tokens__{author}.json`
- **Format**: JSON array of token lists
- **Content**: Tokenized text for each document
- **Purpose**: Provides token-level text data for analysis

- **File**: `sae__{model}__full_texts__{author}.json`
- **Format**: JSON array of strings
- **Content**: Original full text of each document
- **Purpose**: Provides original text for reference and analysis

#### 4.3 SAE Activation Files
- **File**: `sae__{model}__{layer_type}__activations__{author}__layer_{N}.npz`
- **Layer Types**: `res` (residual), `mlp`, `att` (attention)
- **Shape**: `(max_docs_per_author, max_seq_len-1, n_sae_features)`
- **Data Type**: `float32`
- **Content**: Binary activation patterns (0 or 1) for each SAE feature

##### Activation File Contents
```python
{
    'activations': np.array,  # Shape: (docs, seq_len, features)
    'metadata': {
        'author': str,           # Author name
        'docs_count': int,       # Number of documents processed
        'layer': int,            # Layer number
        'tokens_per_doc': np.array,  # Actual token count per document
        'sae_name': str          # SAE identifier
    }
}
```

### Data Shapes and Dimensions

#### Dimension Explanations
- **`max_docs_per_author`**: Maximum number of documents processed for any author (determined at runtime)
- **`max_seq_len-1`**: Sequence length minus 1 (first token excluded from analysis)
- **`n_sae_features`**: Number of features in the SAE (e.g., 16,384 for "16k")

#### Memory Considerations
- **Activation Storage**: Each activation file can be large (e.g., 250 docs × 359 tokens × 16k features ≈ 1.4GB)
- **Compression**: Files are saved with NPZ compression to reduce disk usage
- **Data Types**: Uses `float16` for metrics and `float32` for activations to balance precision and storage

## 5. Multi-GPU Utilization and Parallelization

### 5.1 Parallelization Strategy

The script uses **data parallelism** with **author-based distribution**:

```python
# Each GPU process is assigned authors using modulo operation
author_idx = rank % len(author_list)
author = author_list[author_idx]
```

#### Distribution Logic
- **Process Assignment**: Each GPU runs one process (rank 0, 1, 2, ...)
- **Author Assignment**: Authors are distributed using `rank % n_authors`
- **Load Balancing**: If `n_gpus > n_authors`, some GPUs will be idle
- **Round-Robin**: Authors are distributed in round-robin fashion

### 5.2 GPU Utilization Details

#### 5.2.1 Model and SAE Loading per GPU
```python
# Each GPU loads its own copy of:
device = torch.device(f"cuda:{rank}")
model = HookedTransformer.from_pretrained(model_name).to(device)
res_saes = load_saes(rank, "res", model, layers, ...)  # Loaded to GPU rank
mlp_saes = load_saes(rank, "mlp", model, layers, ...)  # Loaded to GPU rank
att_saes = load_saes(rank, "att", model, layers, ...)  # Loaded to GPU rank
```

**What's on each GPU:**
- **Complete Model**: Full transformer model (2B or 9B parameters)
- **All SAEs**: Residual, MLP, and attention SAEs for specified layers
- **Author Data**: Documents for assigned author(s)
- **Processing Memory**: Batch processing and activation storage

#### 5.2.2 Memory Management per GPU
```python
# Memory monitoring and cleanup
log_gpu_memory_usage()  # Logs memory usage for current GPU
torch.cuda.empty_cache()  # Clears unused GPU memory
gc.collect()  # Python garbage collection
```

**Memory Optimization:**
- **Periodic Cleanup**: Memory cleared every 10 batches
- **Batch Processing**: Small batch sizes to fit in GPU memory
- **Activation Storage**: Stored in CPU memory, transferred to GPU as needed

### 5.3 Parallelization Architecture

#### 5.3.1 Process Spawning
```python
mp.spawn(
    run_inference,           # Function to run on each process
    args=(parsed_args, author_list, author_to_docs, batch_size),
    nprocs=world_size,       # Number of processes = number of GPUs
    join=True                # Wait for all processes to complete
)
```

**Process Management:**
- **Independent Processes**: Each GPU runs in a separate Python process
- **No Inter-Process Communication**: Processes don't share memory or data
- **Synchronization**: `join=True` ensures all processes complete before main exits

#### 5.3.2 Data Distribution
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

### 5.4 GPU Assignment Examples

#### Example 1: 4 GPUs, 3 Authors
```
GPU 0 (rank 0): author1 (rank % 3 = 0)
GPU 1 (rank 1): author2 (rank % 3 = 1)  
GPU 2 (rank 2): author3 (rank % 3 = 2)
GPU 3 (rank 3): author1 (rank % 3 = 0)  # Same as GPU 0
```

#### Example 2: 2 GPUs, 5 Authors
```
GPU 0 (rank 0): author1, author3, author5 (rank % 5 = 0, 2, 4)
GPU 1 (rank 1): author2, author4 (rank % 5 = 1, 3)
```

### 5.5 Performance Considerations

#### 5.5.1 Load Balancing
- **Uneven Distribution**: If `n_gpus` doesn't divide evenly into `n_authors`, some GPUs will process more authors
- **Memory Usage**: Each GPU must have enough memory for the model, SAEs, and batch processing
- **I/O Bottlenecks**: All processes read from the same dataset simultaneously

#### 5.5.2 Scalability
- **Linear Scaling**: Processing time scales inversely with number of GPUs (for equal author distribution)
- **Memory Requirements**: Each GPU needs sufficient VRAM for model + SAEs + batch data
- **Storage I/O**: Multiple processes writing simultaneously may create I/O contention

#### 5.5.3 Optimization Strategies
- **Batch Size Tuning**: Adjust `--batch_size` based on GPU memory
- **Author Distribution**: Ensure `n_authors >= n_gpus` for optimal utilization
- **Memory Monitoring**: Use `log_gpu_memory_usage()` to track memory usage
- **Periodic Cleanup**: Regular `torch.cuda.empty_cache()` prevents memory leaks

### 5.6 Error Handling and Recovery

#### 5.6.1 Process Isolation
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

#### 5.6.2 Resource Management
- **Automatic Cleanup**: GPU memory is cleared on process exit
- **File Safety**: Output files are saved incrementally to prevent data loss
- **Process Monitoring**: Each process logs its status and memory usage

This multi-GPU architecture enables efficient processing of large-scale feature extraction tasks while maintaining data integrity and providing robust error handling.
