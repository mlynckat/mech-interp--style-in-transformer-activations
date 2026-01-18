import os
from pathlib import Path
import json
import pickle
import traceback
import joblib
import logging
import time
from typing import Dict, List, Optional, Any

import torch
import torch.multiprocessing as mp
from sae_lens import SAE, HookedSAETransformer
from huggingface_hub import login
from dotenv import load_dotenv

from backend.src.steering.steering_methods import (
    SteeringConfig, 
    HeuristicSteering, 
    ProjectedGradientSteering,
    SAEDiffSteering,
    SteeringPosition
)
from backend.src.steering.run_config import (
    RunConfig,
    create_baseline_config,
    create_heuristic_steering_config,
    create_projected_gradient_config,
    create_sae_diff_steering_config,
    DEFAULT_AUTHORS
)
from backend.src.steering.calculate_sae_feature_diffs import load_sae_diffs

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================================
# MANUAL TOKEN-BY-TOKEN GENERATION (FOR PROJECTED GRADIENT STEERING)
# ============================================================================

def manual_generate(
    model,
    input_ids: torch.Tensor,
    max_new_tokens: int,
    temperature: float = 1.0,
    top_k: int | None = None,
    top_p: float | None = None,
    eos_token_id: int | None = None,
    pad_token_id: int | None = None,
    clear_cache_every: int = 50,
    use_kv_cache: bool = True,
) -> torch.Tensor:
    """
    Manual token-by-token generation with gradient support inside hooks.
    
    This function replaces model.generate() for projected gradient steering.
    The forward pass runs WITHOUT gradient tracking (for memory efficiency),
    but hooks can locally enable gradients using torch.enable_grad() for
    their internal optimization loops.
    
    Key guarantees:
    - Forward pass runs without gradients (memory efficient)
    - Hooks can enable gradients locally with torch.enable_grad()
    - loss.backward() inside hooks works correctly
    - KV-cache is supported for efficiency (CRITICAL for memory)
    - Compatible with HookedSAETransformer
    
    Args:
        model: HookedSAETransformer model
        input_ids: Input token IDs [batch, seq]
        max_new_tokens: Maximum number of new tokens to generate
        temperature: Sampling temperature (default 1.0, use 0.7 for less randomness)
        top_k: If set, only sample from top-k most likely tokens
        top_p: If set, use nucleus sampling with this probability threshold
        eos_token_id: Token ID that signals end of sequence
        pad_token_id: Token ID used for padding (optional)
        clear_cache_every: Clear CUDA cache every N tokens (default 50)
        use_kv_cache: Whether to use KV caching for memory efficiency (default True)
        
    Returns:
        tokens: Generated token IDs [batch, seq + generated_tokens]
    """
    from transformer_lens.past_key_value_caching import HookedTransformerKeyValueCache
    
    device = input_ids.device
    batch_size = input_ids.shape[0]
    # Detach input to ensure no gradient history is carried over
    tokens = input_ids.detach().clone()
    
    # Initialize proper KV cache for HookedTransformer
    past_kv_cache = None
    if use_kv_cache:
        past_kv_cache = HookedTransformerKeyValueCache.init_cache(
            model.cfg, device, batch_size
        )
    
    model.eval()  # Keep in eval mode (no dropout)
    
    for step in range(max_new_tokens):
        # Forward pass - use only last token if we have cache, otherwise full sequence
        if use_kv_cache and step > 0:
            # Subsequent passes: only process last token (KV cache has context)
            input_for_forward = tokens[:, -1:]
        else:
            # First pass or no cache: process entire sequence
            input_for_forward = tokens
        
        # Run forward pass WITHOUT gradient tracking for the model itself
        # Hooks will enable gradients locally if needed (via torch.enable_grad())
        # This is critical for memory efficiency - we don't need gradients for
        # the model forward pass, only for the optimization inside hooks
        with torch.no_grad():
            outputs = model(
                input_for_forward,
                past_kv_cache=past_kv_cache if use_kv_cache else None,
                return_type="logits",
            )
        
        # Get logits for the last position (already detached due to no_grad context)
        logits = outputs[:, -1, :].clone()  # [batch, vocab_size]
        
        # Apply temperature scaling
        if temperature != 1.0 and temperature > 0:
            logits = logits / temperature
        
        # Apply top-k filtering
        if top_k is not None and top_k > 0:
            # Get top-k values and indices
            topk_vals, topk_idx = logits.topk(top_k, dim=-1)
            # Create filtered logits with -inf everywhere except top-k
            logits = torch.full_like(logits, float("-inf"))
            logits.scatter_(1, topk_idx, topk_vals)
        
        # Apply top-p (nucleus) filtering
        if top_p is not None and 0.0 < top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
            cumulative_probs = torch.softmax(sorted_logits, dim=-1).cumsum(dim=-1)
            
            # Remove tokens with cumulative probability above threshold
            sorted_indices_to_remove = cumulative_probs > top_p
            # Shift to keep the first token above threshold
            sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
            sorted_indices_to_remove[:, 0] = False
            
            # Scatter sorted tensors back to original indexing
            indices_to_remove = sorted_indices_to_remove.scatter(
                1, sorted_indices, sorted_indices_to_remove
            )
            logits = logits.masked_fill(indices_to_remove, float("-inf"))
        
        # Sample next token
        probs = torch.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)  # [batch, 1]
        
        # Append to sequence (no gradient tracking needed for token IDs)
        tokens = torch.cat([tokens, next_token.detach()], dim=1)
        
        # Check for EOS
        if eos_token_id is not None:
            if (next_token == eos_token_id).all():
                logger.debug(f"EOS token detected at step {step + 1}")
                break
        
        # Periodically clear CUDA cache to prevent memory fragmentation
        if clear_cache_every > 0 and (step + 1) % clear_cache_every == 0:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    return tokens


load_dotenv()
login(token=os.environ["HF_TOKEN"])


def load_steered_features(
    path_to_most_important_features: Path,
    n_features: int
) -> Dict[str, List[int]]:
    """
    Load and process the most important features for steering.
    
    Args:
        path_to_most_important_features: Path to JSON file with feature rankings
        n_features: Number of top features to use per author
        
    Returns:
        Dictionary mapping author name to list of feature indices
    """
    with open(path_to_most_important_features, "r") as f:
        most_important_features = json.load(f)

    steered_features = {}
    for author in most_important_features.keys():
        features_author = most_important_features[author]["shap_variance_threshold"][:n_features]
        # Store as list to preserve order (matches classifier weight order)
        steered_features[author] = [int(feature.replace("x", "")) for feature in features_author]
    
    return steered_features


def read_test_data(input_file: Path) -> List[Dict[str, Any]]:
    """
    Read test data from file.
    
    Args:
        input_file: Path to JSON file with test prompts
        
    Returns:
        List of test data entries, empty list if file doesn't exist
    """
    if not input_file.exists():
        logger.warning(f"Input file not found: {input_file}")
        return []
    with open(input_file, "r") as f:
        test_data = json.load(f)
    return test_data


class SAESteeringGenerator:
    """Main class for text generation with SAE steering."""
    
    def __init__(
        self,
        model_name: str = "google/gemma-2-9b-it",
        sae_release: str = "gemma-scope-9b-pt-res",
        sae_id: str = "layer_15/width_16k/average_l0_131",
        classifier_path: str | None = None, 
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        fold_ln: bool = True,
        # Allow passing pre-loaded model and SAE for efficiency
        model: HookedSAETransformer | None = None,
        sae: SAE | None = None,
    ):
        """
        Initialize SAE Steering Generator.
        
        Args:
            model_name: Model name to load (ignored if model is provided)
            sae_release: SAE release name (ignored if sae is provided)
            sae_id: SAE identifier (ignored if sae is provided)
            classifier_path: Path to classifier file (.pkl or .joblib)
            device: Device to use
            fold_ln: Whether to fold LayerNorm (ignored if model is provided)
            model: Pre-loaded HookedSAETransformer (for efficiency when processing multiple authors)
            sae: Pre-loaded SAE (for efficiency when processing multiple authors)
        """
        self.device = device
        
        # Use provided model or load new one
        if model is not None:
            logger.info("Using pre-loaded model")
            self.model = model
        else:
            logger.info("Loading hooked SAE transformer...")
            self.model = HookedSAETransformer.from_pretrained(
                model_name,
                fold_ln=fold_ln,
                center_writing_weights=False,
                center_unembed=False,
                device=device
            )

        self.tokenizer = self.model.tokenizer
        
        # Use provided SAE or load new one
        if sae is not None:
            logger.info("Using pre-loaded SAE")
            self.sae = sae
        else:
            logger.info(f"Loading SAE: {sae_id}")
            self.sae = SAE.from_pretrained(
                release=sae_release,
                sae_id=sae_id,
                device=device
            )
        
        # Load classifier if path provided
        self.classifier = None
        self.classifier_weights = None
        self.classifier_intercept = None
        if classifier_path is not None:
            self.load_classifier(classifier_path)
        
        self.steering_hook = None
        self.steering_mechanism = None
        self.feature_max_activations: dict[int, float] | None = None
        self._feature_max_tensor: torch.Tensor | None = None
    
    def load_classifier(self, classifier_path: str):
        """Load a new classifier (allows switching classifiers without reloading model/SAE)."""
        classifier_path = Path(classifier_path)
        joblib_classifier_path = classifier_path.with_suffix('.joblib')
        
        logger.info(f"Loading classifier from: {classifier_path}")
        try:
            with open(classifier_path, "rb") as f:
                self.classifier = pickle.load(f)
        except Exception:
            logger.warning(f"Failed to load .pkl, trying .joblib")
            self.classifier = joblib.load(joblib_classifier_path)
    
        # Get classifier weights for class 1
        if hasattr(self.classifier, 'coef_'):
            self.classifier_weights = self.classifier.coef_[0]
        else:
            raise ValueError("Classifier must have coef_ attribute (LogisticRegression)")
        
        # Get classifier intercept (bias term) - CRITICAL for correct probability calculation
        if hasattr(self.classifier, 'intercept_'):
            self.classifier_intercept = self.classifier.intercept_[0]
            logger.info(f"Loaded classifier intercept: {self.classifier_intercept}")
        else:
            logger.warning("Classifier has no intercept_ attribute, defaulting to 0.0")
            self.classifier_intercept = 0.0
    

        
    def set_steering_mechanism(
        self,
        mechanism: str,
        config: SteeringConfig
    ):
        """
        Set the steering mechanism to use.
        
        Args:
            mechanism: One of "heuristic", "projected_gradient", or "sae_diff"
            config: SteeringConfig with appropriate parameters
                   - For heuristic/projected_gradient: requires classifier_weights to be loaded
                   - For sae_diff: requires config.sae_diff to be set
        """
        if mechanism == "sae_diff":
            # SAEDiffSteering doesn't require classifier weights
            if config.sae_diff is None:
                raise ValueError("sae_diff must be set in config for SAEDiffSteering")
            self.steering_mechanism = SAEDiffSteering(config)
        else:
            # Other mechanisms require classifier weights
            if self.classifier_weights is None:
                raise ValueError("Classifier must be loaded before setting steering mechanism. Call load_classifier() first.")
            
            # Get intercept (default to 0.0 if not available)
            intercept = self.classifier_intercept if self.classifier_intercept is not None else 0.0
            
            if mechanism == "heuristic":
                self.steering_mechanism = HeuristicSteering(config, self.classifier_weights, intercept)
            elif mechanism == "projected_gradient":
                self.steering_mechanism = ProjectedGradientSteering(config, self.classifier_weights, intercept)
            else:
                raise ValueError(f"Unknown mechanism: {mechanism}")
        
        logger.info(f"Set steering mechanism: {mechanism}")

        
    def _create_steering_hook(self, auto_tune: bool = False, prompt_length: int = 0, 
                               prompt_steering_mode: str = "last"):
        """
        Create hook function for steering.
        
        Args:
            auto_tune: Whether to auto-tune alpha (for HeuristicSteering)
            prompt_length: Number of prompt tokens (used for position-based steering)
            prompt_steering_mode: How to steer prompt positions. Options:
                - "last": Only steer last position (memory efficient, may reduce quality)
                - "all": Steer all positions (best quality, may OOM on long prompts)
                - "skip": Don't steer prompt at all, only steer generated tokens
                - "chunk_N": Steer in chunks of N positions (e.g., "chunk_32")
            
        Returns:
            Hook function that applies steering to activations
        """
        history_feats = []
        # Track current absolute position in sequence (for position-based steering)
        # Use list to allow mutation within closure
        current_seq_position = [0]
        prompt_processed = [False]
        
        # Parse chunk size if using chunked mode
        chunk_size = None
        if prompt_steering_mode.startswith("chunk_"):
            chunk_size = int(prompt_steering_mode.split("_")[1])
            prompt_steering_mode = "chunk"
        
        def steering_hook(activations, hook):
            # Check for invalid activations first (no gradients needed)
            if activations is None or activations.numel() == 0:
                print("Warning: empty activations, skipping steering.")
                return activations

            if torch.isnan(activations).any() or torch.isinf(activations).any():
                print("ERROR: activations contain NaN or Inf; skipping steering.")
                return activations

            batch_size, seq_len, hidden_dim = activations.shape
            
            # Track position in sequence for position-based steering
            # First call with seq_len > 1 is the prompt processing
            # Subsequent calls with seq_len == 1 are token-by-token generation
            if seq_len > 1:
                print(f"Processing the prompt with seq_len {seq_len} and current_seq_position {current_seq_position[0]}")
                # Processing the prompt
                prompt_processed[0] = True
                current_seq_position[0] = seq_len
                absolute_position = None  # Multiple positions
            else:
                # Generating token by token
                absolute_position = current_seq_position[0]
                current_seq_position[0] += 1
            
            # Check if we should apply steering at this position (applies to all steering methods)
            config = self.steering_mechanism.config
            
            # Determine if we should steer at this position
            # token_idx for position checks (use 0 for prompt processing since multiple positions)
            token_idx = absolute_position if absolute_position is not None else 0
            should_steer = config.should_apply_steering(
                token_idx=token_idx,
                prompt_length=prompt_length,
                seq_len=seq_len
            )
            if not should_steer:
                return activations
                
            print(f"Steering at position {current_seq_position[0]}")
            
            # ================================================================
            # PROMPT STEERING MODE: Controls memory vs quality tradeoff
            # - "all": Best quality, steer all positions (may OOM)
            # - "last": Memory efficient, only steer last position
            # - "skip": Skip prompt steering entirely
            # - "chunk": Process in chunks to balance memory and quality
            # ================================================================
            is_prompt = seq_len > 1
            
            if is_prompt:
                if prompt_steering_mode == "skip":
                    # Don't steer prompt at all
                    return activations
                elif prompt_steering_mode == "all":
                    # Steer full sequence (original behavior, may OOM)
                    positions_to_steer = activations
                    prefix_activations = None
                elif prompt_steering_mode == "last":
                    # Only steer last position (memory efficient)
                    positions_to_steer = activations[:, -1:, :]
                    prefix_activations = activations[:, :-1, :]
                elif prompt_steering_mode == "chunk":
                    # Steer in chunks - process last chunk_size positions
                    n_positions = min(chunk_size, seq_len)
                    positions_to_steer = activations[:, -n_positions:, :]
                    prefix_activations = activations[:, :-n_positions, :] if n_positions < seq_len else None
                else:
                    # Default to "last" for safety
                    positions_to_steer = activations[:, -1:, :]
                    prefix_activations = activations[:, :-1, :]
            else:
                # Single token generation - always steer
                positions_to_steer = activations
                prefix_activations = None
            
            # Handle ProjectedGradientSteering separately - it needs gradients enabled
            # for its internal optimization loop. The forward pass runs inside 
            # torch.no_grad(), so we use torch.enable_grad() to override this locally.
            if isinstance(self.steering_mechanism, ProjectedGradientSteering):
                # Use context manager to enable gradients only for this block
                # This overrides the outer torch.no_grad() from manual_generate
                with torch.enable_grad():
                    # Detach activations from the model's computation graph
                    # We only need gradients for the optimization loop, not backprop to model
                    activations_detached = positions_to_steer.detach().requires_grad_(False)
                    
                    # SAE encode
                    sae_features = self.sae.encode(activations_detached)
                    
                    if torch.isnan(sae_features).any() or torch.isinf(sae_features).any():
                        print("ERROR: SAE encode produced NaN or Inf; skipping.")
                        return activations
                    
                    # Run optimization loop - gradients flow through delta_x only
                    steered_positions = self.steering_mechanism.compute_steering(
                        sae_features,
                        decoder=self.sae.decode,
                        original_activations=positions_to_steer
                    )
                
                # Detach result to prevent gradient leakage outside the hook
                steered_positions = steered_positions.detach()
                
                if torch.isnan(steered_positions).any() or torch.isinf(steered_positions).any():
                    print("ERROR: steered activations contain NaN or Inf; skipping steering for this token.")
                    return activations
                
                # Reconstruct full sequence if we only steered a subset
                if prefix_activations is not None:
                    steered_activations = torch.cat([prefix_activations, steered_positions], dim=1)
                else:
                    steered_activations = steered_positions
                
                return steered_activations
            
            # For other steering methods, wrap in no_grad for efficiency
            with torch.no_grad():
                # Encode the selected positions
                sae_features = self.sae.encode(positions_to_steer)

                if torch.isnan(sae_features).any() or torch.isinf(sae_features).any():
                    print("ERROR: SAE encode produced NaN or Inf; skipping.")
                    return activations
                
                # Compute steering based on mechanism type
                if isinstance(self.steering_mechanism, HeuristicSteering):
                    steered_positions = self.steering_mechanism.compute_steering(
                        sae_features,
                        auto_tune=auto_tune,
                        decoder=self.sae.decode,
                        original_activations=positions_to_steer,
                        history_feats=history_feats
                    )
                elif isinstance(self.steering_mechanism, SAEDiffSteering):
                    # SAEDiffSteering adds pre-computed diffs
                    # Position check already done above, so just apply steering
                    steered_positions = self.steering_mechanism.compute_steering(
                        sae_features,
                        decoder=self.sae.decode,
                        original_activations=positions_to_steer
                    )
                else:
                    # Unknown mechanism - return original activations unchanged
                    logger.warning("Unknown steering mechanism, returning original activations")
                    return activations

                if torch.isnan(steered_positions).any() or torch.isinf(steered_positions).any():
                    print("ERROR: steered activations contain NaN or Inf; skipping steering for this token.")
                    return activations

                # Reconstruct full sequence if we only steered a subset
                if prefix_activations is not None:
                    steered_activations = torch.cat([prefix_activations, steered_positions], dim=1)
                else:
                    steered_activations = steered_positions
                
                return steered_activations

        return steering_hook
    
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 500,
        temperature: float = 0.7,
        top_k: int | None = None,
        top_p: float | None = None,
        apply_steering: bool = True,
        auto_tune: bool = False,
        prompt_steering_mode: str = "skip",
        **kwargs
    ) -> str:
        """
        Generate text with optional steering.
        
        For projected gradient steering, uses manual token-by-token generation
        to enable full gradient support inside hooks. For other methods,
        uses the standard model.generate() approach.
        
        Args:
            prompt: Input prompt
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_k: If set, only sample from top-k most likely tokens
            top_p: If set, use nucleus sampling with this probability threshold
            apply_steering: Whether to apply steering
            auto_tune: Auto-tune alpha for heuristic method
            prompt_steering_mode: How to handle steering during prompt processing.
                Options: "skip" (default, most memory efficient), "last" (steer last position),
                "all" (steer all positions, may OOM), "chunk_N" (steer last N positions)
            
        Returns:
            Generated text (without prompt)
        """

        # CRITICAL: Reset model state before processing a new prompt
        # This ensures no state from previous prompts persists
        self.model.reset_hooks()
        
        # Determine if we need manual generation (for projected gradient steering)
        use_manual_generation = (
            apply_steering 
            and self.steering_mechanism is not None
            and isinstance(self.steering_mechanism, ProjectedGradientSteering)
        )
        
        # Tokenize input
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            add_special_tokens=True  
        ).to(self.device)
        prompt_length = inputs.input_ids.shape[1]
        
        # Set up steering hook if steered generation
        if apply_steering and self.steering_mechanism is not None:
            hook_point = self.sae.cfg.metadata.hook_name
            steering_fn = self._create_steering_hook(
                auto_tune=auto_tune, 
                prompt_length=prompt_length,
                prompt_steering_mode=prompt_steering_mode
            )
            self.steering_hook = self.model.add_hook(hook_point, steering_fn)
        
        try:
            if use_manual_generation:
                # Use manual token-by-token generation for projected gradient steering
                # This ensures full gradient support inside hooks
                logger.info("Using manual generation for projected gradient steering")
                
                # Get EOS token ID
                eos_token_id = self.tokenizer.eos_token_id
                pad_token_id = self.tokenizer.pad_token_id
                
                outputs = manual_generate(
                    model=self.model,
                    input_ids=inputs.input_ids,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                    eos_token_id=eos_token_id,
                    pad_token_id=pad_token_id,
                )
                output_tokens = outputs[0]  # [seq_len]
            else:
                # Use standard model.generate() for other steering methods
                # (wrapped in torch.no_grad() for efficiency)
                with torch.no_grad():
                    outputs = self.model.generate(
                        inputs.input_ids,
                        max_new_tokens=max_new_tokens,
                        temperature=temperature,
                        do_sample=True,
                        stop_at_eos=True,
                        prepend_bos=False,  # BOS already in inputs.input_ids from tokenization
                        **kwargs
                    )
                
                # Validate outputs
                if outputs is None or len(outputs) == 0:
                    raise ValueError("Generation returned empty outputs")
                
                # Handle both single tensor and list of tensors
                if isinstance(outputs, torch.Tensor):
                    output_tokens = outputs[0] if outputs.ndim > 1 else outputs
                else:
                    output_tokens = outputs[0]
            
            # Decode and remove prompt
            generated_text = self.tokenizer.decode(
                output_tokens[prompt_length:], 
                skip_special_tokens=True
            )
            logger.info(f"Number of new tokens: {len(output_tokens) - prompt_length}")
        except Exception as e:
            logger.error(f"Error during generation: {e}")
            traceback.print_exc()
            return None
        finally:
            # Always reset hooks after generation to clean up state
            self.model.reset_hooks()
            
            # Clear CUDA cache to free memory
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        return generated_text



def run_generation_on_gpu(
    rank: int,
    author_subsets: List[List[str]],
    steered_features: Dict[str, List[int]],
    run_config: RunConfig,
    test_data: List[Dict[str, Any]],
    existing_data: Dict[str, Dict[str, List[Dict[str, Any]]]] = None,
):
    """
    Run text generation for authors assigned to the given GPU rank.
    
    Args:
        rank: GPU rank (0, 1, 2, ...)
        author_subsets: List of lists, where author_subsets[rank] contains authors for this GPU
        steered_features: Dictionary mapping author to list of feature indices
        run_config: RunConfig with all generation settings
        test_data: List of test prompts/entries
        existing_data: Dictionary mapping (author, method) to list of existing entries.
                      Format: {author: {method: [entries]}} where method is "heuristic", "projected_gradient", or "sae_diff"
    """
    # Setup device for this rank
    if torch.cuda.is_available():
        torch.cuda.set_device(rank)
        device = torch.device(f"cuda:{rank}")
    else:
        device = torch.device("cpu")
        logger.warning(f"CUDA not available, using CPU for rank {rank}")
    
    # Assign authors to this GPU
    authors_for_this_gpu = author_subsets[rank]
    
    logger.info(f"GPU {rank} will process {len(authors_for_this_gpu)} authors: {authors_for_this_gpu}")
    
    # Get paths utility
    paths = run_config.get_paths()
    paths.ensure_directories_exist()
    
    # === OPTIMIZATION: Load model and SAE once for all authors ===
    logger.info(f"GPU {rank}: Loading model and SAE (will be reused for all authors)")
    
    logger.info(f"GPU {rank}: Loading hooked SAE transformer...")
    shared_model = HookedSAETransformer.from_pretrained(
        run_config.model_name,
        fold_ln=True,
        center_writing_weights=False,
        center_unembed=False,
        device=device
    )
    
    logger.info(f"GPU {rank}: Loading SAE: {run_config.sae_id}")
    shared_sae = SAE.from_pretrained(
        release=run_config.sae_release,
        sae_id=run_config.sae_id,
        device=device
    )
    
    # Create generator once with shared model/SAE (no classifier yet)
    generator = SAESteeringGenerator(
        model=shared_model,
        sae=shared_sae,
        device=device
    )
    
    logger.info(f"GPU {rank}: Model and SAE loaded successfully")
    
    # Determine which steering methods to run
    apply_steering = not run_config.is_baseline
    run_heuristic = run_config.steering_method in (None, "heuristic") or run_config.is_baseline
    run_projected_gradient = run_config.steering_method == "projected_gradient" or (
        apply_steering and run_config.steering_method is None
    )
    run_sae_diff = run_config.steering_method == "sae_diff"
    
    # Load SAE diffs if running sae_diff steering
    sae_diffs = {}
    if run_sae_diff:
        logger.info(f"GPU {rank}: Loading SAE diffs for sae_diff steering")
        try:
            sae_diffs = load_sae_diffs(
                output_dir=Path(run_config.path_to_sae_diffs),
                layer_ind=run_config.target_layer,
                layer_type="res"
            )
            logger.info(f"GPU {rank}: Loaded SAE diffs for {len(sae_diffs)} authors")
        except Exception as e:
            logger.error(f"GPU {rank}: Failed to load SAE diffs: {e}")
            raise
    
    # Process each author assigned to this GPU
    for author_idx, author in enumerate(authors_for_this_gpu):
        logger.info(f"\n=== GPU {rank} Processing Author {author_idx + 1}/{len(authors_for_this_gpu)}: {author} ===")
        
        try:
            # For classifier-based steering, check if author has features
            if run_config.n_shap_features is not None and author not in steered_features:
                logger.warning(f"Author {author} not found in steered_features, skipping")
                continue
            
            if run_config.n_shap_features is not None:
                classifier_path = (
                    run_config.path_to_logreg_models / 
                    f"logreg_model__res__{run_config.target_layer}__{author}__shap_best_{run_config.n_shap_features}.pkl"
                )
                
                # Load only the classifier for this author (model/SAE already loaded)
                generator.load_classifier(classifier_path)
                
                # Create steering config from run config
                steering_config = SteeringConfig(
                    target_layer=run_config.target_layer,
                    subset_features=steered_features[author],
                    alpha=run_config.alpha,
                    lambda_rec=run_config.lambda_rec,
                    mu_norm=run_config.mu_norm,
                    max_iterations=run_config.max_iterations,
                    target_confidence=run_config.target_confidence,
                    num_features=run_config.num_sae_features,
                    early_stop_patience=run_config.early_stop_patience,
                    early_stop_min_delta=run_config.early_stop_min_delta
                )
            
            # Run heuristic steering if applicable
            if run_heuristic and (run_config.steering_method == "heuristic" or run_config.is_baseline):
                # Determine output path based on whether baseline or steered
                method_key = "baseline" if run_config.is_baseline else run_config.steering_method
                paths = run_config.get_paths()
                output_file_heuristic = paths.get_generated_text_path(author)

                
                # Load existing data if available
                if existing_data and author in existing_data and method_key in existing_data[author]:
                    output_data_heuristic = existing_data[author][method_key].copy()
                    existing_prompts = {entry["prompt"] for entry in output_data_heuristic}
                    logger.info(f"GPU {rank}: Loaded {len(output_data_heuristic)} existing entries for {author} ({method_key})")
                else:
                    output_data_heuristic = []
                    existing_prompts = set()
                
                logger.info(f"GPU {rank}: Testing heuristic steering for author {author}")
                generator.set_steering_mechanism("heuristic", steering_config)
                
                for index, entry in enumerate(test_data):
                    prompt = entry["prompt"]
                    
                    # Skip if prompt already exists
                    if prompt in existing_prompts:
                        logger.info(f"GPU {rank}: Skipping prompt {index+1}/{len(test_data)} (already exists): {prompt[:50]}...")
                        continue
                    
                    generated = generator.generate(
                        prompt,
                        max_new_tokens=run_config.max_new_tokens,
                        temperature=run_config.temperature,
                        top_k=run_config.top_k,
                        top_p=run_config.top_p,
                        apply_steering=apply_steering,
                        auto_tune=True,
                        prompt_steering_mode=run_config.prompt_steering_mode
                    )

                    if generated:
                        logger.info(f"GPU {rank}: Generated text: {generated[:100]}...")
                    else:
                        logger.warning(f"GPU {rank}: Generated text is None")

                    output_data_heuristic.append({
                        "id": index,
                        "prompt": prompt,
                        "generated_text": generated,
                        "author": entry["author"],
                        "original_article": entry["article"],
                    })
                    logger.info(f"GPU {rank}: Generated text for prompt {index+1}/{len(test_data)} (heuristic)")
                
                    # Save results after each prompt (for fault tolerance)
                    with open(str(output_file_heuristic), "w") as f:
                        json.dump(output_data_heuristic, f, indent=2, ensure_ascii=False)
                
                logger.info(f"GPU {rank}: Saved heuristic results to {output_file_heuristic}")
            
            # Run projected gradient steering if applicable
            if run_projected_gradient and apply_steering:
                logger.info(f"GPU {rank}: Testing projected gradient steering for author {author}")
                generator.set_steering_mechanism("projected_gradient", steering_config)
                
                # Create paths for projected gradient output
                pg_config = RunConfig(
                    run_name=run_config.run_name,
                    base_dir=run_config.base_dir,
                    is_baseline=False,
                    steering_method="projected_gradient",
                    author_list=run_config.author_list
                )
                pg_paths = pg_config.get_paths()
                output_file_projected_gradient = pg_paths.get_generated_text_path(author)
                
                # Load existing data if available
                method_key = "projected_gradient"
                if existing_data and author in existing_data and method_key in existing_data[author]:
                    output_data_projected_gradient = existing_data[author][method_key].copy()
                    existing_prompts = {entry["prompt"] for entry in output_data_projected_gradient}
                    logger.info(f"GPU {rank}: Loaded {len(output_data_projected_gradient)} existing entries for {author} ({method_key})")
                else:
                    output_data_projected_gradient = []
                    existing_prompts = set()
                
                for pg_index, entry in enumerate(test_data):
                    prompt = entry["prompt"]
                    
                    # Skip if prompt already exists
                    if prompt in existing_prompts:
                        logger.info(f"GPU {rank}: Skipping prompt {pg_index+1}/{len(test_data)} (already exists): {prompt[:50]}...")
                        continue
                    generated = generator.generate(
                        prompt,
                        max_new_tokens=run_config.max_new_tokens,
                        temperature=run_config.temperature,
                        top_k=run_config.top_k,
                        top_p=run_config.top_p,
                        apply_steering=True,
                        prompt_steering_mode=run_config.prompt_steering_mode
                    )
                    logger.info(f"GPU {rank}: Generated text for prompt {pg_index+1}/{len(test_data)} (projected gradient)")
                    output_data_projected_gradient.append({
                        "id": pg_index,
                        "prompt": prompt,
                        "generated_text": generated,
                        "author": entry["author"],
                        "original_article": entry["article"],
                    })
                
                    # Save projected gradient results
                    with open(output_file_projected_gradient, "w") as f:
                        json.dump(output_data_projected_gradient, f, indent=2, ensure_ascii=False)
                    logger.info(f"GPU {rank}: Saved projected gradient results to {output_file_projected_gradient}")
            
            # Run SAE-diff steering if applicable
            if run_sae_diff and apply_steering:
                logger.info(f"GPU {rank}: Testing SAE-diff steering for author {author}")
                
                # Check if we have diff for this author
                if author not in sae_diffs:
                    logger.warning(f"GPU {rank}: No SAE diff found for author {author}, skipping")
                    continue
                
                # Create steering config with SAE diff
                # SAE diff steering uses AFTER_PROMPT position by default
                sae_diff_config = SteeringConfig(
                    target_layer=run_config.target_layer,
                    alpha=run_config.alpha,
                    num_features=run_config.num_sae_features,
                    sae_diff=sae_diffs[author],
                    steering_position=SteeringPosition.AFTER_PROMPT
                )
                
                generator.set_steering_mechanism("sae_diff", sae_diff_config)
                
                # Create paths for SAE-diff output
                output_file_sae_diff = run_config.get_paths().get_generated_text_path(author)
                
                # Load existing data if available
                method_key = "sae_diff"
                if existing_data and author in existing_data and method_key in existing_data[author]:
                    output_data_sae_diff = existing_data[author][method_key].copy()
                    existing_prompts = {entry["prompt"] for entry in output_data_sae_diff}
                    logger.info(f"GPU {rank}: Loaded {len(output_data_sae_diff)} existing entries for {author} ({method_key})")
                else:
                    output_data_sae_diff = []
                    existing_prompts = set()
                
                for sd_index, entry in enumerate(test_data):
                    prompt = entry["prompt"]
                    
                    # Skip if prompt already exists
                    if prompt in existing_prompts:
                        logger.info(f"GPU {rank}: Skipping prompt {sd_index+1}/{len(test_data)} (already exists): {prompt[:50]}...")
                        continue
                    generated = generator.generate(
                        prompt,
                        max_new_tokens=run_config.max_new_tokens,
                        temperature=run_config.temperature,
                        top_k=run_config.top_k,
                        top_p=run_config.top_p,
                        apply_steering=True,
                        prompt_steering_mode=run_config.prompt_steering_mode
                    )
                    
                    if generated:
                        logger.info(f"GPU {rank}: Generated text: {generated[:100]}...")
                    else:
                        logger.warning(f"GPU {rank}: Generated text is None")
                    
                    output_data_sae_diff.append({
                        "id": sd_index,
                        "prompt": prompt,
                        "generated_text": generated,
                        "author": entry["author"],
                        "original_article": entry["article"],
                    })
                    logger.info(f"GPU {rank}: Generated text for prompt {sd_index+1}/{len(test_data)} (sae_diff)")
                    
                    # Save results after each prompt (for fault tolerance)
                    with open(str(output_file_sae_diff), "w") as f:
                        json.dump(output_data_sae_diff, f, indent=2, ensure_ascii=False)
                
                logger.info(f"GPU {rank}: Saved SAE-diff results to {output_file_sae_diff}")
            
        except Exception as e:
            logger.error(f"GPU {rank}: Error processing author {author}: {e}")
            traceback.print_exc()
            continue
    
    # Cleanup after all authors processed
    del generator
    del shared_model
    del shared_sae
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    logger.info(f"GPU {rank}: Completed processing all assigned authors")


def run_generation(run_config: RunConfig, continue_generation: bool = False):
    """
    Run text generation with the given configuration.
    
    This is the main entry point for text generation. It handles:
    - Loading test data and feature mappings
    - Distributing work across available GPUs
    - Saving configuration and results
    
    Args:
        run_config: RunConfig specifying all generation parameters
        continue_generation: If True, load existing config and skip prompts that already exist in output files
    """
    logger.info("=" * 80)
    logger.info(f"Starting text generation run: {run_config.run_name}")
    logger.info("=" * 80)
    logger.info(f"  Continue generation: {continue_generation}")
    logger.info(f"  Baseline: {run_config.is_baseline}")
    logger.info(f"  Steering method: {run_config.steering_method}")
    logger.info(f"  N SHAP features: {run_config.n_shap_features}")
    logger.info(f"  Authors: {run_config.author_list}")
    logger.info(f"  Output dir: {run_config.get_output_dir()}")
    
    # If continuing, load existing config
    if continue_generation:
        paths = run_config.get_paths()
        config_path = paths.run_dir / "run_config.json"
        if config_path.exists():
            logger.info(f"Loading existing config from: {config_path}")
            run_config = RunConfig.load_config(config_path)
            logger.info(f"Loaded config: {run_config.run_name}")
        else:
            logger.warning(f"Config file not found at {config_path}, starting fresh run")
            continue_generation = False
    
    # Ensure output directories exist
    paths = run_config.get_paths()
    paths.ensure_directories_exist()
    
    # Save run configuration (only if not continuing, or if config was modified)
    if not continue_generation:
        run_config.save_config()
    
    # Load existing generated texts if continuing
    existing_data = {}
    if continue_generation:
        logger.info("Loading existing generated texts...")
        for author in run_config.author_list:
            existing_data[author] = {}

            method_key = "baseline" if run_config.is_baseline else run_config.steering_method

            output_file = run_config.get_paths().get_generated_text_path(author)
            if output_file.exists():
                try:
                    with open(output_file, "r") as f:
                        existing_data[author][method_key] = json.load(f)
                    logger.info(f"Loaded {len(existing_data[author][method_key])} entries from {output_file}")
                except Exception as e:
                    logger.warning(f"Failed to load {output_file}: {e}")
                    existing_data[author][method_key] = []
        
        logger.info("Finished loading existing generated texts")
    
    # Load steered features
    if run_config.n_shap_features is not None:
        steered_features = load_steered_features(
            run_config.path_to_most_important_features,
            run_config.n_shap_features
        )
        logger.info(f"Loaded steered features for {len(steered_features)} authors")
    else:
        steered_features = None
        logger.info("No steered features loaded")
     
    # Read test data
    test_data = read_test_data(run_config.input_prompts_file)
    logger.info(f"Loaded {len(test_data)} test prompts from {run_config.input_prompts_file}")
    
    if not test_data:
        logger.error("No test data found. Exiting.")
        return
    
    # Get list of authors to process
    if steered_features:
        # For classifier-based steering, filter to authors with features
        authors_list = [a for a in run_config.author_list if a in steered_features]
        if len(authors_list) < len(run_config.author_list):
            missing = set(run_config.author_list) - set(authors_list)
            logger.warning(f"Some authors missing from steered_features: {missing}")
    else:
        # For SAE-diff steering, use all authors
        authors_list = run_config.author_list
    
    logger.info(f"Processing {len(authors_list)} authors: {authors_list}")
    
    # Determine number of GPUs
    world_size = torch.cuda.device_count()
    if world_size == 0:
        logger.warning("No CUDA devices found, falling back to CPU (single process)")
        world_size = 1
    
    logger.info(f"Using {world_size} GPU(s)")
    
    # Split authors across GPUs
    author_subsets = [authors_list[i::world_size] for i in range(world_size)]
    
    for rank in range(world_size):
        logger.info(f"GPU {rank} will process: {author_subsets[rank]}")
    
    # Run multi-GPU generation
    logger.info("Starting multi-GPU text generation...")
    start_time = time.time()
    
    if world_size > 1:
        mp.spawn(
            run_generation_on_gpu,
            args=(
                author_subsets,
                steered_features,
                run_config,
                test_data,
                existing_data,
            ),
            nprocs=world_size,
            join=True
        )
    else:
        # Single process mode (CPU or single GPU)
        run_generation_on_gpu(
            0,
            author_subsets,
            steered_features,
            run_config,
            test_data,
            existing_data,
        )
    
    total_time = time.time() - start_time
    logger.info(f"=== Total runtime: {total_time:.2f}s ({total_time/60:.2f} minutes) ===")
    logger.info(f"Results saved to: {run_config.get_output_dir()}")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    # Example: Heuristic steering run with 24 features
    heuristic_config = create_heuristic_steering_config(
        run_name="heuristic_with_detailed_prompts",
        n_shap_features=24,
        description="Heuristic steering with detailed prompts"
    )
    #     run_name="run_with_24_features",
    #     n_shap_features=24,
    #     description="Heuristic steering with 24 SHAP features"
    # )
    
    # Example: Baseline run (uncomment to run)
    # baseline_config = create_baseline_config(
    #     run_name="baseline_run",
    #     description="Baseline generation without steering"
    # )
    
    # Example: Projected gradient run (uncomment to run)
    # pg_config = create_projected_gradient_config(
    #     run_name="pg_run_24_features",
    #     n_shap_features=24,
    #     description="Projected gradient steering with 24 features"
    # )

    # diff_config = create_projected_gradient_config(  # create_sae_diff_steering_config(
    #    run_name="projected_gradient_test_run_2",
    #    alpha=0.2,
    #    n_shap_features=24,
    #    lambda_rec=0.02,
    #    mu_norm=0.1,
    #    max_iterations=200,
    #    description="First test run for projected gradient steering",
    #    author_list=["Paige Lavender", "Amanda Terkel"]
    #)
    
    # Run the generation
    # Set continue_generation=True to resume from where it left off
    run_generation(heuristic_config, continue_generation=True)
