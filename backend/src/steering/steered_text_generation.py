import sys
import os
from pathlib import Path
import json
import pickle
import traceback
import joblib
import hashlib
import logging
import time

import torch
import torch.multiprocessing as mp
from sae_lens import SAE, HookedSAETransformer
from huggingface_hub import login
from dotenv import load_dotenv

from backend.src.steering.steering_methods import SteeringConfig, HeuristicSteering, ProjectedGradientSteering

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


load_dotenv()
login(token=os.environ["HF_TOKEN"])


base_dir = Path("data/steering/tests")
input_filename = "prompts_test_data.json"
apply_steering = True
output_filename_heuristic = "generated_texts__steered__heuristic__{}.json" if apply_steering else "generated_texts__sae_baseline__{}.json"
output_filename_projected_gradient = "generated_texts__steered__projected_gradient__{}.json" if apply_steering else "generated_texts__sae_baseline__{}.json"
input_file = base_dir / input_filename
output_file_heuristic = base_dir / output_filename_heuristic
output_file_projected_gradient = base_dir / output_filename_projected_gradient
path_to_logreg_models = Path("data/output_data/news/politics/google_gemma-2-9b-it/prepare_features_for_steering/feature_selection_aggregated")
path_to_most_important_features = Path("data/output_data/news/politics/google_gemma-2-9b-it/prepare_features_for_steering/feature_selection_aggregated/most_important_features__res__15.json")



with open(path_to_most_important_features, "r") as f:
    most_important_features = json.load(f)

steered_features = {}

for author in most_important_features.keys():
    features_author = most_important_features[author]["shap_variance_threshold"][:16]
    # Store as list to preserve order (matches classifier weight order)
    steered_features[author] = [int(feature.replace("x", "")) for feature in features_author]


def read_test_data():
    """Read test data from file. Returns empty list if file doesn't exist."""
    if not input_file.exists():
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
        classifier_path: str = "classifier.pkl", # TODO: change to the actual path
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        fold_ln: bool = True
    ):
        """
        Initialize SAE Steering Generator.
        
        Note on fold_ln parameter:
        - fold_ln=True folds layer normalization into weights, which modifies model behavior
        - This is typically required for SAE compatibility, but causes outputs to differ from
          standard AutoModelForCausalLM (used in baseline_text_generation.py)
        - If you need outputs to match baseline, try fold_ln=False, but SAE may not work correctly
        """
        self.device = device
        
        
        # Load hooked SAE transformer
        print("Loading hooked SAE transformer...")
        print(f"Using fold_ln={fold_ln} (NOTE: fold_ln=True changes model architecture and may cause different outputs vs baseline)")
        self.model = HookedSAETransformer.from_pretrained(
            model_name,
            fold_ln=fold_ln,
            center_writing_weights=False,
            center_unembed=False,
            device = device)

        self.tokenizer = self.model.tokenizer
        
        # Load SAE
        print(f"Loading SAE: {sae_id}")
        self.sae, _, _ = SAE.from_pretrained(
            release=sae_release,
            sae_id=sae_id,
            device=device
        )
        joblib_classifier_path = Path(str(classifier_path).replace('.pkl', '.joblib'))
        p = Path(joblib_classifier_path)
        print("path:", p)
        print("exists:", p.exists())
        print("size (bytes):", p.stat().st_size if p.exists() else None)

        # optional: compute checksum for comparing to original
        h = hashlib.sha256(p.read_bytes()).hexdigest() if p.exists() else None
        print("sha256:", h)
        
        # Load classifier
        print(f"Loading classifier from path: {classifier_path}")
        try:
            with open(classifier_path, "rb") as f:
                self.classifier = pickle.load(f)
        except Exception:
            traceback.print_exc()
            self.classifier = joblib.load(joblib_classifier_path)
    
        # Get classifier weights for class 1
        if hasattr(self.classifier, 'coef_'):
            self.classifier_weights = self.classifier.coef_[0]
        else:
            raise ValueError("Classifier must have coef_ attribute (LogisticRegression)")
        
        self.steering_hook = None
        self.steering_mechanism = None
        self.feature_max_activations: dict[int, float] | None = None
        self._feature_max_tensor: torch.Tensor | None = None
    

        
    def set_steering_mechanism(
        self,
        mechanism: str,
        config: SteeringConfig
    ):
        """Set the steering mechanism to use."""
        if mechanism == "heuristic":
            self.steering_mechanism = HeuristicSteering(config, self.classifier_weights)
        elif mechanism == "projected_gradient":
            self.steering_mechanism = ProjectedGradientSteering(config, self.classifier_weights)
        else:
            raise ValueError(f"Unknown mechanism: {mechanism}")
        
        print(f"Set steering mechanism: {mechanism}")

        
    def _create_steering_hook(self, auto_tune: bool = False, prompt_length: int = 0):
        """Create hook function for steering."""

        history_feats = []
        
        def steering_hook(activations, hook):
            # Optionally wrap in no_grad if you know decoding should not require gradients
            with torch.no_grad():
                if activations is None or activations.numel() == 0:
                    print("Warning: empty activations, skipping steering.")
                    return activations

                if torch.isnan(activations).any() or torch.isinf(activations).any():
                    print("ERROR: activations contain NaN or Inf; skipping steering.")
                    return activations

                # Encode
                sae_features = self.sae.encode(activations)
                if torch.isnan(sae_features).any() or torch.isinf(sae_features).any():
                    print("ERROR: SAE encode produced NaN or Inf; skipping.")
                    return activations

                # If SAE output is [batch, token_seq, dim], pick last token; else assume [batch, dim]
                if sae_features.ndim == 3:
                    # take last token
                    cur_feat = sae_features[:, -1, :]
                else:
                    cur_feat = sae_features  # [batch, sae_dim]

                # detach + store in history as plain tensor
                feat_to_store = cur_feat.detach().clone()
                history_feats.append(feat_to_store)

                # If we exceed max_history, drop oldest
                if len(history_feats) > 500:
                    print("WARNING: history_feats is too long, dropping oldest token")
                    history_feats.pop(0)

                # Compute delta_x
                if isinstance(self.steering_mechanism, HeuristicSteering):
                    delta_x = self.steering_mechanism.compute_steering(
                        sae_features,
                        auto_tune=auto_tune,
                        decoder=self.sae.decode,
                        original_activations=activations,
                        history_feats=history_feats
                    )
                elif isinstance(self.steering_mechanism, ProjectedGradientSteering):
                    delta_x = self.steering_mechanism.compute_steering(
                        sae_features,
                        decoder=self.sae.decode,
                        original_activations=activations
                    )
                else:
                    delta_x = torch.zeros_like(sae_features)

                if torch.isnan(delta_x).any() or torch.isinf(delta_x).any():
                    print("ERROR: delta_x contains NaN or Inf; skipping steering for this token.")
                    return activations

                # Broadcast delta_x if necessary
                if delta_x.shape != sae_features.shape:
                    # expand/broadcast
                    delta_x = delta_x.expand_as(sae_features)

                steered_feats = sae_features + delta_x

                if torch.isnan(steered_feats).any() or torch.isinf(steered_feats).any():
                    print("ERROR: steered_feats contain NaN/Inf; skipping.")
                    return activations

                steered_activations = self.sae.decode(steered_feats).to(activations.dtype)

                if torch.isnan(steered_activations).any() or torch.isinf(steered_activations).any():
                    print("ERROR: decoded steered activations contain NaN/Inf; skipping.")
                    return activations

                if steered_activations.shape != activations.shape:
                    print(f"Shape mismatch: input {activations.shape}, output {steered_activations.shape}; skipping.")
                    return activations

                print(f"Steering applied: replacement activations shape {steered_activations.shape}")

                return steered_activations

        return steering_hook
    
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 500,
        temperature: float = 0.7,
        apply_steering: bool = True,
        auto_tune: bool = False,
        **kwargs
    ) -> str:
        """
        Generate text with optional steering.
        
        Args:
            prompt: Input prompt
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            apply_steering: Whether to apply steering
            auto_tune: Auto-tune alpha for heuristic method
            
        Returns:
            Generated text (without prompt)
        """
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
            steering_fn = self._create_steering_hook(auto_tune=auto_tune)
            self.steering_hook = self.model.add_hook(hook_point, steering_fn)
        
        try:
            # Generate
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
            print(f"Number of new tokens: {len(output_tokens) - prompt_length}")
            
        finally:
            # Remove hook
            if self.steering_hook is not None:
                self.steering_hook.remove()
                self.steering_hook = None
        
        return generated_text


def run_generation_on_gpu(
    rank,
    author_subsets,
    steered_features,
    path_to_logreg_models,
    base_dir,
    output_filename_heuristic,
    output_filename_projected_gradient,
    apply_steering,
    test_data,
    model_name,
    sae_release,
    sae_id
):
    """
    Run text generation for authors assigned to the given GPU rank.
    
    Args:
        rank: GPU rank (0, 1, 2, ...)
        author_subsets: List of lists, where author_subsets[rank] contains authors for this GPU
        steered_features: Dictionary mapping author to list of feature indices
        path_to_logreg_models: Path to directory containing classifier models
        base_dir: Base directory for output files
        output_filename_heuristic: Format string for heuristic output filename
        output_filename_projected_gradient: Format string for projected gradient output filename
        apply_steering: Whether to apply steering
        test_data: List of test prompts/entries
        model_name: Model name to use
        sae_release: SAE release name
        sae_id: SAE identifier
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
    
    # Process each author assigned to this GPU
    for author_idx, author in enumerate(authors_for_this_gpu):
        logger.info(f"\n=== GPU {rank} Processing Author {author_idx + 1}/{len(authors_for_this_gpu)}: {author} ===")
        
        try:
            classifier_path = path_to_logreg_models / f"logreg_model__res__15__{author}__shap_best_16.pkl"
            
            if author not in steered_features:
                logger.warning(f"Author {author} not found in steered_features, skipping")
                continue
            
            config = SteeringConfig(
                target_layer=15,
                subset_features=steered_features[author],
                alpha=0.1,  # Moderate steering
                lambda_rec=10.0,
                mu_norm=0.01,
                max_iterations=20,
                target_confidence=0.7,
                max_relative_rec_loss=0.01,  # 1% increase allowed
                num_features=16384
            )
            
            # Initialize generator
            logger.info(f"GPU {rank}: Initializing generator for author {author}")
            generator = SAESteeringGenerator(
                model_name=model_name,
                sae_release=sae_release,
                sae_id=sae_id,
                classifier_path=classifier_path,
                device=device
            )
            
            output_data_heuristic = []
            output_data_projected_gradient = []
            
            # Test with heuristic steering
            logger.info(f"GPU {rank}: Testing heuristic steering for author {author}")
            generator.set_steering_mechanism("heuristic", config)

            output_file_heuristic = str(base_dir / output_filename_heuristic).format(author)
            
            for entry in test_data:
                prompt = entry["prompt"]
                
                generated = generator.generate(
                    prompt,
                    max_new_tokens=500,
                    apply_steering=apply_steering,
                    auto_tune=True
                )
                output_data_heuristic.append({
                    "prompt": prompt,
                    "generated_text": generated,
                    "author": entry["author"],
                    "original_article": entry["article"],
                })
                logger.info(f"GPU {rank}: Generated text for prompt (heuristic)")
            
                # Save heuristic results
                
                with open(str(output_file_heuristic), "w") as f:
                    json.dump(output_data_heuristic, f, indent=2, ensure_ascii=False)
                logger.info(f"GPU {rank}: Saved heuristic results to {output_file_heuristic}")
            
            if apply_steering:
                # Test with projected gradient steering
                logger.info(f"GPU {rank}: Testing projected gradient steering for author {author}")
                generator.set_steering_mechanism("projected_gradient", config)
                
                for entry in test_data:
                    prompt = entry["prompt"]
                    generated = generator.generate(
                        prompt,
                        max_new_tokens=500,
                        apply_steering=True
                    )
                    logger.info(f"GPU {rank}: Generated text for prompt (projected gradient)")
                    output_data_projected_gradient.append({
                        "prompt": prompt,
                        "generated_text": generated,
                        "author": entry["author"],
                        "original_article": entry["article"],
                    })
                
                # Save projected gradient results
                output_file_projected_gradient = base_dir / output_filename_projected_gradient.format(author)
                with open(str(output_file_projected_gradient), "w") as f:
                    json.dump(output_data_projected_gradient, f, indent=2, ensure_ascii=False)
                logger.info(f"GPU {rank}: Saved projected gradient results to {output_file_projected_gradient}")
            
            # Cleanup generator for this author
            del generator
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
        except Exception as e:
            logger.error(f"GPU {rank}: Error processing author {author}: {e}")
            traceback.print_exc()
            continue
    
    logger.info(f"GPU {rank}: Completed processing all assigned authors")


# Example usage
if __name__ == "__main__":
    # Configuration
    model_name = "google/gemma-2-9b-it"
    sae_release = "gemma-scope-9b-pt-res"
    sae_id = "layer_15/width_16k/average_l0_131"
    
    # Read test data once
    test_data = read_test_data()
    logger.info(f"Loaded {len(test_data)} test prompts")
    
    # Get list of authors to process
    authors_list = list(steered_features.keys())
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
                path_to_logreg_models,
                base_dir,
                output_filename_heuristic,
                output_filename_projected_gradient,
                apply_steering,
                test_data,
                model_name,
                sae_release,
                sae_id
            ),
            nprocs=world_size,
            join=True
        )
    else:
        # Single process mode (CPU or single GPU)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        run_generation_on_gpu(
            0,
            author_subsets,
            steered_features,
            path_to_logreg_models,
            base_dir,
            output_filename_heuristic,
            output_filename_projected_gradient,
            apply_steering,
            test_data,
            model_name,
            sae_release,
            sae_id
        )
    
    total_time = time.time() - start_time
    logger.info(f"=== Total runtime: {total_time:.2f}s ({total_time/60:.2f} minutes) ===")
