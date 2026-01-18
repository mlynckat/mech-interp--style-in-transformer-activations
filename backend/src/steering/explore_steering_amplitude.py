import sys
import os
import argparse
from pathlib import Path
import json
import pickle
import traceback
import joblib
import logging
import time
import pandas as pd
from typing import Optional, Union, List, Set

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


base_dir = Path("data/steering/explore_steering_amplitude")
run_suffix = "with_rec_loss__pt_15_res_saes"
output_dir = base_dir / run_suffix
os.makedirs(output_dir, exist_ok=True)


input_filename = "prompts_test_data.json"
output_filename_cosine_similarity = "cosine_similarity__{}.csv" 
output_filename_euclidian_distance = "euclidian_distance__{}.csv" 
output_filename_reconstruction_loss = "reconstruction_loss__{}.csv"
input_file = base_dir / input_filename
output_path_cosine_similarity = output_dir / output_filename_cosine_similarity
output_path_euclidian_distance = output_dir / output_filename_euclidian_distance
output_path_reconstruction_loss = output_dir / output_filename_reconstruction_loss
path_to_logreg_models = Path("data/output_data/news/politics/google_gemma-2-9b-it/prepare_features_for_steering/feature_selection_aggregated")
path_to_most_important_features = Path("data/output_data/news/politics/google_gemma-2-9b-it/prepare_features_for_steering/feature_selection_aggregated/most_important_features__res__15.json")
output_path_samples = output_dir / "samples__{}.json"


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


class SAESteeringAmplitudeGenerator:
    """Main class for text generation with SAE steering."""
    
    def __init__(
        self,
        model_name: str = "google/gemma-2-9b-it",
        sae_release: str = "gemma-scope-9b-pt-res",
        sae_id: str = "layer_15/width_16k/average_l0_131",
        classifier_path: str = "classifier.pkl",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        fold_ln: bool = True
    ):
        
        # Initialize SAE Steering Generator.
        self.device = device
        
        
        # Load hooked SAE transformer
        print("Loading hooked SAE transformer...")
        print(f"Using fold_ln={fold_ln} (NOTE: fold_ln=True changes model architecture and may cause different outputs vs baseline)")
        """self.model = HookedSAETransformer.from_pretrained(
            model_name,
            fold_ln=fold_ln,
            center_writing_weights=False,
            center_unembed=False,
            device = device)"""

        self.model = HookedSAETransformer.from_pretrained_no_processing(
            model_name,
            device=device
        )

        self.tokenizer = self.model.tokenizer
        
        # Load SAE
        print(f"Loading SAE: {sae_id}")
        self.sae = SAE.from_pretrained(
            release=sae_release,
            sae_id=sae_id,
            device=device
        )
        joblib_classifier_path = Path(str(classifier_path).replace('.pkl', '.joblib'))
        
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
        
        # Get classifier intercept (bias term) - CRITICAL for correct probability calculation
        if hasattr(self.classifier, 'intercept_'):
            self.classifier_intercept = self.classifier.intercept_[0]
            logger.info(f"Loaded classifier intercept: {self.classifier_intercept}")
        else:
            logger.warning("Classifier has no intercept_ attribute, defaulting to 0.0")
            self.classifier_intercept = 0.0
        
        self.steering_hook = None
        self.steering_mechanism = None
        self.feature_max_activations: dict[int, float] | None = None
        self._feature_max_tensor: torch.Tensor | None = None
        self._token_position = 0  # Track current token position during generation
        self._prompt_length = 0  # Store prompt length for position tracking

    def set_steering_mechanism(
        self,
        mechanism: str,
        config: SteeringConfig
    ):
        """Set the steering mechanism to use."""
        # Get intercept (default to 0.0 if not available)
        intercept = self.classifier_intercept if hasattr(self, 'classifier_intercept') and self.classifier_intercept is not None else 0.0
        
        if mechanism == "heuristic":
            self.steering_mechanism = HeuristicSteering(config, self.classifier_weights, intercept)
        elif mechanism == "projected_gradient":
            self.steering_mechanism = ProjectedGradientSteering(config, self.classifier_weights, intercept)
        else:
            raise ValueError(f"Unknown mechanism: {mechanism}")
        
        print(f"Set steering mechanism: {mechanism}")
    
    def _should_steer_token(self, token_position: int, steered_token_positions: Optional[Union[List[int], Set[int], str]]) -> bool:
        """
        Determine if steering should be applied at the given token position.
        
        Args:
            token_position: Current token position (0-indexed, relative to prompt end)
            steered_token_positions: 
                - None or "all": steer all tokens
                - List/Set of ints: steer only at these positions (0 = first generated token)
                - "first_n" where n is int: steer only first n tokens
                
        Returns:
            True if steering should be applied, False otherwise
        """
        if steered_token_positions is None or steered_token_positions == "all":
            return True
        
        if isinstance(steered_token_positions, str):
            # Handle special string formats like "first_10"
            if steered_token_positions.startswith("first_"):
                try:
                    n = int(steered_token_positions.split("_")[1])
                    return token_position < n
                except (ValueError, IndexError):
                    logger.warning(f"Invalid steered_token_positions format: {steered_token_positions}, defaulting to all")
                    return True
            else:
                logger.warning(f"Unknown steered_token_positions format: {steered_token_positions}, defaulting to all")
                return True
        
        if isinstance(steered_token_positions, (list, set)):
            return token_position in steered_token_positions
        
        return True

        
    def _create_steering_hook(
        self, 
        output_data_cosine_similarity: dict, 
        output_data_euclidian_distance: dict,
        output_data_reconstruction_loss: dict,
        steered_token_positions: Optional[Union[List[int], Set[int], str]] = None
    ):
        """Create hook function for steering with token position control."""
        
        def steering_hook(activations, hook):
            try:
                with torch.no_grad():
                    if activations is None or activations.numel() == 0:
                        logger.warning("Warning: empty activations, skipping steering.")
                        return activations

                    if torch.isnan(activations).any() or torch.isinf(activations).any():
                        logger.error("ERROR: activations contain NaN or Inf; skipping steering.")
                        return activations
                    
                    # Determine current token position
                    # activations shape: [batch, seq_len, hidden_dim]
                    # During generation, seq_len grows. The last position is the current token being generated
                    batch_size, seq_len, hidden_dim = activations.shape
                    current_token_position = seq_len - self._prompt_length - 1  # 0-indexed relative to prompt end
                    
                    # Check if we should steer at this position
                    if not self._should_steer_token(current_token_position, steered_token_positions):
                        return activations
                    
                    # Encode
                    sae_features = self.sae.encode(activations)
                    
                    if torch.isnan(sae_features).any() or torch.isinf(sae_features).any():
                        logger.error("ERROR: SAE encode produced NaN or Inf; skipping.")
                        return activations

                    for alpha in output_data_cosine_similarity.keys():
                        delta_x = self.steering_mechanism.compute_steering_with_alpha(
                            sae_features,
                            alpha=alpha
                        )
                        
                        if torch.isnan(delta_x).any() or torch.isinf(delta_x).any():
                            logger.warning(f"ERROR: delta_x contains NaN or Inf for alpha {alpha}; skipping.")
                            continue
                        
                        steered_feats = sae_features + delta_x
                        
                        if torch.isnan(steered_feats).any() or torch.isinf(steered_feats).any():
                            logger.warning(f"ERROR: steered_feats contain NaN/Inf for alpha {alpha}; skipping.")
                            continue
                        
                        steered_activations = self.sae.decode(steered_feats).to(activations.dtype)
                        reconstructed_original_activations = self.sae.decode(sae_features)
                        
                        # Handle multi-dimensional tensors - compute mean across batch and sequence dimensions
                        # activations shape is typically [batch, seq_len, hidden_dim]
                        cosine_similarity = torch.nn.functional.cosine_similarity(
                            steered_activations, reconstructed_original_activations, dim=-1
                        )
                        euclidian_distance = torch.norm(steered_activations - reconstructed_original_activations, dim=-1)
                        
                        # Reconstruction loss: MSE between original activations and SAE reconstruction
                        reconstruction_loss = torch.nn.functional.mse_loss(
                            reconstructed_original_activations, activations, reduction='none'
                        ).mean(dim=-1)  # Mean across hidden_dim to get per-token loss
                        
                        # Take mean across all dimensions except the last one (feature dimension)
                        # This gives us a single scalar value per token position
                        if cosine_similarity.ndim > 1:
                            # Average across batch dimension if present
                            cosine_similarity = cosine_similarity.mean(dim=0)
                        if euclidian_distance.ndim > 1:
                            # Average across batch dimension if present
                            euclidian_distance = euclidian_distance.mean(dim=0)
                        if reconstruction_loss.ndim > 1:
                            # Average across batch dimension if present
                            reconstruction_loss = reconstruction_loss.mean(dim=0)
                        
                        # Now average across sequence length to get a single value per forward pass
                        cosine_val = cosine_similarity.mean().item()
                        euclidian_val = euclidian_distance.mean().item()
                        reconstruction_loss_val = reconstruction_loss.mean().item()
                        
                        output_data_cosine_similarity[alpha].append(cosine_val)
                        output_data_euclidian_distance[alpha].append(euclidian_val)
                        output_data_reconstruction_loss[alpha].append(reconstruction_loss_val)
                    
                    logger.debug(f"Hook completed. Cosine dict length: {len(output_data_cosine_similarity.get(list(output_data_cosine_similarity.keys())[0], []))}")
                    
            except Exception as e:
                logger.error(f"Error in steering hook: {e}")
                traceback.print_exc()
                # Don't raise - return activations unchanged to allow generation to continue
                return activations

            return activations

        return steering_hook

    def _create_steering_hook_for_alpha(
        self, 
        alpha: float, 
        cosine_similarities: list[float], 
        euclidian_distances: list[float], 
        relative_changes: list[float],
        confidence_scores: list[float],
        steered_token_positions: Optional[Union[List[int], Set[int], str]] = None
    ):
        """Create hook function for steering with token position control."""
        
        def steering_hook(activations, hook):
            try:
                with torch.no_grad():
                    if activations is None or activations.numel() == 0:
                        logger.warning("Warning: empty activations, skipping steering.")
                        return activations

                    if torch.isnan(activations).any() or torch.isinf(activations).any():
                        logger.error("ERROR: activations contain NaN or Inf; skipping steering.")
                        return activations
                    
                    # Determine current token position
                    # activations shape: [batch, seq_len, hidden_dim]
                    batch_size, seq_len, hidden_dim = activations.shape
                    current_token_position = seq_len - self._prompt_length - 1  # 0-indexed relative to prompt end
                    
                    # Check if we should steer at this position
                    if not self._should_steer_token(current_token_position, steered_token_positions):
                        return activations
                    
                    # Encode
                    sae_features = self.sae.encode(activations)
                    
                    if torch.isnan(sae_features).any() or torch.isinf(sae_features).any():
                        logger.error("ERROR: SAE encode produced NaN or Inf; skipping.")
                        return activations

                    delta_x = self.steering_mechanism.compute_steering_with_alpha(
                        sae_features,
                        alpha=alpha
                    )
                    
                    if torch.isnan(delta_x).any() or torch.isinf(delta_x).any():
                        logger.warning(f"ERROR: delta_x contains NaN or Inf for alpha {alpha}; skipping.")
                        return activations
                    
                    #steered_feats = sae_features + delta_x
                    delta_activation = self.sae.decode(sae_features + delta_x) - self.sae.decode(sae_features)
    
                    
                    if torch.isnan(delta_activation).any() or torch.isinf(delta_activation).any():
                        logger.warning(f"ERROR: steered_feats contain NaN/Inf for alpha {alpha}; skipping.")
                        return activations
                    
                    #steered_activations = self.sae.decode(steered_feats).to(activations.dtype)
                    steered_activations = activations + delta_activation
                        
                    # Handle multi-dimensional tensors - compute mean across batch and sequence dimensions
                    # activations shape is typically [batch, seq_len, hidden_dim]
                    cosine_similarity = torch.nn.functional.cosine_similarity(
                        steered_activations, activations, dim=-1
                    )
                    euclidian_distance = torch.norm(steered_activations - activations, dim=-1)

                    confidence_score = self.steering_mechanism.calculate_confidence(sae_features + delta_x)
                    
                    cosine_val = cosine_similarity.mean().item()
                    euclidian_val = euclidian_distance.mean().item()
                    
                    cosine_similarities.append(cosine_val)
                    euclidian_distances.append(euclidian_val)
                    confidence_scores.append(confidence_score.item())

                    activation_norm = torch.norm(activations, dim=-1).mean().item()
                    relative_change = euclidian_val / activation_norm if activation_norm > 0 else 0.0
                    relative_changes.append(relative_change)
                    
            except Exception as e:
                logger.error(f"Error in steering hook: {e}")
                traceback.print_exc()
                # Don't raise - return activations unchanged to allow generation to continue
                return activations

            return steered_activations

        return steering_hook
    
    def generate_samples(
        self,
        prompt: str,
        max_new_tokens: int = 500,
        temperature: float = 0.7,
        current_output_data: dict = {},
        alphas: list[float] = [0.01, 0.03, 0.05, 0.07, 0.09, 0.11, 0.13, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        steered_token_positions: Optional[Union[List[int], Set[int], str]] = None,
        **kwargs
    ) -> str:
        """
        Generate text with optional steering.
        
        Args:
            prompt: Input prompt
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            current_output_data: Dictionary to store generated samples
            
        Returns:
            Generated text (without prompt)
        """
        # CRITICAL: Reset model state before processing a new prompt
        # This ensures no state from previous prompts persists
        self.model.reset_hooks()
        
        # Tokenize input
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            add_special_tokens=True  
        ).to(self.device)
        
        prompt_length = inputs.input_ids.shape[1]
        self._prompt_length = prompt_length
        self._token_position = 0

        for alpha in alphas:
            current_output_data[alpha] = {}

            cosine_similarities = []
            euclidian_distances = []
            relative_changes = []
            confidence_scores = []
            
            # CRITICAL: Reset all hooks and caches before each generation
            # This prevents state from previous alphas/prompts from persisting
            self.model.reset_hooks()
            
            # Set up steering hook if steered generation
            hook_point = self.sae.cfg.metadata.hook_name
            steering_fn = self._create_steering_hook_for_alpha(
                alpha, 
                cosine_similarities, 
                euclidian_distances, 
                relative_changes,
                confidence_scores,
                steered_token_positions=steered_token_positions
            )
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

                current_output_data[alpha]["average_cosine_similarity"] = sum(cosine_similarities) / len(cosine_similarities)
                current_output_data[alpha]["average_euclidian_distance"] = sum(euclidian_distances) / len(euclidian_distances)
                current_output_data[alpha]["average_relative_change"] = sum(relative_changes) / len(relative_changes)
                current_output_data[alpha]["average_confidence"] = sum(confidence_scores) / len(confidence_scores)
                
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

                current_output_data[alpha]["generated_text"] = generated_text

                print(f"Number of new tokens: {len(output_tokens) - prompt_length}")

                print(f"Output data for alpha {alpha}: {current_output_data[alpha]}")

            except Exception as e:
                logger.error(f"Error in generate_samples: {e}")
                traceback.print_exc()
                current_output_data[alpha] = {}
                current_output_data[alpha]["error"] = str(e)
                current_output_data[alpha]["cosine_similarities"] = None
                current_output_data[alpha]["euclidian_distances"] = None
                current_output_data[alpha]["relative_changes"] = None
                current_output_data[alpha]["confidence_scores"] = None
                current_output_data[alpha]["generated_text"] = None
                return current_output_data
                
            finally:
                # Remove hook
                if self.steering_hook is not None:
                    self.steering_hook.remove()
                    self.steering_hook = None
        
        return current_output_data

    def retrieve_metrics(
        self,
        prompt,
        max_new_tokens,
        output_data_cosine_similarity,
        output_data_euclidian_distance,
        output_data_reconstruction_loss,
        steered_token_positions: Optional[Union[List[int], Set[int], str]] = None
    ):
        """Retrieve metrics for the given prompt."""
        # CRITICAL: Reset model state before processing a new prompt
        # This ensures no state from previous prompts persists
        self.model.reset_hooks()
        
        # Tokenize input
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            add_special_tokens=True  
        ).to(self.device)
        prompt_length = inputs.input_ids.shape[1]
        self._prompt_length = prompt_length
        self._token_position = 0

        hook_point = self.sae.cfg.metadata.hook_name
        logger.info(f"Setting up hook at: {hook_point}")
        logger.info(f"Initial dict sizes - Cosine: {[len(v) for v in output_data_cosine_similarity.values()]}, Euclidean: {[len(v) for v in output_data_euclidian_distance.values()]}, Reconstruction: {[len(v) for v in output_data_reconstruction_loss.values()]}")
        
        steering_fn = self._create_steering_hook(
            output_data_cosine_similarity, 
            output_data_euclidian_distance,
            output_data_reconstruction_loss,
            steered_token_positions=steered_token_positions
        )
        self.steering_hook = self.model.add_hook(hook_point, steering_fn)
        logger.info(f"Hook registered successfully")
        
        try:
            # Generate
            logger.info(f"Starting generation with max_new_tokens={max_new_tokens}")
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs.input_ids,
                    max_new_tokens=max_new_tokens,
                    temperature=0.7,
                    do_sample=True,
                    stop_at_eos=True,
                    prepend_bos=False
                )
            
            logger.info(f"Generation completed. Output shape: {outputs.shape if hasattr(outputs, 'shape') else type(outputs)}")
            logger.info(f"Final dict sizes - Cosine: {[len(v) for v in output_data_cosine_similarity.values()]}, Euclidean: {[len(v) for v in output_data_euclidian_distance.values()]}, Reconstruction: {[len(v) for v in output_data_reconstruction_loss.values()]}")
            
        except Exception as e:
            logger.error(f"Error during generation: {e}")
            traceback.print_exc()
            raise
        finally:
            # Remove hook
            if self.steering_hook is not None:
                self.steering_hook.remove()
                self.steering_hook = None
                logger.info("Hook removed")
        
        return output_data_cosine_similarity, output_data_euclidian_distance, output_data_reconstruction_loss

def run_on_gpu(
    rank,
    author_subsets,
    steered_features,
    path_to_logreg_models,
    output_path_cosine_similarity,
    output_path_euclidian_distance,
    output_path_reconstruction_loss,
    test_data,
    model_name,
    sae_release,
    sae_id,
    steered_token_positions: Optional[Union[List[int], Set[int], str]] = None,
    max_test_entries: Optional[int] = None
):
    """
    Run text generation for authors assigned to the given GPU rank.
    
    Args:
        rank: GPU rank (0, 1, 2, ...)
        author_subsets: List of lists, where author_subsets[rank] contains authors for this GPU
        steered_features: Dictionary mapping author to list of feature indices
        path_to_logreg_models: Path to directory containing classifier models
        output_path_cosine_similarity: Path to directory for cosine similarity output
        output_path_euclidian_distance: Path to directory for euclidian distance output
        output_path_reconstruction_loss: Path to directory for reconstruction loss output
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
        file_name_cosine_similarity = str(output_path_cosine_similarity).format(author)
        file_name_euclidian_distance = str(output_path_euclidian_distance).format(author)
        file_name_reconstruction_loss = str(output_path_reconstruction_loss).format(author)

        output_data_cosine_similarity = {}
        output_data_euclidian_distance = {}
        output_data_reconstruction_loss = {}

        alphas = [0.0, 0.01, 0.03, 0.05, 0.07, 0.09, 0.11, 0.13, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        for alpha in alphas:
            output_data_cosine_similarity[alpha] = []
            output_data_euclidian_distance[alpha] = []
            output_data_reconstruction_loss[alpha] = []

        try:
            classifier_path = path_to_logreg_models / f"logreg_model__res__15__{author}__shap_best_16.pkl"
            
            if author not in steered_features:
                logger.warning(f"Author {author} not found in steered_features, skipping")
                continue
            
            config = SteeringConfig(
                target_layer=15,
                subset_features=steered_features[author],
                alpha=0.01,  # Low steering strength
                lambda_rec=10.0,
                mu_norm=0.01,
                max_iterations=20,
                target_confidence=0.7,
                num_features=16384
            )
            
            # Initialize generator
            logger.info(f"GPU {rank}: Initializing generator for author {author}")
            generator = SAESteeringAmplitudeGenerator(
                model_name=model_name,
                sae_release=sae_release,
                sae_id=sae_id,
                classifier_path=classifier_path,
                device=device
            )
            generator.set_steering_mechanism("heuristic", config)
            
            for index, entry in enumerate(test_data):
                if max_test_entries is not None and index >= max_test_entries:
                    break
                prompt = entry["prompt"]
                output_data_cosine_similarity, output_data_euclidian_distance, output_data_reconstruction_loss = generator.retrieve_metrics(
                    prompt,
                    max_new_tokens=500,
                    output_data_cosine_similarity=output_data_cosine_similarity,
                    output_data_euclidian_distance=output_data_euclidian_distance,
                    output_data_reconstruction_loss=output_data_reconstruction_loss,
                    steered_token_positions=steered_token_positions
                )
            
            # Cleanup generator for this author
            del generator
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # Debug: Check data before writing
            logger.info(f"Before writing CSV for {author}:")
            logger.info(f"Cosine similarity dict keys: {list(output_data_cosine_similarity.keys())}")
            logger.info(f"Cosine similarity dict lengths: {[(k, len(v)) for k, v in output_data_cosine_similarity.items()]}")
            logger.info(f"Euclidean distance dict lengths: {[(k, len(v)) for k, v in output_data_euclidian_distance.items()]}")
            logger.info(f"Reconstruction loss dict lengths: {[(k, len(v)) for k, v in output_data_reconstruction_loss.items()]}")
            
            # Check if data is empty
            total_cosine_values = sum(len(v) for v in output_data_cosine_similarity.values())
            total_euclidean_values = sum(len(v) for v in output_data_euclidian_distance.values())
            total_reconstruction_values = sum(len(v) for v in output_data_reconstruction_loss.values())
            
            if total_cosine_values == 0:
                logger.warning(f"WARNING: No cosine similarity data collected for {author}!")
            if total_euclidean_values == 0:
                logger.warning(f"WARNING: No euclidean distance data collected for {author}!")
            if total_reconstruction_values == 0:
                logger.warning(f"WARNING: No reconstruction loss data collected for {author}!")
            
            # Write CSV files
            df_cosine = pd.DataFrame.from_dict(output_data_cosine_similarity)
            df_euclidean = pd.DataFrame.from_dict(output_data_euclidian_distance)
            df_reconstruction = pd.DataFrame.from_dict(output_data_reconstruction_loss)
            
            logger.info(f"DataFrame shapes - Cosine: {df_cosine.shape}, Euclidean: {df_euclidean.shape}, Reconstruction: {df_reconstruction.shape}")
            
            df_cosine.to_csv(file_name_cosine_similarity)
            df_euclidean.to_csv(file_name_euclidian_distance)
            df_reconstruction.to_csv(file_name_reconstruction_loss)
            
            logger.info(f"CSV files written: {file_name_cosine_similarity}, {file_name_euclidian_distance}, {file_name_reconstruction_loss}")
            
        except Exception as e:
            logger.error(f"GPU {rank}: Error processing author {author}: {e}")
            traceback.print_exc()
            continue
    
    logger.info(f"GPU {rank}: Completed processing all assigned authors")


def run_sample_generation_on_gpu(
    rank,
    author_subsets,
    steered_features,
    path_to_logreg_models,
    output_path_samples_file,
    test_data,
    model_name,
    sae_release,
    sae_id,
    steered_token_positions: Optional[Union[List[int], Set[int], str]] = None,
    max_test_entries: Optional[int] = None
):
    """
    Run text generation for authors assigned to the given GPU rank.
    
    Args:
        rank: GPU rank (0, 1, 2, ...)
        author_subsets: List of lists, where author_subsets[rank] contains authors for this GPU
        steered_features: Dictionary mapping author to list of feature indices
        path_to_logreg_models: Path to directory containing classifier models
        output_path_samples_file: Path to file to save  for samples output
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

        output_data = {}

        alphas = [0, 0.01, 0.03, 0.05, 0.07, 0.09, 0.11, 0.13, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

        try:
            classifier_path = path_to_logreg_models / f"logreg_model__res__15__{author}__shap_best_16.pkl"
            
            if author not in steered_features:
                logger.warning(f"Author {author} not found in steered_features, skipping")
                continue
            
            config = SteeringConfig(
                target_layer=15,
                subset_features=steered_features[author],
                alpha=0.01,  # Low steering strength
                lambda_rec=10.0,
                mu_norm=0.01,
                max_iterations=20,
                target_confidence=0.7,
                num_features=16384
            )
            
            # Initialize generator
            logger.info(f"GPU {rank}: Initializing generator for author {author}")
            generator = SAESteeringAmplitudeGenerator(
                model_name=model_name,
                sae_release=sae_release,
                sae_id=sae_id,
                classifier_path=classifier_path,
                device=device
            )
            generator.set_steering_mechanism("heuristic", config)
            
            for index, entry in enumerate(test_data):
                if max_test_entries is not None and index >= max_test_entries:
                    break
                prompt = entry["prompt"]
                output_data[index] = {}
                output_data[index]["prompt"] = prompt
                generations = generator.generate_samples(
                    prompt,
                    max_new_tokens=500,
                    current_output_data={},  # Pass empty dict to avoid modifying output_data[index] directly
                    alphas=alphas,
                    steered_token_positions=steered_token_positions
                )
                output_data[index]["generations"] = generations
            
            # Cleanup generator for this author
            del generator
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # Save output data
            with open(str(output_path_samples_file).format(author), "w") as f:
                json.dump(output_data, f, indent=2, ensure_ascii=False)
            
        except Exception as e:
            logger.error(f"GPU {rank}: Error processing author {author}: {e}")
            traceback.print_exc()
            continue
    
    logger.info(f"GPU {rank}: Completed processing all assigned authors")


def parse_steered_token_positions(token_arg: Optional[str]) -> Optional[Union[List[int], Set[int], str]]:
    """
    Parse steered_token_positions argument.
    
    Supports:
    - None or "all": steer all tokens
    - "first_n": steer first n tokens (e.g., "first_10")
    - Comma-separated list: steer specific positions (e.g., "0,1,2,5,10")
    - Range: steer range of positions (e.g., "0-10" for positions 0 through 10)
    """
    if token_arg is None or token_arg.lower() == "all":
        return None
    
    token_arg = token_arg.strip()
    
    # Handle "first_n" format
    if token_arg.startswith("first_"):
        return token_arg
    
    # Handle comma-separated list
    if "," in token_arg:
        try:
            positions = [int(x.strip()) for x in token_arg.split(",")]
            return set(positions)
        except ValueError:
            logger.warning(f"Invalid token positions format: {token_arg}, defaulting to all")
            return None
    
    # Handle range format "start-end"
    if "-" in token_arg:
        try:
            start, end = token_arg.split("-")
            positions = list(range(int(start.strip()), int(end.strip()) + 1))
            return set(positions)
        except ValueError:
            logger.warning(f"Invalid token range format: {token_arg}, defaulting to all")
            return None
    
    # Single number
    try:
        return [int(token_arg)]
    except ValueError:
        logger.warning(f"Invalid token positions format: {token_arg}, defaulting to all")
        return None


# Example usage
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run steering amplitude exploration")
    parser.add_argument(
        "--mode",
        type=str,
        choices=["metrics", "samples"],
        default="samples",
        help="Mode: 'metrics' for retrieve_metrics, 'samples' for generate_samples"
    )
    parser.add_argument(
        "--steered-token-positions",
        type=str,
        default=None,
        help="Token positions to steer. Options: 'all' (default), 'first_n' (e.g., 'first_10'), comma-separated list (e.g., '0,1,2,5'), or range (e.g., '0-10')"
    )
    parser.add_argument(
        "--max-test-entries",
        type=int,
        default=20,
        help="Maximum number of test entries to process per author (for testing)"
    )
    
    args = parser.parse_args()
    
    # Configuration
    model_name = "google/gemma-2-9b-it"
    sae_release = "gemma-scope-9b-pt-res"
    sae_id = "layer_15/width_16k/average_l0_131"
    
    # Parse steered token positions
    steered_token_positions = parse_steered_token_positions(args.steered_token_positions)
    logger.info(f"Steered token positions: {steered_token_positions}")
    
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
    logger.info(f"Starting multi-GPU text generation in '{args.mode}' mode...")
    start_time = time.time()
    
    if args.mode == "metrics":
        # Use run_on_gpu for metrics collection
        if world_size > 1:
            mp.spawn(
                run_on_gpu,
                args=(
                    author_subsets,
                    steered_features,
                    path_to_logreg_models,
                    output_path_cosine_similarity,
                    output_path_euclidian_distance,
                    output_path_reconstruction_loss,
                    test_data,
                    model_name,
                    sae_release,
                    sae_id,
                    steered_token_positions,
                    args.max_test_entries
                ),
                nprocs=world_size,
                join=True
            )
        else:
            # Single process mode (CPU or single GPU)
            run_on_gpu(
                0,
                author_subsets,
                steered_features,
                path_to_logreg_models,
                output_path_cosine_similarity,
                output_path_euclidian_distance,
                output_path_reconstruction_loss,
                test_data,
                model_name,
                sae_release,
                sae_id,
                steered_token_positions,
                args.max_test_entries
            )
    else:
        # Use run_sample_generation_on_gpu for sample generation
        if world_size > 1:
            mp.spawn(
                run_sample_generation_on_gpu,
                args=(
                    author_subsets,
                    steered_features,
                    path_to_logreg_models,
                    output_path_samples,
                    test_data,
                    model_name,
                    sae_release,
                    sae_id,
                    steered_token_positions,
                    args.max_test_entries
                ),
                nprocs=world_size,
                join=True
            )
        else:
            # Single process mode (CPU or single GPU)
            run_sample_generation_on_gpu(
                0,
                author_subsets,
                steered_features,
                path_to_logreg_models,
                output_path_samples,
                test_data,
                model_name,
                sae_release,
                sae_id,
                steered_token_positions,
                args.max_test_entries
            )
    
    elapsed_time = time.time() - start_time
    logger.info(f"Completed in {elapsed_time:.2f} seconds")
