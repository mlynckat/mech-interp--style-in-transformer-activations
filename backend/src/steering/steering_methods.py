import logging
import torch
import torch.nn.functional as F
from abc import ABC, abstractmethod
from enum import Enum
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, List, Union

logger = logging.getLogger(__name__)


class SteeringPosition(Enum):
    """
    Enum specifying where in the sequence to apply steering.
    
    Values:
        ALL: Apply steering at all token positions
        AFTER_PROMPT: Apply steering only after the prompt tokens
        SPECIFIC_RANGE: Apply steering at specific token indices
    """
    ALL = "all"
    AFTER_PROMPT = "after_prompt"
    SPECIFIC_RANGE = "specific_range"


@dataclass
class SteeringConfig:
    """Configuration for steering parameters."""
    target_layer: int = 15
    subset_features: Optional[List[int]] = None
    alpha: float = 0.1  # Moderate steering strength
    lambda_rec: float = 10.0
    mu_norm: float = 0.01
    max_iterations: int = 20
    learning_rate: float = 0.01
    target_confidence: float = 0.8
    num_features: int = 16384
    
    # Early stopping for optimization convergence
    early_stop_patience: int = 5  # Stop if no improvement for N iterations
    early_stop_min_delta: float = 1e-6  # Minimum change to count as improvement
    
    # Steering position control
    steering_position: SteeringPosition = SteeringPosition.ALL
    # For SPECIFIC_RANGE: tuple of (start, end) token indices (inclusive)
    # For AFTER_PROMPT: automatically determined based on prompt_length
    position_start: Optional[int] = None
    position_end: Optional[int] = None
    
    # For SAEDiffSteering: pre-computed diff vector
    sae_diff: Optional[np.ndarray] = None
    
    def should_apply_steering(self, token_idx: int, prompt_length: int = 0, seq_len: int = 1) -> bool:
        """
        Check if steering should be applied at the given token position.
        
        Args:
            token_idx: Current token index in the sequence (0-indexed absolute position)
            prompt_length: Length of the prompt in tokens (for AFTER_PROMPT mode)
            seq_len: Current sequence length being processed (>1 during prompt, 1 during generation)
            
        Returns:
            True if steering should be applied at this position
        """
        if self.steering_position == SteeringPosition.ALL:
            return True
        elif self.steering_position == SteeringPosition.AFTER_PROMPT:
            # Apply steering only at the prompt (when processing multiple tokens at once)
            # seq_len > 1 means we're in the prompt phase, seq_len == 1 means generation
            return seq_len > 1
        elif self.steering_position == SteeringPosition.SPECIFIC_RANGE:
            start = self.position_start or 0
            end = self.position_end or float('inf')
            return start <= token_idx <= end
        return True


class SteeringMechanism(ABC):
    """Abstract base class for steering mechanisms."""
    
    def __init__(self, config: SteeringConfig, classifier_weights: np.ndarray, classifier_intercept: float = 0.0):
        self.config = config
        # Map classifier weights (in local order) to full feature space (in global order)
        self.w = self._map_weights_to_full_space(classifier_weights)
        self.mask = self._create_mask()
        # Store classifier intercept (bias term) - CRITICAL for correct probability
        self.intercept = classifier_intercept
        logger.info(f"SteeringMechanism initialized with intercept: {self.intercept}")
    
    def _map_weights_to_full_space(self, classifier_weights: np.ndarray) -> torch.Tensor:
        """
        Map classifier weights from local order to full feature space.
        
        The classifier was trained on a subset of features. The weights are in the
        order those features were selected during training. This method maps them
        to the full 16k feature space at the correct global indices.
        
        Args:
            classifier_weights: Weights in local order, shape (n_selected_features,)
            
        Returns:
            Full weight vector, shape (num_features,) with weights at correct global indices
        """
        # Create full-size weight vector
        w_full = torch.zeros(self.config.num_features, dtype=torch.float32)
        
        # Determine the order of feature indices
        if self.config.subset_features is not None:
            # Use provided ordered indices
            feature_indices = self.config.subset_features
        else:
            raise ValueError("subset_features must be provided")
        
        # Map weights to global indices
        if len(feature_indices) != len(classifier_weights):
            raise ValueError(
                f"Mismatch: feature_indices_ordered has {len(feature_indices)} indices but "
                f"classifier_weights has {len(classifier_weights)} weights."
            )
        
        for local_idx, global_idx in enumerate(feature_indices):
            # Handle both int and string formats (e.g., "x5" -> 5)
            if isinstance(global_idx, str):
                global_idx = int(global_idx.replace("x", ""))
            w_full[global_idx] = classifier_weights[local_idx]
        
        return w_full
        
    def _create_mask(self) -> torch.Tensor:
        """Create mask for subset of features."""
        if self.config.subset_features is None:
            return torch.ones(self.config.num_features, dtype=torch.float32)
        mask = torch.zeros(self.config.num_features, dtype=torch.float32)
        feature_indices = self.config.subset_features
        for global_idx in feature_indices:
            # Handle both int and string formats (e.g., "x5" -> 5)
            if isinstance(global_idx, str):
                global_idx = int(global_idx.replace("x", ""))
            mask[global_idx] = 1.0
        return mask
    
    @abstractmethod
    def compute_steering(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Compute steering vector for given SAE activations.
        
        This is the core method that each steering mechanism must implement.
        It takes the current SAE feature activations and returns a delta (Δx)
        that should be ADDED to those activations to steer the generation.
        
        Args:
            x: Current SAE feature activations with shape [batch, seq_len, sae_dim]
               - batch: typically 1 during generation
               - seq_len: current sequence length (grows as tokens are generated)
               - sae_dim: number of SAE features (e.g., 16384 for 16k SAE)
               
            **kwargs: Additional arguments specific to each steering method:
                - For HeuristicSteering: auto_tune, decoder, original_tokens
                - For ProjectedGradientSteering: decoder, original_reconstruction
        
        Returns:
            steered_activations: Steered activations with same shape as x [batch, seq_len, sae_dim]
        
        The method is called during generation via a hook that:
        1. Intercepts activations at the target layer
        2. Encodes them to SAE features (x)
        3. Calls this method to get steering delta
        4. Adds delta to features: x_steered = x + delta_x
        5. Decodes steered features back to activation space
        6. Returns steered activations to continue generation
        """
        pass


class HeuristicSteering(SteeringMechanism):
    """Simple heuristic steering based on classifier weights."""
    
    def compute_steering(
        self, 
        x: torch.Tensor,
        auto_tune: bool = False,
        **kwargs
    ) -> torch.Tensor:
        """
        Compute steering using minimal L2 norm approach.
        
        Args:
            x: Current SAE activations [batch, seq_len, sae_dim]
            auto_tune: Whether to automatically tune alpha
            decoder: (optional) SAE decoder for quality checking
            original_activations: (optional) Original activations (before SAE encoding)
            history_feats: (optional) History of SAE features from all previous tokens [batch, n_tokens, sae_dim]: list of tensors
            
        Returns:
            Steered activations (original_activations + delta from steering)
        """
        device = x.device
        # Both mask and w are now full-size (num_features,), so element-wise multiply works
        w_s = (self.mask * self.w).to(device)
        
        # Normalize by L2 norm
        w_s_norm = torch.norm(w_s)
        if w_s_norm < 1e-8:
            # No steering possible, return original activations if available
            original_activations = kwargs.get('original_activations')
            if original_activations is not None:
                return original_activations
            else:
                raise ValueError("original_activations must be provided")
        
        normalized_steering = w_s / w_s_norm
        
        # Scale by alpha
        alpha = self.config.alpha
        if auto_tune:
            final_alpha, steered_activations = self._tune_alpha(
                starting_alpha=alpha, x=x, normalized_steering=normalized_steering, **kwargs
            )
        else:
            # Apply fixed alpha steering
            delta_x = self.compute_steering_with_alpha(x, alpha, **kwargs)
            decoder = kwargs.get('decoder')
            original_activations = kwargs.get('original_activations')
            
            if decoder is not None and original_activations is not None:
                steered_x = x + delta_x
                delta_activations = decoder(steered_x) - decoder(x)
                steered_activations = original_activations + delta_activations
            else:
                # Fallback: return delta_x directly (less accurate)
                steered_activations = x + delta_x
                if decoder is not None:
                    steered_activations = decoder(steered_activations)
            
        return steered_activations

    def compute_steering_with_alpha(
        self, 
        x: torch.Tensor,
        alpha: float,
        **kwargs
    ) -> torch.Tensor:
        """
        Compute steering using minimal L2 norm approach.
        
        Args:
            x: Current SAE activations [batch, seq_len, sae_dim]
            auto_tune: Whether to automatically tune alpha
            decoder: (optional) SAE decoder for quality checking
            original_reconstruction: (optional) Original decoded activations
            history_feats: (optional) History of SAE features from all previous tokens [batch, n_tokens, sae_dim]: list of tensors
            
        Returns:
            Steering vector delta_x
        """
        device = x.device
        # Both mask and w are now full-size (num_features,), so element-wise multiply works
        w_s = (self.mask * self.w).to(device)
        
        # Normalize by L2 norm
        w_s_norm = torch.norm(w_s)
        if w_s_norm < 1e-8:
            return torch.zeros_like(x)
        
        normalized_steering = w_s / w_s_norm
        
        # Scale by alpha
        # Broadcast to match input shape
        delta_x = alpha * normalized_steering
        # Expand dimensions to match [batch, seq_len, sae_dim]
        while delta_x.ndim < x.ndim:
            delta_x = delta_x.unsqueeze(0)
        
        # Ensure delta_x can broadcast to x's shape
        # If x is [batch, seq_len, sae_dim] and delta_x is [1, 1, sae_dim], expand it
        if delta_x.shape != x.shape:
            delta_x = delta_x.expand_as(x)
            
        return delta_x
    
    def confidence_achieved(self, x_vector: torch.Tensor) -> bool:
        """
        Check if confidence threshold is achieved on current token(s).
        
        Note: This is a per-token confidence check (not aggregated over sequence).
        For document-level aggregated confidence matching training, use 
        confidence_achieved_aggregated() instead.
        """
        logit = (x_vector * self.w.to(x_vector.device)).sum(dim=-1).mean() + self.intercept
        confidence = torch.sigmoid(logit)
        logger.debug(f"Confidence: {confidence} (logit: {logit}, intercept: {self.intercept})")
        if confidence > self.config.target_confidence:
            logger.debug(f"Breaking due to confidence: {confidence}")
            return True
        return False

    def calculate_confidence(self, x_vector: torch.Tensor) -> float:
        """
        Calculate confidence on current token(s).
        
        """
        logit = (x_vector * self.w.to(x_vector.device)).sum(dim=-1).mean() + self.intercept
        confidence = torch.sigmoid(logit)
        
        return confidence

    def confidence_achieved_aggregated(self, x_vector: torch.Tensor, threshold: float = 0.7, skip_prompt_tokens: int = 0) -> bool:
        """
        Check if confidence threshold is achieved on aggregated activations.
        
        This method aggregates SAE activations over tokens by averaging, then computes 
        classifier confidence. This matches how the classifier was trained on document-level 
        aggregated data.
        
        IMPORTANT: Training uses from_token=10 to skip prompt tokens. When x_vector comes
        from history_feats, the first token (last prompt token) should already be skipped
        before calling this method. Use skip_prompt_tokens parameter if prompt tokens
        might still be included.
        
        Args:
            x_vector: SAE activations with shape [batch, n_tokens, sae_dim]
            threshold: Confidence threshold (default 0.7 for 70%)
            skip_prompt_tokens: Number of tokens to skip from start (default 0, assumes already skipped)
            
        Returns:
            True if confidence >= threshold, False otherwise
        """
        # Skip prompt tokens if needed (matching training which uses from_token=10)
        if x_vector.shape[1] <= skip_prompt_tokens:
            # Not enough tokens yet
            return False
        
        # Aggregate over sequence dimension (axis=1) by averaging, skipping prompt tokens if needed
        # This matches the aggregation in retrieve_and_combine_author_data_aggregated
        # which does: doc_tokens.sum(axis=0) / n_valid_tokens (equivalent to mean)
        if skip_prompt_tokens > 0:
            x_aggregated = x_vector[:, skip_prompt_tokens:, :].mean(dim=1)  # [batch, sae_dim]
        else:
            x_aggregated = x_vector.mean(dim=1)  # [batch, sae_dim]
        
        # Compute confidence: dot product with weights + intercept, then sigmoid
        w = self.w.to(x_aggregated.device)
        # x_aggregated: [batch, sae_dim], w: [sae_dim]
        # Sum over feature dimension, then average over batch, then add intercept
        logit = (x_aggregated * w).sum(dim=-1).mean() + self.intercept
        confidence = torch.sigmoid(logit)
        
        logger.debug(f"Aggregated confidence: {confidence.item():.4f} (threshold: {threshold}, n_tokens: {x_vector.shape[1] - skip_prompt_tokens})")
        if confidence >= threshold:
            logger.debug(f"Breaking alpha tuning due to aggregated confidence >= {threshold}")
            return True
        return False

        

    def _tune_alpha(
        self, 
        starting_alpha: float,
        x: torch.Tensor, 
        normalized_steering: torch.Tensor,
        decoder=None,
        original_activations=None,  # Fixed: proper terminology
        history_feats=None,  # History of SAE features from all previous tokens
        **kwargs
    ) -> tuple[float, torch.Tensor]:
        """
        Tune alpha using line search.
        Start at 0.01, multiply by 2 until quality degrades.
        
        Args:
            x: Current SAE features
            normalized_steering: Unit-norm steering direction
            decoder: SAE decoder function
            original_activations: Original activations (for comparison)
            history_feats: History of SAE features from all previous tokens [batch, n_tokens, sae_dim]: list of tensors
        """
        alpha = starting_alpha
        final_alpha = alpha

        if self.confidence_achieved(x):
            if history_feats is not None:
                history_feats.append(x.detach().clone())
            return 0, original_activations
        
        for _ in range(10):  # Max 10 iterations

            test_steering = alpha * normalized_steering
            
            # Expand dimensions
            while test_steering.ndim < x.ndim: 
                test_steering = test_steering.unsqueeze(0) # from torch.Size([16384]) to torch.Size([1, 1, 16384])

            steered_x = x + test_steering
            
            delta_activations = decoder(steered_x) - decoder(x)

            steered_activations = original_activations + delta_activations
            
            final_alpha = alpha
            logger.debug(f"Final alpha: {final_alpha}")
            
            # Check classifier confidence on aggregated activations
            # The classifier was trained on document-level aggregated data, so we should
            # aggregate all previously generated tokens (including current) before checking confidence
            # IMPORTANT: history_feats[0] is the last prompt token, so skip it to match training
            if history_feats is not None and len(history_feats) > 1:
                # Skip first element (last prompt token) to match training which uses from_token=10
                generated_feats = history_feats[1:]  # Only generated tokens
                history_feats_tensor = torch.stack(generated_feats + [steered_x.detach().clone()], dim=1)

                if self.confidence_achieved_aggregated(history_feats_tensor, threshold=0.7, skip_prompt_tokens=1):
                    history_feats.append(steered_x.detach().clone())
                    break
            
            # Also check per-token confidence (heuristic check, not matching training)
            # For aggregated confidence matching training, see confidence_achieved_aggregated above
            if self.confidence_achieved(steered_x):
                if history_feats is not None:
                    history_feats.append(steered_x.detach().clone())
                break

            alpha *= 1.5
            
                
        return final_alpha, steered_activations


class ProjectedGradientSteering(SteeringMechanism):
    """
    Projected gradient ascent steering with reconstruction loss.
    
    This steering method optimizes a perturbation delta_x in SAE feature space
    to maximize classifier confidence while minimizing reconstruction loss.
    
    IMPORTANT: This method requires full gradient support during generation.
    Use with manual_generate() in steered_text_generation.py, NOT with 
    model.generate() which wraps generation in torch.no_grad().
    
    Gradient flow:
    - delta_x is a leaf tensor with requires_grad=True
    - Optimization: loss.backward() computes gradients for delta_x
    - Model weights and SAE weights are NOT updated (only delta_x is optimized)
    - decoder() must allow gradients through for reconstruction loss computation
    """
    
    def compute_steering(
        self,
        x: torch.Tensor,
        decoder,
        return_history: bool = False,
        **kwargs
    ) -> torch.Tensor:
        """
        Optimize steering via projected gradient ascent.
        
        This method runs a local optimization loop to find the best perturbation
        delta_x that maximizes classifier confidence while preserving reconstruction
        quality. Gradients flow through the decoder for the reconstruction loss.
        
        Args:
            x: Current SAE activations [batch, seq_len, sae_dim]
            decoder: SAE decoder function (must support gradient computation)
            original_reconstruction: Original decoded output for comparison (no gradients)
            return_history: Whether to return optimization history
            
        Returns:
            Steered activations (decoded from x + optimized delta_x)
        """
        device = x.device
        w = self.w.to(device)
        mask = self.mask.to(device)

        original_activations = kwargs.get('original_activations')
        
        # CRITICAL: Ensure gradients are enabled for the optimization loop
        # This is required for loss.backward() to compute gradients for delta_x
        # Note: The hook should wrap this call in torch.enable_grad() context
        if not torch.is_grad_enabled():
            raise RuntimeError(
                "ProjectedGradientSteering requires gradients to be enabled. "
                "The hook should wrap this call in torch.enable_grad() context."
            )
        
        logger.info(
            f"[ProjectedGradientSteering] Starting compute_steering with "
            f"batch_shape={tuple(x.shape)}, lr={self.config.learning_rate}, "
            f"max_iters={self.config.max_iterations}, "
            f"target_confidence={self.config.target_confidence}"
        )
        
        # Detach x to ensure we don't backprop through the model
        # We only need gradients for delta_x optimization
        x = x.detach()
        
        # PRE-COMPUTE: Cache decoder(x) once since x never changes
        # This avoids redundant decoder calls later
        with torch.no_grad():
            decoder_x_cached = decoder(x)
        
        # PRE-COMPUTE: Expand mask once instead of per-iteration
        mask_expanded = mask.unsqueeze(0).unsqueeze(0)
        
        # Initialize delta_x as a leaf tensor with gradients enabled
        # This is the only tensor being optimized - model/SAE weights are frozen
        delta_x = torch.zeros_like(x, requires_grad=True)
        
        optimizer = torch.optim.Adam([delta_x], lr=self.config.learning_rate)
        
        history = {
            'loss': [],
            'classifier_logit': [],
            'rec_loss': [],
            'norm_penalty': []
        } if return_history else None
        
        final_classifier_prob = None
        iterations_run = 0
        
        # Early stopping tracking
        best_loss = float('inf')
        no_improvement_count = 0
        
        for iteration in range(self.config.max_iterations):
            optimizer.zero_grad()
            
            # Apply mask to delta_x (use pre-computed expanded mask)
            masked_delta = delta_x * mask_expanded
            x_steered = x + masked_delta
            
            # Compute classifier logit (dot product with weights + intercept)
            # IMPORTANT: Aggregate features first (matching training approach)
            # x_steered shape: [batch, seq_len, sae_dim] -> [batch, sae_dim]
            x_aggregated = x_steered.mean(dim=1)
            # Then compute logit on aggregated features (don't forget intercept!)
            z1 = (x_aggregated * w).sum(dim=-1).mean() + self.intercept
            
            # Compute reconstruction loss
            # Compares decoded ACTIVATIONS, not tokens!
            reconstructed = decoder(x_steered)
            rec_loss = F.mse_loss(reconstructed, original_activations)
            
            # Norm penalty (use sum instead of norm squared for slightly faster compute)
            norm_penalty = (masked_delta ** 2).sum()
            
            # Total loss to maximize (we'll negate for minimization)
            loss = -(z1 - self.config.lambda_rec * rec_loss - 
                    self.config.mu_norm * norm_penalty)

            # Extract scalar values before backward (avoids repeated .item() calls)
            current_loss = loss.item()
            z1_val = z1.item()
            rec_loss_val = rec_loss.item()
            norm_penalty_val = norm_penalty.item()
            classifier_prob_val = torch.sigmoid(z1).item()
            
            final_classifier_prob = classifier_prob_val
            iterations_run = iteration + 1
            
            logger.debug(
                f"[ProjectedGradientSteering][iter {iteration}] "
                f"logit={z1_val:.4f}, prob={classifier_prob_val:.4f}, "
                f"rec_loss={rec_loss_val:.6f}, "
                f"norm_penalty={norm_penalty_val:.6f}, "
                f"loss={current_loss:.6f}"
            )
            
            loss.backward()
            optimizer.step()
            
            # Project back to mask (use pre-computed expanded mask)
            with torch.no_grad():
                delta_x.data.mul_(mask_expanded)
            
            # Store history if needed
            if history is not None:
                history['loss'].append(current_loss)
                history['classifier_logit'].append(z1_val)
                history['rec_loss'].append(rec_loss_val)
                history['norm_penalty'].append(norm_penalty_val)
            
            # Early stopping check: loss convergence
            loss_improvement = best_loss - current_loss
            if loss_improvement > self.config.early_stop_min_delta:
                best_loss = current_loss
                no_improvement_count = 0
            else:
                no_improvement_count += 1
                if no_improvement_count >= self.config.early_stop_patience:
                    logger.info(
                        f"[ProjectedGradientSteering] Early stopping at iter {iteration}: "
                        f"no improvement for {no_improvement_count} iterations"
                    )
                    break
            
            # Stopping criteria: confidence threshold
            if classifier_prob_val > self.config.target_confidence:
                logger.info(
                    f"[ProjectedGradientSteering] Stopping at iter {iteration} "
                    f"because prob {classifier_prob_val:.4f} "
                    f"exceeded target {self.config.target_confidence:.4f}"
                )
                if iteration == 0:
                    if return_history:
                        return original_activations, history
                    return original_activations
                break
        
        # Compute final steered activations using cached decoder(x)
        with torch.no_grad():
            final_delta = delta_x.detach() * mask_expanded
            final_x_steered = x + final_delta
            steered_activations = decoder(final_x_steered)
            
            # Use cached decoder(x) instead of calling decoder again
            delta_activations = steered_activations - decoder_x_cached
            steered_activations_clean = original_activations + delta_activations
        
        logger.info(
            f"[ProjectedGradientSteering] Finished compute_steering with "
            f"final_prob={final_classifier_prob if final_classifier_prob is not None else float('nan'):.4f}, "
            f"total_iters={iterations_run}, "
            f"final_loss={current_loss:.4f}"
            f"final_rec_loss={rec_loss_val:.4f}"
            f"final_norm_penalty={norm_penalty_val:.4f}"
    )
        
        if return_history:
            return steered_activations_clean, history

        return steered_activations_clean


class SAEDiffSteering:
    """
    SAE-diff-based steering that adds pre-computed feature diffs to activations.
    
    Unlike other steering methods that use classifier weights, this method uses
    the difference between SAE features of original texts and baseline-generated texts.
    This represents the direction from baseline style to the target author's style.
    
    The diff is typically computed as: diff = mean(original_features) - mean(baseline_features)
    and is loaded from pre-computed files.
    
    This method does NOT require classifier weights, only the pre-computed diff vector.
    """
    
    def __init__(self, config: SteeringConfig):
        """
        Initialize SAEDiffSteering.
        
        Args:
            config: SteeringConfig containing the sae_diff vector
                   The sae_diff should be a numpy array of shape (num_features,)
        """
        self.config = config
        
        if config.sae_diff is None:
            raise ValueError("SAEDiffSteering requires sae_diff to be set in config")
        
        # Convert diff to tensor
        self.diff = torch.from_numpy(config.sae_diff).float()
        
        logger.info(f"SAEDiffSteering initialized with diff L2 norm: {torch.norm(self.diff):.4f}")
    
    def compute_steering(
        self,
        x: torch.Tensor,
        decoder,
        original_activations: torch.Tensor,
        **kwargs
    ) -> torch.Tensor:
        """
        Apply SAE-diff steering to the given activations.
        
        The diff is scaled by alpha and added to SAE features, then decoded back
        to activation space. 
        
        Note: Position-based steering (AFTER_PROMPT, SPECIFIC_RANGE) is handled
        in the hook before calling this method. When this method is called,
        steering should be applied to all positions in x.
        
        Args:
            x: Current SAE feature activations [batch, seq_len, sae_dim]
               During autoregressive generation, seq_len is typically 1
            decoder: SAE decoder function to convert features back to activations
            original_activations: Original activations (before SAE encoding)
            prompt_length: Number of prompt tokens (unused here, position check done in hook)
            
        Returns:
            Steered activations with same shape as original_activations
        """
        device = x.device
        batch_size, seq_len, sae_dim = x.shape
        
        # Move diff to device
        diff = self.diff.to(device)
        
        # Scale diff by alpha (steering strength)
        alpha = self.config.alpha
        scaled_diff = alpha * diff
        
        # Expand diff to match x shape [batch, seq_len, sae_dim]
        # diff: [sae_dim] -> [1, 1, sae_dim] -> [batch, seq_len, sae_dim]
        delta_x = scaled_diff.unsqueeze(0).unsqueeze(0).expand(batch_size, seq_len, sae_dim)
        
        # Add delta to SAE features
        steered_x = x + delta_x
        
        # Compute steering in activation space
        # delta_activations = decoder(steered_x) - decoder(x)
        # steered_activations = original_activations + delta_activations
        delta_activations = decoder(steered_x) - decoder(x)

        steered_activations = original_activations + delta_activations
        
        # Log steering magnitude for verification
        delta_norm = torch.norm(delta_activations).item()
        logger.info(f"SAEDiff steering applied: alpha={alpha:.4f}, delta_norm={delta_norm:.4f}")

        return steered_activations
    
    def compute_steering_delta(
        self,
        seq_len: int,
        device: torch.device = None
    ) -> torch.Tensor:
        """
        Compute the steering delta in SAE feature space.
        
        Useful for pre-computing the delta before decoding.
        
        Args:
            seq_len: Sequence length
            device: Target device
            
        Returns:
            Delta tensor of shape [1, seq_len, sae_dim]
        """
        if device is None:
            device = self.diff.device
        
        diff = self.diff.to(device)
        
        # Scale by alpha and expand
        alpha = self.config.alpha
        scaled_diff = alpha * diff
        delta_x = scaled_diff.unsqueeze(0).unsqueeze(0).expand(1, seq_len, -1)
        
        return delta_x