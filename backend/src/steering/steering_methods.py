import torch
import torch.nn.functional as F
from abc import ABC, abstractmethod
import numpy as np
from dataclasses import dataclass
from typing import Optional, Set, List, Union


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
    
    # CRITICAL: Use relative thresholds, not absolute
    max_relative_rec_loss: float = 0.3  # 30% increase allowed



class SteeringMechanism(ABC):
    """Abstract base class for steering mechanisms."""
    
    def __init__(self, config: SteeringConfig, classifier_weights: np.ndarray):
        self.config = config
        # Map classifier weights (in local order) to full feature space (in global order)
        self.w = self._map_weights_to_full_space(classifier_weights)
        self.mask = self._create_mask()
    
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
            delta_x: Steering vector with same shape as x [batch, seq_len, sae_dim]
                    This vector is ADDED to x to produce steered features:
                    x_steered = x + delta_x
        
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
        alpha = self.config.alpha
        if auto_tune:
            alpha = self._tune_alpha(starting_alpha=alpha, x=x, normalized_steering=normalized_steering, **kwargs)
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
        confidence = torch.sigmoid((x_vector * self.w.to(x_vector.device)).sum(dim=-1).mean())
        print(f"Confidence: {confidence}")
        if confidence > self.config.target_confidence:
            print(f"Breaking due to confidence")
            return True
        return False

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
        
        # Compute confidence: dot product with weights, then sigmoid
        w = self.w.to(x_aggregated.device)
        # x_aggregated: [batch, sae_dim], w: [sae_dim]
        # Sum over feature dimension, then average over batch
        logit = (x_aggregated * w).sum(dim=-1).mean()
        confidence = torch.sigmoid(logit)
        
        print(f"Aggregated confidence: {confidence.item():.4f} (threshold: {threshold}, n_tokens: {x_vector.shape[1] - skip_prompt_tokens})")
        if confidence >= threshold:
            print(f"Breaking alpha tuning due to aggregated confidence >= {threshold}")
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
    ) -> float:
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
        best_alpha = alpha

        if self.confidence_achieved(x):
            return 0
        
        for _ in range(10):  # Max 10 iterations
            test_steering = alpha * normalized_steering
            
            # Expand dimensions
            print(f"Test steering shape pre: {test_steering.shape}")
            while test_steering.ndim < x.ndim:
                test_steering = test_steering.unsqueeze(0)

            print(f"Test steering shape post: {test_steering.shape}")

            print(f"Test steering:")
            for i, value in enumerate(test_steering.squeeze()):
                if value != 0:
                    print(f"  Index {i}: {value}")
            x_steered = x + test_steering
            
            # Check if quality is acceptable by comparing reconstructed activations
            if decoder is not None and original_activations is not None:
                # Decode steered features back to activation space
                steered_reconstruction = decoder(x_steered)
                original_reconstruction = decoder(x)

                print(f"Steered reconstruction shape: {steered_reconstruction.shape}")  
                print(f"Original activations shape: {original_activations.shape}")
                
                # Check if shapes match
                if steered_reconstruction.shape != original_activations.shape:
                    print(f"ERROR: Shape mismatch! Cannot compute MSE loss.")
                    print(f"  Steered: {steered_reconstruction.shape}")
                    print(f"  Original: {original_activations.shape}")
                    # Try to handle shape mismatch by reshaping or skipping
                    break
                
                # Compare in activation space (NOT token space!)
                # This measures how much steering distorted the internal representations
                rec_loss_steered = F.mse_loss(steered_reconstruction, original_activations)
                rec_loss_original = F.mse_loss(original_reconstruction, original_activations)
                rec_loss_value_steered = rec_loss_steered.item()  # Convert tensor to float
                rec_loss_value_original = rec_loss_original.item()  # Convert tensor to float

                # Calculate euclidian distance between steered and original reconstruction
                euclidian_distance = torch.norm(steered_reconstruction - original_activations)
                print(f"Euclidian distance: {euclidian_distance}")

                # Calculate cosine similarity between steered and original reconstruction
                cosine_similarity = torch.nn.functional.cosine_similarity(steered_reconstruction, original_activations, dim=-1)
                print(f"Cosine similarity: {cosine_similarity}")

                
                relative_increase = (rec_loss_value_steered/rec_loss_value_original) - 1.0
                print(f"Relative increase in MSEs between reconstruction losses: {relative_increase} (calculated as {rec_loss_value_steered}/{rec_loss_value_original} - 1.0)")
                if relative_increase > self.config.max_relative_rec_loss:
                    print(f"Breaking due to relative increase (threshold: {self.config.max_relative_rec_loss})")
                    break
            
            best_alpha = alpha
            print(f"Best alpha: {best_alpha}")
            
            # Check classifier confidence on aggregated activations
            # The classifier was trained on document-level aggregated data, so we should
            # aggregate all previously generated tokens (including current) before checking confidence
            # IMPORTANT: history_feats[0] is the last prompt token, so skip it to match training
            if history_feats is not None and len(history_feats) > 1:
                # Skip first element (last prompt token) to match training which uses from_token=10
                generated_feats = history_feats[1:]  # Only generated tokens
                history_feats_tensor = torch.stack(generated_feats, dim=1)
                
                # Check confidence on aggregated (averaged) activations
                if self.confidence_achieved_aggregated(history_feats_tensor, threshold=0.7, skip_prompt_tokens=1):
                    break
           
            
            alpha *= 1.5
            
            # Also check per-token confidence (heuristic check, not matching training)
            # For aggregated confidence matching training, see confidence_achieved_aggregated above
            if self.confidence_achieved(x_steered):
                break
            
                
        return best_alpha


class ProjectedGradientSteering(SteeringMechanism):
    """Projected gradient ascent steering with reconstruction loss."""
    
    def compute_steering(
        self,
        x: torch.Tensor,
        decoder,
        original_reconstruction: torch.Tensor,
        return_history: bool = False,
        **kwargs
    ) -> torch.Tensor:
        """
        Optimize steering via projected gradient ascent.
        
        Args:
            x: Current SAE activations [batch, seq_len, sae_dim]
            decoder: SAE decoder function
            original_reconstruction: Original decoded output for comparison
            return_history: Whether to return optimization history
            
        Returns:
            Optimized steering vector delta_x (and optionally history)
        """
        device = x.device
        w = self.w.to(device)
        mask = self.mask.to(device)
        
        # Initialize delta_x
        delta_x = torch.zeros_like(x, requires_grad=True)
        
        optimizer = torch.optim.Adam([delta_x], lr=self.config.learning_rate)
        
        history = {
            'loss': [],
            'classifier_logit': [],
            'rec_loss': [],
            'norm_penalty': []
        }
        
        for iteration in range(self.config.max_iterations):
            optimizer.zero_grad()
            
            # Apply mask to delta_x
            masked_delta = delta_x * mask.unsqueeze(0).unsqueeze(0)
            x_steered = x + masked_delta
            
            # Compute classifier logit (dot product with weights)
            # IMPORTANT: Aggregate features first (matching training approach)
            # Training aggregates features over tokens before computing logit
            # x_steered shape: [batch, seq_len, sae_dim] -> [batch, sae_dim]
            x_aggregated = x_steered.mean(dim=1)
            # Then compute logit on aggregated features
            z1 = (x_aggregated * w).sum(dim=-1).mean()
            
            # Compute reconstruction loss
            # Compares decoded ACTIVATIONS, not tokens!
            # This measures distortion in the activation space at layer 15
            reconstructed = decoder(x_steered)
            rec_loss = F.mse_loss(reconstructed, original_reconstruction)
            
            # Norm penalty
            norm_penalty = torch.norm(masked_delta) ** 2
            
            # Total loss to maximize (we'll negate for minimization)
            loss = -(z1 - self.config.lambda_rec * rec_loss - 
                    self.config.mu_norm * norm_penalty)
            
            loss.backward()
            optimizer.step()
            
            # Project back to mask
            with torch.no_grad():
                delta_x.mul_(mask.unsqueeze(0).unsqueeze(0))
            
            # Store history
            if return_history:
                history['loss'].append(loss.item())
                history['classifier_logit'].append(z1.item())
                history['rec_loss'].append(rec_loss.item())
                history['norm_penalty'].append(norm_penalty.item())
            
            # Stopping criteria
            with torch.no_grad():
                classifier_prob = torch.sigmoid(z1)
                if classifier_prob > self.config.target_confidence:
                    break
                
                # Check relative reconstruction loss
                if self.config.should_stop_steering(rec_loss.item()):
                    break
        
        result = delta_x.detach() * mask.unsqueeze(0).unsqueeze(0)
        
        if return_history:
            return result, history
        return result