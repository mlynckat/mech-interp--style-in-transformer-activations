# Steering

This module implements SAE-based steering for text generation using sparse autoencoder (SAE) features.

## Files

### `baseline_text_generation.py`
Generate non-steered texts based on test and train prompts using standard HuggingFace text generation.

### `explore_steering_amplitude.py`
Exploratory runs of the SAE-hooked transformer with different alpha values. Measures euclidean distance and cosine similarity between steered and original reconstructions, along with the corresponding generated texts.

### `get_prompts_for_test.py`
Run an OpenAI model through test or train texts and generate prompts for experiments.

### `prepare_features_for_steering.py`
Run different classification models and SHAP analysis to find the best settings (models and sets of features). Saves necessary data for further usage in steering experiments.

### `steered_text_generation.py`
Generate steered texts via different steering methods (heuristic and projected gradient).

### `steering_methods.py`
Two different methods to compute steering vectors:

#### Heuristic Steering

Computes steering using minimal L2 norm approach based on classifier weights:

$$\mathbf{w}_s = \text{mask} \odot \mathbf{w}$$

$$\hat{\mathbf{w}}_s = \frac{\mathbf{w}_s}{\|\mathbf{w}_s\|_2}$$

$$\Delta \mathbf{x} = \alpha \cdot \hat{\mathbf{w}}_s$$

Where:
- $\mathbf{w}$ — classifier weights mapped to full SAE feature space
- $\text{mask}$ — binary mask for selected features
- $\alpha$ — steering strength parameter
- $\Delta \mathbf{x}$ — steering vector added to SAE activations

#### Projected Gradient Steering (not yet tested)

Optimizes steering via projected gradient ascent with reconstruction loss constraint:

$$\max_{\Delta \mathbf{x}} \quad z_1 - \lambda_{\text{rec}} \cdot \mathcal{L}_{\text{rec}} - \mu_{\text{norm}} \cdot \|\Delta \mathbf{x}\|^2$$

Where:
- $z_1 = \langle \bar{\mathbf{x}}_{\text{steered}}, \mathbf{w} \rangle$ — classifier logit on aggregated features
- $\bar{\mathbf{x}}_{\text{steered}} = \text{mean}_t(\mathbf{x} + \Delta \mathbf{x})$ — token-averaged steered features
- $\mathcal{L}_{\text{rec}} = \|\text{decode}(\mathbf{x} + \Delta \mathbf{x}) - \text{decode}(\mathbf{x})\|^2$ — reconstruction loss
- $\lambda_{\text{rec}}$ — reconstruction loss weight
- $\mu_{\text{norm}}$ — norm penalty weight

The optimization uses Adam and projects $\Delta \mathbf{x}$ back to the feature mask after each step.

