from typing import List, Optional
from pydantic import Field
from dataclasses import dataclass
import logging

# Set up logging
logger = logging.getLogger(__name__)


@dataclass
class ModelConfig:
    model_name: str
    layer_indices: Optional[List[int]] = None
    layer_types: Optional[List[str]] = None

    def __post_init__(self):
        if self.model_name not in ["google/gemma-2-2b", "google/gemma-2-9b", "google/gemma-2-9b-it"]:
            raise ValueError(f"Model {self.model_name} is not supported.")
        max_layers = 26 if self.model_name == "google/gemma-2-2b" else 42
        if self.layer_indices is None:
            self.layer_indices = list(range(max_layers))
        else:
            len_initial = len(self.layer_indices)
            self.layer_indices = [layer for layer in self.layer_indices if layer < max_layers]
            if len(self.layer_indices) < len_initial:
                # Warning
                logger.warning(f"{len_initial - len(self.layer_indices)} layer indices are out of range for model {self.model_name}. Removing them.")
        
        if self.layer_types is None:
            if self.model_name == "google/gemma-2-9b-it" or self.model_name == "google/gemma-2-2b"  :
                self.layer_types = ["res", "mlp", "att"]
            elif self.model_name == "google/gemma-2-9b":
                self.layer_types = ["res"]
            else:
                raise ValueError(f"Model {self.model_name} is not supported.")


@dataclass
class SAELayerConfig:
    model_name: str
    layer_type: str
    layer_index: int
    width: str
    canonical: bool
    l0: Optional[int] = None
    release_name: str = Field(default_factory=str, init=False)
    sae_id: str = Field(default_factory=str, init=False)

    def __post_init__(self):
        self.release_name = None
        if self.canonical:
            if self.model_name == "google/gemma-2-2b":
                if self.layer_type == "res":
                    self.release_name = "gemma-scope-2b-pt-res-canonical"
                elif self.layer_type == "mlp":
                    self.release_name = "gemma-scope-2b-pt-mlp-canonical"
                elif self.layer_type == "att":
                    self.release_name = "gemma-scope-2b-pt-att-canonical"
                else:
                    raise ValueError(f"Unknown layer_type {self.layer_type}. Only res, mlp, and att are supported for google/gemma-2-2b.")
            elif self.model_name == "google/gemma-2-9b":
                if self.layer_type == "res":
                    self.release_name = "gemma-scope-9b-pt-res-canonical"
                else:
                    raise ValueError(f"Unknown layer_type {self.layer_type}. Only res are supported for google/gemma-2-9b in canonical releases.")
            else:
                raise ValueError(f"Unknown model_name {self.model_name}. Only google/gemma-2-2b and google/gemma-2-9b are supported.")
            # Build the sae_id path: canonical is defined per layer/width
            self.sae_id = f"layer_{self.layer_index}/width_{self.width}/canonical"
        else:
            if self.l0 is None:
                raise ValueError(f"l0 is required for non-canonical SAEs.")
            else:
                # raise not implemented error
                raise NotImplementedError(f"Non-canonical SAEs are not supported for google/gemma-2-9b at the moment.")

@dataclass
class DatasetConfig:
    dataset_name: str
    min_length_doc: int = 0
    max_sequence_length: int = Field(default_factory=int, init=False)
    max_n_docs_per_author: Optional[int] = None
    author_list: Optional[List[str]] = None

    def __post_init__(self):
        if self.dataset_name not in ["AuthorMix", "news", "synthetic"]:
            raise ValueError(f"Dataset {self.dataset_name} is not supported.")
        if self.dataset_name == "news":
            self.max_sequence_length = 512
        elif self.dataset_name == "synthetic":
            self.max_sequence_length = 512
        else:
            self.max_sequence_length = 360