import pandas as pd
from abc import ABC, abstractmethod
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, BatchEncoding
import torch

import pickle


class BaseDataTransformer(ABC):
    """Abstract base class for data transformers"""
    
    def __init__(self, **kwargs):
        self.params = kwargs
    
    @abstractmethod
    def fit(self, data: pd.DataFrame) -> 'BaseDataTransformer':
        """Fit transformer to data"""
        pass
    
    @abstractmethod
    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """Transform data"""
        pass


class TFIDFTransformer(BaseDataTransformer):
    """Standardize features using StandardScaler"""
    
    def __init__(self, path_to_models: Path, author: str, **kwargs):
        super().__init__(**kwargs)
        self.tfidf_transformer = TfidfVectorizer(**kwargs)
        self.model_path = path_to_models / f"{author}_vectorizer.pk"
    
    def fit(self, data: pd.DataFrame) -> 'TFIDFTransformer':
        self.tfidf_transformer.fit(data)
        self.save_model()
        return self
    
    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        transformed = self.tfidf_transformer.transform(data)
        return transformed
    
    def save_model(self):
        with open(self.model_path, 'wb') as fin:
            pickle.dump(self.tfidf_transformer, fin)

    def load_model(self):
        try:
            with open(self.model_path, 'rb') as fin:
                self.tfidf_transformer = pickle.load(fin)
        except Exception as e:
            print(f"Error loading model: {e}")
            raise Exception(f"Error loading model: {e}")


class SentenceEmbeddingTransformer(BaseDataTransformer):
    """Transform text data into sentence embeddings with hugging face models"""
    
    def __init__(self, path_to_models: Path, author: str, **kwargs):
        super().__init__(**kwargs)
        self.sentence_embeddings_model = SentenceTransformer(kwargs.get("model_name", "all-MiniLM-L6-v2"))
    
    def fit(self, data: pd.DataFrame) -> 'SentenceEmbeddingTransformer':
        return self
    
    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        if isinstance(data, pd.Series):
            sentences = data.tolist()
        elif isinstance(data, pd.DataFrame):
            # assume single text column
            sentences = data.iloc[:, 0].tolist()
        else:
            sentences = list(data)

        embeddings = self.sentence_embeddings_model.encode(sentences)
        return embeddings

class ModernBertTransformer(BaseDataTransformer):
    """Transform text data into sentence embeddings with hugging face models"""
    
    def __init__(self, path_to_models: Path, author: str, **kwargs):
        super().__init__(**kwargs)
        self.tokenizer = AutoTokenizer.from_pretrained("answerdotai/ModernBERT-base")
        self.tokenizer.model_max_length = 512

    def fit(self, data: pd.Series) -> 'ModernBertTransformer':
        return self
    
    def transform(self, data: pd.Series) -> pd.DataFrame:
        # assume single text column

        if isinstance(data, pd.Series):
            sentences = data.tolist()
        elif isinstance(data, pd.DataFrame):
            # assume single text column
            sentences = data.iloc[:, 0].tolist()
        else:
            sentences = list(data)
        
            
        encoded_inputs = self.tokenizer(
            sentences,
            return_tensors="pt",
            add_special_tokens=True,
            max_length=512,
            truncation=True,
            padding="max_length"
        )
        # Handle token_type_ids - some tokenizers don't return it by default
        if "token_type_ids" in encoded_inputs:
            token_type_ids = encoded_inputs.token_type_ids.tolist()
        else:
            # Create zeros tensor with same shape as input_ids (for single-sequence tasks)
            token_type_ids = torch.zeros_like(encoded_inputs.input_ids).tolist()
        
        return pd.DataFrame( 
            list(zip(encoded_inputs.input_ids.tolist(), encoded_inputs.attention_mask.tolist(), token_type_ids)), 
            columns=["input_ids", "attention_mask", "token_type_ids"], 
            index=data.index
        )