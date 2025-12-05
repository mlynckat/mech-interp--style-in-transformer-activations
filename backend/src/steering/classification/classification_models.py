from abc import ABC, abstractmethod
from typing import Dict, Any
import pandas as pd
from pathlib import Path
import numpy as np
import torch

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import precision_score, recall_score, f1_score
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments, pipeline


from backend.src.steering.classification.data_transformers import BaseDataTransformer

import pickle


class BaseModel(ABC):
    """Abstract base class for ML models"""
    
    def __init__(self, **kwargs):
        self.params = kwargs
        self.is_fitted = False
    
    @abstractmethod
    def fit(self, X: pd.DataFrame, y: pd.Series) -> 'BaseModel':
        """Train the model"""
        pass
    
    @abstractmethod
    def predict(self, X: pd.DataFrame) -> pd.Series:
        """Make predictions"""
        pass
    
    @abstractmethod
    def get_metrics(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        """Calculate model performance metrics for class 1"""
        pass




class LogisticRegressionModel(BaseModel):
    """Logistic Regression wrapper"""
    
    def __init__(self, path_to_models: Path, author: str, data_transformer: BaseDataTransformer, **kwargs):
        super().__init__(**kwargs)
        self.model = LogisticRegression(**kwargs)
        self.model_path = path_to_models / f"{author}_{data_transformer.__class__.__name__}_logistic_regression_model.pkl"
    
    def fit(self, X: pd.DataFrame, y: pd.Series) -> 'LogisticRegressionModel':
        self.model.fit(X, y)
        self.is_fitted = True
        self.save_model()
        return self
    
    def predict(self, X: pd.DataFrame) -> pd.Series:
        predictions = self.model.predict(X)
        
        return pd.Series(predictions)
    
    def get_metrics(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        y_pred = self.predict(X)
        return {
            'precision': precision_score(y, y_pred, pos_label=1, average='binary', zero_division=0),
            'recall': recall_score(y, y_pred, pos_label=1, average='binary', zero_division=0),
            'f1': f1_score(y, y_pred, pos_label=1, average='binary', zero_division=0)
        }
    def save_model(self):
        with open(self.model_path, 'wb') as fin:
            pickle.dump(self.model, fin)
    
    def load_model(self):
        with open(self.model_path, 'rb') as fin:
            self.model = pickle.load(fin)


class RandomForestModel(BaseModel):
    """Random Forest wrapper"""
    
    def __init__(self, path_to_models: Path, author: str, data_transformer: BaseDataTransformer, **kwargs):
        super().__init__(**kwargs)
        self.model = RandomForestClassifier(**kwargs)
        self.model_path = path_to_models / f"{author}_{data_transformer.__class__.__name__}_random_forest_model.pkl"
    
    def save_model(self):
        with open(self.model_path, 'wb') as fin:
            pickle.dump(self.model, fin)
    
    def load_model(self):
        with open(self.model_path, 'rb') as fin:
            self.model = pickle.load(fin)

    def fit(self, X: pd.DataFrame, y: pd.Series) -> 'RandomForestModel':
        self.model.fit(X, y)
        self.is_fitted = True
        self.save_model()
        return self
    
    def predict(self, X: pd.DataFrame) -> pd.Series:
        predictions = self.model.predict(X)

        return pd.Series(predictions)
    
    def get_metrics(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        y_pred = self.predict(X)
        return {
            'precision': precision_score(y, y_pred, pos_label=1, average='binary', zero_division=0),
            'recall': recall_score(y, y_pred, pos_label=1, average='binary', zero_division=0),
            'f1': f1_score(y, y_pred, pos_label=1, average='binary', zero_division=0),
        }


class SGDClassifierModel(BaseModel):
    """SGD Classifier wrapper"""
    
    def __init__(self, path_to_models: Path, author: str, data_transformer: BaseDataTransformer, **kwargs):
        super().__init__(**kwargs)
        self.model = SGDClassifier(**kwargs)
        self.model_path = path_to_models / f"{author}_{data_transformer.__class__.__name__}_sgd_classifier_model.pkl"
    
    def save_model(self):
        with open(self.model_path, 'wb') as fin:
            pickle.dump(self.model, fin)
    
    def load_model(self):
        with open(self.model_path, 'rb') as fin:
            self.model = pickle.load(fin)
        return self.model

    def fit(self, X: pd.DataFrame, y: pd.Series) -> 'SGDClassifierModel':
        self.model.fit(X, y)
        self.is_fitted = True
        self.save_model()
        return self
    
    def predict(self, X: pd.DataFrame) -> pd.Series:
        predictions = self.model.predict(X)
        return pd.Series(predictions)
    
    def get_metrics(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        y_pred = self.predict(X)
        return {
            'precision': precision_score(y, y_pred, pos_label=1, average='binary', zero_division=0),
            'recall': recall_score(y, y_pred, pos_label=1, average='binary', zero_division=0),
            'f1': f1_score(y, y_pred, pos_label=1, average='binary', zero_division=0)
        }


class ModernBertClassifierModel(BaseModel):
    """ModernBert-based Classifier wrapper"""
    
    def __init__(self, path_to_models: Path, author: str, data_transformer: BaseDataTransformer, **kwargs):
        super().__init__(**kwargs)
        model_id = "answerdotai/ModernBERT-base"
        num_labels = 2
        label2id = {0: "other", 1: author}
        id2label = {0: "other", 1: author}
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_id,
            num_labels=num_labels,
            label2id=label2id,
            id2label=id2label
            )
        self.tokenizer = data_transformer.tokenizer
        self.model_path = path_to_models / f"{author}_{data_transformer.__class__.__name__}_modern_bert_classifier_model"
        
 
    def transform_data_for_trainer(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """Transforms Dataframe and Series to Dicts of Lists like {"labels": [], "input_ids": [], "attention_mask": [], "token_type_ids": []}"""

        # First make sure that indexes are the same
        if X.index.tolist() != y.index.tolist():
            raise ValueError("Indexes of X and y are not the same")
        
        data = []
        for i, row in X.iterrows():
            data.append({
                "labels": y[i],  # Labels should be scalar for classification
                "input_ids": row["input_ids"],  # Already a list from tokenizer
                "attention_mask": row["attention_mask"],  # Already a list from tokenizer
                "token_type_ids": row["token_type_ids"],  # Already a list from tokenizer
            })
        print(data[0])
        return data

    
    def save_model(self):
        self.model.save_pretrained(self.model_path, from_pt=True) 
    
    def load_model(self):
        """Load saved model from disk. Raises error if model doesn't exist."""
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Saved model not found at {self.model_path}. "
                f"Please train the model first using fit() method."
            )
        print(f"Loading trained model from: {self.model_path}")
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_path)
        print(f"Successfully loaded trained model from {self.model_path}")
        

    def fit(self, X: pd.DataFrame, y: pd.Series, X_test: pd.DataFrame = None, y_test: pd.Series = None) -> 'ModernBertClassifierModel':

        training_args = TrainingArguments(
            output_dir= self.model_path,
            per_device_train_batch_size=32,
            per_device_eval_batch_size=16,
            learning_rate=5e-5,
                num_train_epochs=5,
            bf16=True, # bfloat16 training 
            optim="adamw_torch_fused", # improved optimizer 
            # logging & evaluation strategies
            logging_strategy="steps",
            logging_steps=100,
            eval_strategy="epoch",
            save_strategy="epoch", 
            save_total_limit=1,  # Don't save any checkpoints during training
            load_best_model_at_end=True,  # Load best model at the end
            metric_for_best_model="f1",
            push_to_hub=False
        )
        
        X_train = self.transform_data_for_trainer(X, y)
        X_val = self.transform_data_for_trainer(X_test, y_test)
        
        # Create a Trainer instance
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=X_train,
            eval_dataset=X_val,
            compute_metrics=self.compute_metrics,
        )
        trainer.train()
        self.is_fitted = True
        self.save_model()
        return self
    
    def predict(self, X: pd.DataFrame) -> pd.Series:
        
        # Convert DataFrame columns to tensors
        input_ids = torch.tensor(X['input_ids'].tolist())
        attention_mask = torch.tensor(X['attention_mask'].tolist())
        
        # Move to device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device)
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        
        # Get predictions
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        
        # Get predicted class IDs
        predicted_ids = outputs.logits.argmax(-1).cpu().numpy()
        
        return pd.Series(predicted_ids, index=X.index)
    
    def compute_metrics(self, eval_pred) -> Dict[str, float]:
        """Compute metrics for Trainer. eval_pred is a tuple of (predictions, labels)."""
        predictions, labels = eval_pred
        # predictions are logits, need to get the predicted class
        predictions = np.argmax(predictions, axis=1)
        return {
            'precision': precision_score(labels, predictions, pos_label=1, average='binary', zero_division=0),
            'recall': recall_score(labels, predictions, pos_label=1, average='binary', zero_division=0),
            'f1': f1_score(labels, predictions, pos_label=1, average='binary', zero_division=0)
        }
    
    def get_metrics(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        y_pred = self.predict(X)
        return {
            'precision': precision_score(y, y_pred, pos_label=1, average='binary', zero_division=0),
            'recall': recall_score(y, y_pred, pos_label=1, average='binary', zero_division=0),
            'f1': f1_score(y, y_pred, pos_label=1, average='binary', zero_division=0)
        }

