import pandas as pd
from typing import Dict, Any
from datetime import datetime
import json
from pathlib import Path

from backend.src.steering.classification.data_transformers import BaseDataTransformer
from backend.src.steering.classification.classification_models import BaseModel

class MLPipeline:
    """Main pipeline orchestrator"""
    
    def __init__(self, data_transformer: BaseDataTransformer, classification_model: BaseModel):
        """
        Initialize pipeline with transformer and model
        
        Args:
            data_transformer: Instance of BaseDataTransformer
            classification_model: Instance of BaseModel
        """
        self.data_transformer = data_transformer
        self.classification_model = classification_model
        self.results = {}
    
    def run(self, X_train: pd.DataFrame, y_train: pd.Series, 
            X_test: pd.DataFrame = None, y_test: pd.Series = None) -> Dict[str, Any]:
        """
        Execute full pipeline
        
        Args:
            X_train: Training features
            y_train: Training target
            X_test: Test features (optional)
            y_test: Test target (optional)
        
        Returns:
            Dictionary with results
        """
        print(f"Starting pipeline for {self.data_transformer.__class__.__name__} and {self.classification_model.__class__.__name__}")
        
        # Step 1: Transform data
        print("Transforming data...")
        X_train_transformed = self.data_transformer.fit(X_train).transform(X_train)
        X_test_transformed = self.data_transformer.transform(X_test)
        
        # Step 2: Train model
        print("Training model...")
        self.classification_model.fit(X_train_transformed, y_train, X_test_transformed, y_test)
        
        # Step 3: Evaluate
        print("Evaluating...")
        train_metrics = self.classification_model.get_metrics(X_train_transformed, y_train)

        try:
            classification_model_params = self.classification_model.get_params()
        except:
            classification_model_params = None

        results = {
            'timestamp': datetime.now().isoformat(),
            'data_transformer': self.data_transformer.__class__.__name__,
            'data_transformer_params': self.data_transformer.params,
            'classification_model': self.classification_model.__class__.__name__,
            'classification_model_params': classification_model_params,
            'train_metrics': train_metrics
        }
        
        
        test_metrics = self.classification_model.get_metrics(X_test_transformed, y_test)
        results['test_metrics'] = test_metrics

        predicted_labels = self.classification_model.predict(X_test_transformed)
        results['predicted_labels'] = predicted_labels
        
        self.results = results
        return results

    
    def predict_samples(self, X: pd.Series) -> pd.Series:
        """
        Predict samples
        
        Args:
            X: Samples to predict
        
        Returns:
            Predictions
        """
        # Step 1: Transform data
        print("Transforming data...")

        if self.data_transformer.__class__.__name__ == "TFIDFTransformer":
            self.data_transformer.load_model()

        self.classification_model.load_model()
        
        X_transformed = self.data_transformer.transform(X)

        # Step 2: Predict
        print("Predicting...")
        predictions = self.classification_model.predict(X_transformed)

        return predictions
    
    def predict_proba_samples(self, X: pd.Series) -> pd.Series:
        """
        Predict probabilities for samples
        
        Args:
            X: Samples to predict
        
        Returns:
            Probabilities of class 1
        """
        # Step 1: Transform data
        print("Transforming data...")

        if self.data_transformer.__class__.__name__ == "TFIDFTransformer":
            self.data_transformer.load_model()

        self.classification_model.load_model()
        
        X_transformed = self.data_transformer.transform(X)

        # Step 2: Predict probabilities
        print("Predicting probabilities...")
        probas = self.classification_model.predict_proba(X_transformed)

        return probas