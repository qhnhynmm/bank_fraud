"""
Inference script for fraud detection model.
"""

import os
import torch
import numpy as np
import pandas as pd
from typing import Dict, Union, List
import logging
import joblib
import scipy.sparse

from src.model.mlp_model import create_mlp_model
from src.pipelines.data_pipeline import BankingDataPipeline
from src.metrics.metrics import FraudMetrics

logger = logging.getLogger(__name__)

class ModelInference:
    """Class to handle model inference."""
    
    def __init__(self, config: Dict):
        """
        Initialize inference with configuration.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.pipeline = BankingDataPipeline()
        
        # Load the model
        self._load_model()
    
    def _load_model(self):
        """Load the trained model based on type."""
        model_type = self.config['model']['type']
        logger.info(f"Loading {model_type} model")
        
        if model_type == 'mlp':
            # Load PyTorch model
            model_path = os.path.join('checkpoints', f'best_model_{model_type}.pt')
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model file not found at {model_path}")
            
            # Create model with same configuration - FIX: Pass parameters separately
            input_dim = self.pipeline.get_feature_dim()
            hidden_dims = self.config['model']['mlp']['hidden_dims'] 
            dropout_rate = self.config['model']['mlp']['dropout_rate']
            
            # Create model with properly unpacked parameters
            self.model = create_mlp_model(
                input_dim=input_dim, 
                hidden_dims=hidden_dims, 
                dropout_rate=dropout_rate
            )
            
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            self.model.to(self.device)
            self.model.eval()
        else:
            # Load traditional ML model
            model_path = os.path.join('checkpoints', f'best_model_{model_type}.joblib')
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model file not found at {model_path}")
                
            self.model = joblib.load(model_path)
        
        logger.info(f"Successfully loaded {model_type} model")
    
    def predict(self, data: Union[pd.DataFrame, np.ndarray], 
               return_proba: bool = False) -> Union[np.ndarray, List[float]]:
        """
        Make predictions on new data.
        
        Args:
            data: Input data
            return_proba: Whether to return probability scores
            
        Returns:
            Predictions (binary or probability scores)
        """
        # Preprocess data using pipeline
        processed_data = self.pipeline.preprocess_inference_data(data)
        logger.info(f"Preprocessed data shape: {processed_data.shape}")
        
        if self.config['model']['type'] == 'mlp':
            # Convert to tensor for PyTorch model
            # Check if processed_data is a sparse matrix and convert to dense if needed
            if scipy.sparse.issparse(processed_data):
                logger.info("Converting sparse matrix to dense for PyTorch model")
                processed_data = processed_data.toarray()
            
            processed_data = torch.FloatTensor(processed_data).to(self.device)
            
            with torch.no_grad():
                predictions = self.model(processed_data)
                
            if return_proba:
                return predictions.cpu().numpy().flatten()
            else:
                threshold = self.config['inference']['threshold']
                return (predictions.cpu() >= threshold).numpy().flatten().astype(int)
        else:
            # Use sklearn/traditional ML model
            if return_proba:
                if hasattr(self.model, 'predict_proba'):
                    return self.model.predict_proba(processed_data)[:, 1]
                else:
                    # If model doesn't have predict_proba, use decision function if available
                    if hasattr(self.model, 'decision_function'):
                        scores = self.model.decision_function(processed_data)
                        # Normalize to 0-1 range if needed
                        return 1 / (1 + np.exp(-scores))
                    else:
                        # Fall back to binary predictions
                        logger.warning("Model doesn't support probability predictions")
                        return self.model.predict(processed_data).astype(float)
            else:
                return self.model.predict(processed_data)
    
    def create_sample_transaction(self):
        """
        Create a sample transaction with the correct structure based on the dataset.
        
        Returns:
            DataFrame: Sample transaction
        """
        # Create a sample transaction with all required fields based on our dataset
        sample = {
            'Transaction ID': '#9999 999',
            'Date': '14-Oct-20',
            'Day of Week': 'Wednesday',
            'Time': 12,
            'Type of Card': 'Visa',
            'Entry Mode': 'PIN',
            'Amount': 250,
            'Type of Transaction': 'POS',
            'Merchant Group': 'Electronics',
            'Country of Transaction': 'United Kingdom',
            'Shipping Address': 'United Kingdom',
            'Country of Residence': 'United Kingdom',
            'Gender': 'M',
            'Age': 35.0,
            'Bank': 'Barclays'
        }
        
        return pd.DataFrame([sample])
    
    def predict_single(self, transaction: Dict = None, return_proba: bool = False) -> float:
        """
        Make prediction on a single transaction.
        
        Args:
            transaction: Dictionary containing transaction details (optional)
            return_proba: Whether to return probability score
            
        Returns:
            Prediction (binary or probability score)
        """
        # If no transaction provided, create a sample one
        if transaction is None:
            logger.info("No transaction provided. Using sample transaction.")
            df = self.create_sample_transaction()
        else:
            # If transaction is provided, try to create a valid transaction dataframe
            try:
                # Start with a sample to ensure all fields are present
                sample = self.create_sample_transaction().iloc[0].to_dict()
                
                # Update with provided values
                for key, value in transaction.items():
                    if key in sample:
                        sample[key] = value
                    else:
                        logger.warning(f"Ignoring unknown field: {key}")
                
                df = pd.DataFrame([sample])
                
            except Exception as e:
                logger.error(f"Error creating transaction: {e}")
                logger.info("Using sample transaction instead.")
                df = self.create_sample_transaction()
        
        logger.info(f"Transaction for prediction: {df.iloc[0].to_dict()}")
        
        # Make prediction
        prediction = self.predict(df, return_proba=return_proba)
        
        # Log results
        threshold = self.config['inference']['threshold']
        Fraud = prediction[0] >= threshold if return_proba else bool(prediction[0])
        label = "FRAUD" if Fraud else "LEGITIMATE"
        
        if return_proba:
            logger.info(f"Transaction fraud probability: {prediction[0]:.4f} (Threshold: {threshold:.2f})")
        logger.info(f"Transaction classified as: {label}")
        
        return prediction[0]

    def evaluate_predictions(self, y_true: np.ndarray, predictions: np.ndarray):
        """
        Evaluate predictions using our metrics.
        
        Args:
            y_true: True labels
            predictions: Model predictions
        """
        metrics = FraudMetrics.calculate_metrics(y_true, predictions)
        logger.info("Evaluation metrics:")
        FraudMetrics.print_metrics(metrics)
        return metrics

def main():
    """Example usage of inference."""
    # Load config
    pipeline = BankingDataPipeline()
    config = pipeline.config
    
    # Initialize inference
    inference = ModelInference(config)
    
    # Example: Make predictions on test data
    data_path = config['inference'].get('test_file', 'data/test.csv')
    if os.path.exists(data_path):
        # Load test data
        data = pd.read_csv(data_path)
        logger.info(f"Loaded test data from {data_path} with shape {data.shape}")
        
        # Get predictions
        predictions = inference.predict(data)
        probabilities = inference.predict(data, return_proba=True)
        
        # Evaluate if 'Fraud' column exists
        if 'Fraud' in data.columns:
            metrics = inference.evaluate_predictions(data['Fraud'].values, predictions)
        
        # Save predictions
        results = data.copy()
        results['Fraud_Probability'] = probabilities
        results['Predicted_Fraud'] = predictions
        
        # Save to CSV
        os.makedirs('predictions', exist_ok=True)
        output_path = os.path.join('predictions', 'test_predictions.csv')
        results.to_csv(output_path, index=False)
        logger.info(f"Saved predictions to {output_path}")
    else:
        logger.warning(f"Test file not found: {data_path}")
        
        # Example: Single transaction prediction
        logger.info("Running prediction on sample transaction instead")
        single_prediction = inference.predict_single(return_proba=True)
        logger.info(f"Sample transaction fraud probability: {single_prediction:.4f}")

if __name__ == "__main__":
    main()
