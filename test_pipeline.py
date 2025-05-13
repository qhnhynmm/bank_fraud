"""
Test script to validate the entire fraud detection pipeline.
"""

import logging
import os
import pandas as pd
import numpy as np
from src.pipelines.data_pipeline import BankingDataPipeline
from src.task.train import ModelTrainer
from src.task.inference import ModelInference

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_data_pipeline():
    """Test the data pipeline."""
    logger.info("\n=== Testing Data Pipeline ===")
    
    # Initialize data pipeline
    pipeline = BankingDataPipeline()
    
    # Run pipeline
    try:
        X_train, X_val, X_test, y_train, y_val, y_test = pipeline.run_pipeline()
        logger.info("✅ Data pipeline ran successfully!")
        
        # Validate outputs
        logger.info(f"Training data shape: {X_train.shape}")
        logger.info(f"Validation data shape: {X_val.shape}")
        logger.info(f"Test data shape: {X_test.shape}")
        
        # Verify fraud distribution
        train_fraud_ratio = np.mean(y_train)
        val_fraud_ratio = np.mean(y_val)
        test_fraud_ratio = np.mean(y_test)
        
        logger.info(f"Training fraud ratio: {train_fraud_ratio:.4f}")
        logger.info(f"Validation fraud ratio: {val_fraud_ratio:.4f}")
        logger.info(f"Test fraud ratio: {test_fraud_ratio:.4f}")
        
        return X_train, X_val, X_test, y_train, y_val, y_test
        
    except Exception as e:
        logger.error(f"❌ Data pipeline failed: {e}")
        raise

def test_model_training(X_train, X_val, X_test, y_train, y_val, y_test):
    """Test model training."""
    logger.info("\n=== Testing Model Training ===")
    
    # Load config
    pipeline = BankingDataPipeline()
    config = pipeline.config
    
    # Set model type to test
    model_types = ["mlp", "random_forest", "xgboost", "lightgbm"]
    test_model = model_types[0]  # Change index to test different models
    
    logger.info(f"Testing model type: {test_model}")
    config['model']['type'] = test_model
    
    # Initialize trainer
    trainer = ModelTrainer(config)
    
    # Override run_pipeline method to use our data
    def mock_run_pipeline():
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    trainer.pipeline = pipeline
    trainer.pipeline.run_pipeline = mock_run_pipeline
    
    # Train model
    try:
        model = trainer.train()
        logger.info("✅ Model training successful!")
        return model, config
        
    except Exception as e:
        logger.error(f"❌ Model training failed: {e}")
        raise

def test_inference(config, test_data=None):
    """Test model inference."""
    logger.info("\n=== Testing Model Inference ===")
    
    # Initialize inference engine
    try:
        inference = ModelInference(config)
        logger.info("✅ Inference engine initialized successfully!")
        
        # Create sample transaction if test data not provided
        if test_data is None:
            sample_transaction = {
                'Amount': 1000.0,
                'Type': 'PAYMENT',
                'Day of Week': 3,
                'Hour': 12,
                'Is_Weekend': 0
            }
            
            # Test single transaction prediction
            logger.info("Testing prediction for a single transaction:")
            prob = inference.predict_single(sample_transaction, return_proba=True)
            prediction = 1 if prob >= config['inference']['threshold'] else 0
            
            logger.info(f"Fraud probability: {prob:.4f}")
            logger.info(f"Prediction: {'Fraudulent' if prediction else 'Legitimate'}")
            
        else:
            # Test batch prediction
            logger.info(f"Testing batch prediction with {len(test_data)} samples")
            probabilities = inference.predict(test_data, return_proba=True)
            predictions = inference.predict(test_data, return_proba=False)
            
            # Summarize results
            fraud_count = np.sum(predictions)
            fraud_rate = fraud_count / len(predictions)
            
            logger.info(f"Detected {fraud_count} fraudulent transactions ({fraud_rate:.2%})")
            logger.info(f"Average fraud probability: {np.mean(probabilities):.4f}")
            logger.info(f"Max fraud probability: {np.max(probabilities):.4f}")
        
        logger.info("✅ Inference testing successful!")
        
    except Exception as e:
        logger.error(f"❌ Inference testing failed: {e}")
        raise

def run_complete_test():
    """Run a complete test of the fraud detection system."""
    try:
        # Test data pipeline
        X_train, X_val, X_test, y_train, y_val, y_test = test_data_pipeline()
        
        # Test model training
        model, config = test_model_training(X_train, X_val, X_test, y_train, y_val, y_test)
        
        # Test inference
        # Get a small sample of test data for batch testing
        if isinstance(X_test, np.ndarray):
            # Create sample dataframe
            feature_count = X_test.shape[1]
            sample_data = pd.DataFrame(
                X_test[:5], 
                columns=[f'feature_{i}' for i in range(feature_count)]
            )
        else:
            sample_data = X_test[:5]
            
        test_inference(config, sample_data)
        
        # Test single transaction inference
        test_inference(config)
        
        logger.info("\n✅✅✅ Complete test finished successfully! The system is working correctly.")
        
    except Exception as e:
        logger.error(f"\n❌❌❌ Test failed: {e}")
        raise

if __name__ == "__main__":
    run_complete_test() 