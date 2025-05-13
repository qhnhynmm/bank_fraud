"""
Main script for running fraud detection model training and inference.
"""

import os
import argparse
import logging
from typing import Dict
import yaml
import pandas as pd

from src.task.train import ModelTrainer
from src.task.inference import ModelInference
from src.pipelines.data_pipeline import BankingDataPipeline

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_config(config_path: str = "src/config/config.yaml") -> Dict:
    """
    Load configuration from yaml file.
    
    Args:
        config_path: Path to config file
        
    Returns:
        Configuration dictionary
    """
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config

def setup_directories():
    """Create necessary directories if they don't exist."""
    dirs = ['data/raw', 'data/processed', 'checkpoints', 'logs', 'predictions']
    for dir_path in dirs:
        os.makedirs(dir_path, exist_ok=True)
        logger.info(f"Created directory: {dir_path}")

def train_model(config: Dict):
    """
    Train the model using specified configuration.
    
    Args:
        config: Configuration dictionary
    """
    logger.info("Starting model training...")
    
    # Train model
    trainer = ModelTrainer(config)
    model = trainer.train()
    
    logger.info("Model training completed")
    return model

def run_inference(config: Dict, data_path: str = None):
    """
    Run inference using trained model.
    
    Args:
        config: Configuration dictionary
        data_path: Optional path to test data (overrides config)
    """
    logger.info("Starting inference...")
    
    # Initialize inference
    inference = ModelInference(config)
    
    # Use data_path from arguments if provided, otherwise use the one from config
    if data_path is None and 'test_file' in config['inference']:
        data_path = config['inference']['test_file']
        logger.info(f"Using test file from config: {data_path}")
    
    if data_path and os.path.exists(data_path):
        # Load and predict on provided data
        try:
            logger.info(f"Loading test data from: {data_path}")
            data = pd.read_csv(data_path)
            logger.info(f"Loaded data with shape: {data.shape}")
            
            predictions = inference.predict(data, return_proba=True)
            
            # Save predictions
            output_dir = config['inference']['output_dir']
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, 'predictions.csv')
            
            # Add predictions to data
            results = data.copy()
            results['Fraud_Probability'] = predictions
            results['Fraud'] = (predictions >= config['inference']['threshold']).astype(int)
            
            # Save to CSV
            results.to_csv(output_path, index=False)
            
            logger.info(f"Saved predictions to {output_path}")
            
            # Print summary statistics
            fraud_count = results['Fraud'].sum()
            total_count = len(results)
            fraud_rate = fraud_count / total_count * 100
            
            logger.info(f"Fraud Detection Summary:")
            logger.info(f"Total transactions: {total_count}")
            logger.info(f"Fraudulent transactions: {fraud_count} ({fraud_rate:.2f}%)")
            logger.info(f"Average fraud probability: {predictions.mean():.4f}")
            
        except Exception as e:
            logger.error(f"Error during inference: {e}")
            raise
    else:
        if data_path:
            logger.warning(f"Test data file not found: {data_path}")
        
        # Run example inference
        logger.info("No test data provided. Running example inference on a sample transaction.")
        
        # Use the sample transaction method instead of creating our own
        prob = inference.predict_single(return_proba=True)
        prediction = "FRAUD" if prob >= config['inference']['threshold'] else "LEGITIMATE"
        logger.info(f"Fraud probability for sample transaction: {prob:.4f}")
        logger.info(f"Prediction: {prediction}")
    
    logger.info("Inference completed")

def main():
    """Main function to run training and/or inference."""
    parser = argparse.ArgumentParser(description='Run fraud detection model training/inference')
    parser.add_argument('--mode', type=str, choices=['train', 'inference', 'both'], 
                       default='both', help='Mode to run')
    parser.add_argument('--config', type=str, default='src/config/config.yaml',
                       help='Path to config file')
    parser.add_argument('--test-data', type=str, help='Path to test data for inference (overrides config)')
    parser.add_argument('--model', type=str, choices=['mlp', 'random_forest', 'xgboost', 'lightgbm'],
                       help='Model type to use')
    
    args = parser.parse_args()
    
    # Create necessary directories
    setup_directories()
    
    # Load configuration
    config = load_config(args.config)
    
    # Override model type if specified
    if args.model:
        config['model']['type'] = args.model
        logger.info(f"Using model type: {args.model}")
    
    try:
        if args.mode in ['train', 'both']:
            model = train_model(config)
        
        if args.mode in ['inference', 'both']:
            run_inference(config, args.test_data)
            
    except Exception as e:
        logger.error(f"Error occurred: {e}")
        raise

if __name__ == "__main__":
    main() 