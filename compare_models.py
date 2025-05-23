"""
Script to compare MLP and CNN models for fraud detection
"""

import os
import yaml
import logging
from src.task.train import ModelTrainer

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_config(config_path='src/config/config.yaml'):
    """Load configuration from yaml file."""
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def train_and_compare():
    """Train both MLP and CNN models and compare their performance."""
    
    print("Comparing MLP vs CNN for Fraud Detection")
    print("="*60)
    
    # Load base config
    config = load_config()
    
    models_to_test = ['mlp', 'cnn']
    results = {}
    
    for model_type in models_to_test:
        print(f"\n🔥 Training {model_type.upper()} Model")
        print("-" * 40)
        
        # Update config for current model
        config['model']['type'] = model_type
        
        # Create trainer
        trainer = ModelTrainer(config)
        
        try:
            # Train model
            model = trainer.train()
            
            # Store results
            results[model_type] = {
                'model': model,
                'trainer': trainer,
                'success': True
            }
            
            print(f"✅ {model_type.upper()} training completed successfully!")
            
        except Exception as e:
            logger.error(f"❌ Error training {model_type} model: {e}")
            results[model_type] = {
                'model': None,
                'trainer': trainer,
                'success': False,
                'error': str(e)
            }
    
    # Print comparison summary
    print("\n" + "="*60)
    print("TRAINING SUMMARY")
    print("="*60)
    
    for model_type, result in results.items():
        print(f"\n{model_type.upper()} Model:")
        if result['success']:
            print("  ✅ Training: SUCCESS")
            model_path = f"checkpoints/best_model_{model_type}.pt"
            if os.path.exists(model_path):
                print(f"  📁 Model saved: {model_path}")
            else:
                print("  ⚠️  Model file not found")
        else:
            print("  ❌ Training: FAILED")
            print(f"  🔴 Error: {result.get('error', 'Unknown error')}")
    
    return results

if __name__ == "__main__":
    results = train_and_compare()
    
    # Additional information
    print("\n" + "="*60)
    print("NEXT STEPS")
    print("="*60)
    print("1. Check the logs for detailed training metrics")
    print("2. Run inference tests with: python main.py --inference")
    print("3. Compare model files in the 'checkpoints/' directory")
    print("4. Update config.yaml to use your preferred model type")
    
    # Suggest which model to use based on training success
    successful_models = [name for name, result in results.items() if result['success']]
    if successful_models:
        print(f"\n🎯 Successfully trained models: {', '.join(successful_models).upper()}")
        print("📋 You can now use either model by updating the 'type' field in config.yaml")
    else:
        print("\n⚠️  No models trained successfully. Please check the logs for errors.") 