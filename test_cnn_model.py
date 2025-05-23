"""
Test script for CNN model implementation
"""

import torch
import numpy as np
from src.model.cnn_model import create_cnn_model

def test_cnn_model():
    """Test CNN model creation and forward pass."""
    
    # Test parameters
    input_dim = 52  # Typical feature dimension after preprocessing
    batch_size = 32
    
    print("Testing CNN Model for Fraud Detection")
    print("="*50)
    
    # Create model
    model = create_cnn_model(
        input_dim=input_dim,
        num_filters=[64, 128, 64],
        kernel_sizes=[3, 3, 3],
        hidden_dims=[128, 64],
        dropout_rate=0.3,
        activation='relu',
        initialization='xavier'
    )
    
    print(f"Model architecture: {model.get_architecture()}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Test forward pass
    dummy_input = torch.randn(batch_size, input_dim)
    print(f"\nTesting forward pass with input shape: {dummy_input.shape}")
    
    # Forward pass
    model.eval()
    with torch.no_grad():
        output = model(dummy_input)
        predictions = model.predict(dummy_input)
        probabilities = model.predict_proba(dummy_input)
    
    print(f"Output shape: {output.shape}")
    print(f"Output range: [{output.min().item():.4f}, {output.max().item():.4f}]")
    print(f"Predictions shape: {predictions.shape}")
    print(f"Predictions unique values: {torch.unique(predictions).tolist()}")
    print(f"Probabilities shape: {probabilities.shape}")
    print(f"Sample probabilities: {probabilities[:5].flatten().tolist()}")
    
    # Test training mode
    model.train()
    train_output = model(dummy_input)
    print(f"\nTraining mode output shape: {train_output.shape}")
    print(f"Training mode output range: [{train_output.min().item():.4f}, {train_output.max().item():.4f}]")
    
    print("\n✅ CNN Model test completed successfully!")
    
    return model

if __name__ == "__main__":
    test_cnn_model() 