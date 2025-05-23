"""
Test script for LSTM model implementation
"""

import torch
import numpy as np
from src.model.lstm_model import create_lstm_model

def test_lstm_model():
    """Test LSTM model creation and forward pass."""
    
    # Test parameters
    input_dim = 52  # Typical feature dimension after preprocessing
    batch_size = 32
    
    print("Testing LSTM Model for Fraud Detection")
    print("="*50)
    
    # Test different configurations
    configs = [
        {
            'name': 'Bidirectional LSTM',
            'lstm_hidden_dims': [128, 64],
            'num_layers': [2, 2],
            'hidden_dims': [128, 64],
            'bidirectional': True
        },
        {
            'name': 'Unidirectional LSTM',
            'lstm_hidden_dims': [64, 32],
            'num_layers': [1, 1],
            'hidden_dims': [64, 32],
            'bidirectional': False
        }
    ]
    
    for config in configs:
        print(f"\n{config['name']} Configuration:")
        print("-" * 40)
        
        # Create model
        model = create_lstm_model(
            input_dim=input_dim,
            lstm_hidden_dims=config['lstm_hidden_dims'],
            num_layers=config['num_layers'],
            hidden_dims=config['hidden_dims'],
            dropout_rate=0.3,
            activation='relu',
            bidirectional=config['bidirectional'],
            initialization='xavier'
        )
        
        print(f"Model architecture: {model.get_architecture()}")
        print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
        
        # Test forward pass
        dummy_input = torch.randn(batch_size, input_dim)
        print(f"Testing forward pass with input shape: {dummy_input.shape}")
        
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
        print(f"Training mode output shape: {train_output.shape}")
        print(f"Training mode output range: [{train_output.min().item():.4f}, {train_output.max().item():.4f}]")
        
        # Test different input sizes
        for test_dim in [20, 100]:
            if test_dim != input_dim:
                print(f"\nTesting with input dimension: {test_dim}")
                test_model = create_lstm_model(input_dim=test_dim)
                test_input = torch.randn(8, test_dim)
                test_output = test_model(test_input)
                print(f"Output shape: {test_output.shape}")
        
        print(f"\n✅ {config['name']} test completed successfully!")
    
    # Test gradient flow
    print("\n" + "="*50)
    print("Testing Gradient Flow")
    print("="*50)
    
    model = create_lstm_model(input_dim=input_dim)
    dummy_input = torch.randn(batch_size, input_dim, requires_grad=True)
    target = torch.randint(0, 2, (batch_size, 1)).float()
    
    # Forward pass
    output = model(dummy_input)
    loss = torch.nn.BCELoss()(output, target)
    
    # Backward pass
    loss.backward()
    
    # Check gradients
    grad_norms = []
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_norm = param.grad.norm().item()
            grad_norms.append(grad_norm)
            if 'lstm' in name:
                print(f"{name}: grad norm = {grad_norm:.6f}")
    
    print(f"Average gradient norm: {np.mean(grad_norms):.6f}")
    print(f"Max gradient norm: {np.max(grad_norms):.6f}")
    
    if np.max(grad_norms) > 10.0:
        print("⚠️  Warning: Large gradients detected, consider gradient clipping")
    else:
        print("✅ Gradient flow looks good")
    
    print("\n✅ All LSTM Model tests completed successfully!")
    
    return model

if __name__ == "__main__":
    test_lstm_model() 