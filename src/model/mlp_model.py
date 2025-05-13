"""
Multi-layer Perceptron model for fraud detection.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import logging

logger = logging.getLogger(__name__)

class MLPClassifier(nn.Module):
    """Multi-layer Perceptron for binary classification."""
    
    def __init__(self, config: dict):
        """
        Initialize the MLP model.
        
        Args:
            config: Model configuration dictionary containing:
                - input_dim: Number of input features
                - hidden_dims: List of hidden layer dimensions
                - dropout_rate: Dropout probability
        """
        super(MLPClassifier, self).__init__()
        
        self.input_dim = config['input_dim']
        self.hidden_dims = config['hidden_dims']
        self.dropout_rate = config.get('dropout_rate', 0.2)
        
        # Build layers
        layers = []
        prev_dim = self.input_dim
        
        # Add hidden layers
        for hidden_dim in self.hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(self.dropout_rate)
            ])
            prev_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(prev_dim, 1))
        
        self.model = nn.Sequential(*layers)
        logger.info(f"Created MLP with architecture: {self.get_architecture()}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the model.
        
        Args:
            x: Input tensor of shape (batch_size, input_dim)
            
        Returns:
            Model predictions
        """
        return torch.sigmoid(self.model(x))
    
    def get_architecture(self) -> str:
        """Returns a string representation of the model architecture."""
        dims = [self.input_dim] + self.hidden_dims + [1]
        return ' -> '.join(map(str, dims))
    
    def predict(self, x: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
        """
        Make binary predictions.
        
        Args:
            x: Input tensor
            threshold: Classification threshold
            
        Returns:
            Binary predictions
        """
        with torch.no_grad():
            probs = self(x)
            return (probs >= threshold).float()
    
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """
        Predict probabilities.
        
        Args:
            x: Input tensor
            
        Returns:
            Probability predictions
        """
        with torch.no_grad():
            return self(x)

def create_mlp_model(config: dict) -> MLPClassifier:
    """
    Factory function to create and initialize MLP model.
    
    Args:
        config: Model configuration
        
    Returns:
        Initialized MLP model
    """
    model = MLPClassifier(config)
    
    # Initialize weights
    for layer in model.modules():
        if isinstance(layer, nn.Linear):
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)
    
    return model 