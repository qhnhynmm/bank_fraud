import torch
import torch.nn as nn
import torch.nn.functional as F
import logging

logger = logging.getLogger(__name__)

class MLPClassifier(nn.Module):
    """Multi-layer Perceptron for binary classification."""
    
    def __init__(self, input_dim, hidden_dims=[128, 64, 32], dropout_rate=0.3, activation='relu'):
        """
        Initialize the MLP model.
        
        Args:
            input_dim: Number of input features
            hidden_dims: List of hidden layer dimensions, default [128, 64, 32]
            dropout_rate: Dropout probability, default 0.3
            activation: Activation function to use, default 'relu'
        """
        super(MLPClassifier, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.dropout_rate = dropout_rate
        
        # Select activation function
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'sigmoid':
            self.activation = nn.Sigmoid()
        else:
            logger.warning(f"Unknown activation {activation}, using ReLU")
            self.activation = nn.ReLU()
        
        # Build layers
        layers = []
        prev_dim = self.input_dim
        
        # Add hidden layers
        for hidden_dim in self.hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                self.activation,
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

def create_mlp_model(input_dim, hidden_dims=[128, 64, 32], dropout_rate=0.3, 
                     activation='relu', initialization='xavier') -> MLPClassifier:
    """
    Factory function to create and initialize MLP model.
    
    Args:
        input_dim: Number of input features
        hidden_dims: List of hidden layer dimensions, default [128, 64, 32]
        dropout_rate: Dropout probability, default 0.3
        activation: Activation function to use, default 'relu'
        initialization: Weight initialization method, default 'xavier'
        
    Returns:
        Initialized MLP model
    """
    model = MLPClassifier(input_dim, hidden_dims, dropout_rate, activation)
    
    # Initialize weights
    if initialization == 'xavier':
        for layer in model.modules():
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)
    elif initialization == 'kaiming':
        for layer in model.modules():
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_normal_(layer.weight)
                nn.init.zeros_(layer.bias)
    else:
        logger.warning(f"Unknown initialization method {initialization}, using default")
    
    return model 