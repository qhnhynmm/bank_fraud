import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
import math

logger = logging.getLogger(__name__)

class CNNClassifier(nn.Module):
    """1D Convolutional Neural Network for binary classification of tabular data."""
    
    def __init__(self, input_dim, num_filters=[64, 128, 64], kernel_sizes=[3, 3, 3], 
                 hidden_dims=[128, 64], dropout_rate=0.3, activation='relu'):
        """
        Initialize the CNN model.
        
        Args:
            input_dim: Number of input features
            num_filters: List of filter numbers for each conv layer, default [64, 128, 64]
            kernel_sizes: List of kernel sizes for each conv layer, default [3, 3, 3]
            hidden_dims: List of hidden layer dimensions for final FC layers, default [128, 64]
            dropout_rate: Dropout probability, default 0.3
            activation: Activation function to use, default 'relu'
        """
        super(CNNClassifier, self).__init__()
        
        self.input_dim = input_dim
        self.num_filters = num_filters
        self.kernel_sizes = kernel_sizes
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
        
        # Input projection layer to add "sequence dimension" for 1D conv
        # We'll reshape input to (batch_size, 1, input_dim) for 1D convolution
        
        # Build convolutional layers
        conv_layers = []
        in_channels = 1  # Start with 1 channel (tabular data as single channel)
        current_length = input_dim
        
        for i, (out_channels, kernel_size) in enumerate(zip(num_filters, kernel_sizes)):
            # Add padding to maintain some spatial dimension
            padding = kernel_size // 2
            
            conv_layers.extend([
                nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding),
                nn.BatchNorm1d(out_channels),
                self.activation,
                nn.Dropout(dropout_rate)
            ])
            
            # Add pooling layer except for the last conv layer
            if i < len(num_filters) - 1:
                conv_layers.append(nn.MaxPool1d(kernel_size=2, stride=2))
                current_length = current_length // 2
            
            in_channels = out_channels
        
        self.conv_layers = nn.Sequential(*conv_layers)
        
        # Global average pooling to get fixed-size representation
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        
        # Build fully connected layers
        fc_layers = []
        prev_dim = num_filters[-1]  # After global pooling, we have num_filters[-1] features
        
        # Add hidden layers
        for hidden_dim in hidden_dims:
            fc_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                self.activation,
                nn.Dropout(dropout_rate)
            ])
            prev_dim = hidden_dim
        
        # Output layer
        fc_layers.append(nn.Linear(prev_dim, 1))
        
        self.fc_layers = nn.Sequential(*fc_layers)
        
        logger.info(f"Created CNN with architecture: {self.get_architecture()}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the model.
        
        Args:
            x: Input tensor of shape (batch_size, input_dim)
            
        Returns:
            Model predictions
        """
        # Reshape for 1D convolution: (batch_size, input_dim) -> (batch_size, 1, input_dim)
        x = x.unsqueeze(1)
        
        # Apply convolutional layers
        x = self.conv_layers(x)
        
        # Global average pooling: (batch_size, channels, length) -> (batch_size, channels, 1)
        x = self.global_avg_pool(x)
        
        # Flatten: (batch_size, channels, 1) -> (batch_size, channels)
        x = x.squeeze(-1)
        
        # Apply fully connected layers
        x = self.fc_layers(x)
        
        return torch.sigmoid(x)
    
    def get_architecture(self) -> str:
        """Returns a string representation of the model architecture."""
        conv_arch = f"Conv1D: {self.num_filters} filters, kernels: {self.kernel_sizes}"
        fc_arch = ' -> '.join([str(self.num_filters[-1])] + [str(dim) for dim in self.hidden_dims] + ['1'])
        return f"{conv_arch} | FC: {fc_arch}"
    
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

def create_cnn_model(input_dim, num_filters=[64, 128, 64], kernel_sizes=[3, 3, 3],
                     hidden_dims=[128, 64], dropout_rate=0.3, activation='relu', 
                     initialization='xavier') -> CNNClassifier:
    """
    Factory function to create and initialize CNN model.
    
    Args:
        input_dim: Number of input features
        num_filters: List of filter numbers for each conv layer, default [64, 128, 64]
        kernel_sizes: List of kernel sizes for each conv layer, default [3, 3, 3]
        hidden_dims: List of hidden layer dimensions for final FC layers, default [128, 64]
        dropout_rate: Dropout probability, default 0.3
        activation: Activation function to use, default 'relu'
        initialization: Weight initialization method, default 'xavier'
        
    Returns:
        Initialized CNN model
    """
    model = CNNClassifier(input_dim, num_filters, kernel_sizes, hidden_dims, dropout_rate, activation)
    
    # Initialize weights
    if initialization == 'xavier':
        for layer in model.modules():
            if isinstance(layer, (nn.Conv1d, nn.Linear)):
                nn.init.xavier_uniform_(layer.weight)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)
    elif initialization == 'kaiming':
        for layer in model.modules():
            if isinstance(layer, (nn.Conv1d, nn.Linear)):
                nn.init.kaiming_normal_(layer.weight)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)
    else:
        logger.warning(f"Unknown initialization method {initialization}, using default")
    
    return model 