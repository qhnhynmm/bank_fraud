import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
import math

logger = logging.getLogger(__name__)

class LSTMClassifier(nn.Module):
    """LSTM Neural Network for binary classification of tabular data."""
    
    def __init__(self, input_dim, lstm_hidden_dims=[128, 64], num_layers=[2, 2], 
                 hidden_dims=[128, 64], dropout_rate=0.3, activation='relu', bidirectional=True):
        """
        Initialize the LSTM model.
        
        Args:
            input_dim: Number of input features
            lstm_hidden_dims: List of hidden dimensions for each LSTM layer, default [128, 64]
            num_layers: List of number of layers for each LSTM, default [2, 2]
            hidden_dims: List of hidden layer dimensions for final FC layers, default [128, 64]
            dropout_rate: Dropout probability, default 0.3
            activation: Activation function to use, default 'relu'
            bidirectional: Whether to use bidirectional LSTM, default True
        """
        super(LSTMClassifier, self).__init__()
        
        self.input_dim = input_dim
        self.lstm_hidden_dims = lstm_hidden_dims
        self.num_layers = num_layers
        self.hidden_dims = hidden_dims
        self.dropout_rate = dropout_rate
        self.bidirectional = bidirectional
        
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
        
        # Input projection to create sequence-like data
        # We'll create a small sequence from tabular data by grouping features
        self.seq_length = min(16, input_dim // 4)  # Create sequence length based on input features
        self.feature_per_step = input_dim // self.seq_length
        if input_dim % self.seq_length != 0:
            # Add padding if needed
            self.padding_size = self.seq_length - (input_dim % self.seq_length)
        else:
            self.padding_size = 0
            
        actual_input_dim = input_dim + self.padding_size
        self.feature_per_step = actual_input_dim // self.seq_length
        
        # Build LSTM layers
        lstm_layers = []
        prev_dim = self.feature_per_step
        
        for i, (hidden_dim, n_layers) in enumerate(zip(lstm_hidden_dims, num_layers)):
            lstm = nn.LSTM(
                input_size=prev_dim,
                hidden_size=hidden_dim,
                num_layers=n_layers,
                batch_first=True,
                dropout=dropout_rate if n_layers > 1 else 0,
                bidirectional=bidirectional
            )
            lstm_layers.append(lstm)
            
            # Add dropout between LSTM layers
            if i < len(lstm_hidden_dims) - 1:
                lstm_layers.append(nn.Dropout(dropout_rate))
            
            # Update prev_dim for next LSTM layer
            prev_dim = hidden_dim * (2 if bidirectional else 1)
        
        self.lstm_layers = nn.ModuleList(lstm_layers)
        
        # Global pooling to get fixed-size representation
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        self.global_max_pool = nn.AdaptiveMaxPool1d(1)
        
        # Build fully connected layers
        fc_layers = []
        # After pooling, we have 2 * prev_dim features (avg + max pooling)
        fc_input_dim = 2 * prev_dim
        
        # Add hidden layers
        for hidden_dim in hidden_dims:
            fc_layers.extend([
                nn.Linear(fc_input_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                self.activation,
                nn.Dropout(dropout_rate)
            ])
            fc_input_dim = hidden_dim
        
        # Output layer
        fc_layers.append(nn.Linear(fc_input_dim, 1))
        
        self.fc_layers = nn.Sequential(*fc_layers)
        
        logger.info(f"Created LSTM with architecture: {self.get_architecture()}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the model.
        
        Args:
            x: Input tensor of shape (batch_size, input_dim)
            
        Returns:
            Model predictions
        """
        batch_size = x.size(0)
        
        # Add padding if needed
        if self.padding_size > 0:
            padding = torch.zeros(batch_size, self.padding_size, device=x.device)
            x = torch.cat([x, padding], dim=1)
        
        # Reshape to sequence: (batch_size, seq_length, feature_per_step)
        x = x.view(batch_size, self.seq_length, self.feature_per_step)
        
        # Apply LSTM layers
        for i, layer in enumerate(self.lstm_layers):
            if isinstance(layer, nn.LSTM):
                x, _ = layer(x)
            else:  # Dropout layer
                x = layer(x)
        
        # x shape: (batch_size, seq_length, lstm_hidden_dim * (2 if bidirectional else 1))
        
        # Global pooling: (batch_size, seq_length, features) -> (batch_size, features)
        # Transpose for pooling: (batch_size, features, seq_length)
        x_transposed = x.transpose(1, 2)
        
        # Apply both average and max pooling
        avg_pooled = self.global_avg_pool(x_transposed).squeeze(-1)  # (batch_size, features)
        max_pooled = self.global_max_pool(x_transposed).squeeze(-1)  # (batch_size, features)
        
        # Concatenate pooled features
        x = torch.cat([avg_pooled, max_pooled], dim=1)
        
        # Apply fully connected layers
        x = self.fc_layers(x)
        
        return torch.sigmoid(x)
    
    def get_architecture(self) -> str:
        """Returns a string representation of the model architecture."""
        lstm_arch = f"LSTM: {self.lstm_hidden_dims} hidden dims, {self.num_layers} layers, bidirectional={self.bidirectional}"
        fc_arch = ' -> '.join([str(2 * self.lstm_hidden_dims[-1] * (2 if self.bidirectional else 1))] + 
                             [str(dim) for dim in self.hidden_dims] + ['1'])
        seq_info = f"Seq length: {self.seq_length}, Features per step: {self.feature_per_step}"
        return f"{seq_info} | {lstm_arch} | FC: {fc_arch}"
    
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

def create_lstm_model(input_dim, lstm_hidden_dims=[128, 64], num_layers=[2, 2],
                      hidden_dims=[128, 64], dropout_rate=0.3, activation='relu',
                      bidirectional=True, initialization='xavier') -> LSTMClassifier:
    """
    Factory function to create and initialize LSTM model.
    
    Args:
        input_dim: Number of input features
        lstm_hidden_dims: List of hidden dimensions for each LSTM layer, default [128, 64]
        num_layers: List of number of layers for each LSTM, default [2, 2]
        hidden_dims: List of hidden layer dimensions for final FC layers, default [128, 64]
        dropout_rate: Dropout probability, default 0.3
        activation: Activation function to use, default 'relu'
        bidirectional: Whether to use bidirectional LSTM, default True
        initialization: Weight initialization method, default 'xavier'
        
    Returns:
        Initialized LSTM model
    """
    model = LSTMClassifier(input_dim, lstm_hidden_dims, num_layers, hidden_dims, 
                          dropout_rate, activation, bidirectional)
    
    # Initialize weights
    if initialization == 'xavier':
        for name, param in model.named_parameters():
            if 'weight_ih' in name or 'weight_hh' in name:
                # LSTM weights
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                # LSTM biases
                nn.init.zeros_(param)
                # Set forget gate bias to 1 for better learning
                if 'bias_hh' in name:
                    hidden_size = param.size(0) // 4
                    param.data[hidden_size:2*hidden_size].fill_(1.0)
        
        # Initialize FC layers
        for layer in model.modules():
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)
    elif initialization == 'kaiming':
        for name, param in model.named_parameters():
            if 'weight_ih' in name or 'weight_hh' in name:
                # LSTM weights
                nn.init.kaiming_normal_(param)
            elif 'bias' in name:
                # LSTM biases
                nn.init.zeros_(param)
                # Set forget gate bias to 1
                if 'bias_hh' in name:
                    hidden_size = param.size(0) // 4
                    param.data[hidden_size:2*hidden_size].fill_(1.0)
        
        # Initialize FC layers
        for layer in model.modules():
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_normal_(layer.weight)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)
    else:
        logger.warning(f"Unknown initialization method {initialization}, using default")
    
    return model 