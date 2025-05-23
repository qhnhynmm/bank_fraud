import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import logging
from typing import Dict, Tuple, Union, Any
import pandas as pd
import scipy.sparse

from src.model.mlp_model import create_mlp_model
from src.model.cnn_model import create_cnn_model
from src.model.lstm_model import create_lstm_model
from src.model.ml_models import ModelFactory
from src.pipelines.data_pipeline import BankingDataPipeline
from src.metrics.metrics import FraudMetrics

logger = logging.getLogger(__name__)

class ModelTrainer:
    """Class to handle model training and evaluation."""
    
    def __init__(self, config: Dict):
        """
        Initialize trainer with configuration.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
        
        # Set random seeds
        torch.manual_seed(config['preprocessing']['random_state'])
        np.random.seed(config['preprocessing']['random_state'])
        
        # Initialize metrics tracking
        self.best_val_metric = 0.0
        self.patience_counter = 0
        
        # Initialize data pipeline
        self.pipeline = BankingDataPipeline()
    
    def create_model(self) -> Union[nn.Module, Any]:
        """
        Create model based on configuration.
        
        Returns:
            Created model instance
        """
        model_type = self.config['model']['type']
        logger.info(f"Creating {model_type} model")
        
        if model_type == 'mlp':
            # Sử dụng thông số mặc định từ hàm create_mlp_model
            return create_mlp_model(
                input_dim=self.input_dim,
                hidden_dims=[128, 64, 32],  # Thông số mặc định
                dropout_rate=0.3,
                activation='relu',
                initialization='xavier'
            ).to(self.device)
        elif model_type == 'cnn':
            return create_cnn_model(
                input_dim=self.input_dim,
                num_filters=self.config['model']['cnn']['num_filters'],
                kernel_sizes=self.config['model']['cnn']['kernel_sizes'], 
                hidden_dims=self.config['model']['cnn']['hidden_dims'],
                dropout_rate=self.config['model']['cnn']['dropout_rate'],
                activation='relu',
                initialization='xavier'
            ).to(self.device)
        elif model_type == 'lstm':
            return create_lstm_model(
                input_dim=self.input_dim,
                lstm_hidden_dims=self.config['model']['lstm']['lstm_hidden_dims'],
                num_layers=self.config['model']['lstm']['num_layers'],
                hidden_dims=self.config['model']['lstm']['hidden_dims'],
                dropout_rate=self.config['model']['lstm']['dropout_rate'],
                bidirectional=self.config['model']['lstm']['bidirectional'],
                activation='relu',
                initialization='xavier'
            ).to(self.device)
        else:
            # Gọi trực tiếp đến các class model với tham số mặc định
            return ModelFactory.create_model(model_type)
    
    def train_deep_learning_model(self, X_train: np.ndarray, X_val: np.ndarray, X_test: np.ndarray,
                          y_train: np.ndarray, y_val: np.ndarray, y_test: np.ndarray) -> nn.Module:
        """Train PyTorch model."""
        train_loader, val_loader, test_loader = self.prepare_data(
            X_train, X_val, X_test, y_train, y_val, y_test
        )
        
        # Create model
        self.input_dim = X_train.shape[1]
        model = self.create_model()
        
        # Create checkpoints directory if it doesn't exist
        os.makedirs('checkpoints', exist_ok=True)
        
        # Setup training
        criterion = nn.BCELoss()
        optimizer = optim.Adam(
            model.parameters(),
            lr=0.001  # Thông số mặc định
        )
        
        # Training loop
        best_model_state = None
        best_val_metrics = None
        num_epochs = 50  # Thông số mặc định
        patience = 5     # Thông số mặc định
        
        logger.info(f"Starting training for {num_epochs} epochs")
        for epoch in range(num_epochs):
            # Train
            train_loss = self.train_epoch(model, train_loader, criterion, optimizer)
            
            # Evaluate
            val_loss, val_metrics = self.evaluate(model, val_loader, criterion)
            test_loss, test_metrics = self.evaluate(model, test_loader, criterion)
            
            # Log metrics
            logger.info(f"Epoch {epoch+1}/{num_epochs}")
            logger.info(f"Train Loss: {train_loss:.4f}")
            logger.info("Validation Metrics:")
            FraudMetrics.print_metrics(val_metrics)
            
            # Early stopping based on validation F1 score
            if best_val_metrics is None or val_metrics['f1'] > best_val_metrics['f1']:
                best_val_metrics = val_metrics
                self.patience_counter = 0
                # Save best model
                best_model_state = model.state_dict()
                model_path = os.path.join('checkpoints', f'best_model_{self.config["model"]["type"]}.pt')
                torch.save(best_model_state, model_path)
                logger.info(f"Saved best model to {model_path}")
            else:
                self.patience_counter += 1
                if self.patience_counter >= patience:
                    logger.info("Early stopping triggered")
                    break
        
        # Load best model for final evaluation
        if best_model_state is not None:
            model.load_state_dict(best_model_state)
            final_test_loss, final_metrics = self.evaluate(model, test_loader, criterion)
            logger.info("Final Test Metrics:")
            FraudMetrics.print_metrics(final_metrics)
        
        return model
    
    def train_ml_model(self, X_train: np.ndarray, X_val: np.ndarray, X_test: np.ndarray,
                      y_train: np.ndarray, y_val: np.ndarray, y_test: np.ndarray) -> Any:
        """Train machine learning model."""
        # Create and train model
        self.input_dim = X_train.shape[1]
        model = self.create_model()
        
        logger.info(f"Training {self.config['model']['type']} model")
        model.fit(X_train, y_train)
        
        # Get predictions
        train_pred = model.predict(X_train)
        val_pred = model.predict(X_val)
        test_pred = model.predict(X_test)
        
        # Calculate metrics
        train_metrics = FraudMetrics.calculate_metrics(y_train, train_pred)
        val_metrics = FraudMetrics.calculate_metrics(y_val, val_pred)
        test_metrics = FraudMetrics.calculate_metrics(y_test, test_pred)
        
        # Log metrics
        logger.info("Training Metrics:")
        FraudMetrics.print_metrics(train_metrics)
        logger.info("Validation Metrics:")
        FraudMetrics.print_metrics(val_metrics)
        logger.info("Test Metrics:")
        FraudMetrics.print_metrics(test_metrics)
        
        # Save model
        os.makedirs('checkpoints', exist_ok=True)
        model_path = os.path.join('checkpoints', f'best_model_{self.config["model"]["type"]}.joblib')
        
        # Save using joblib
        import joblib
        joblib.dump(model, model_path)
        logger.info(f"Saved model to {model_path}")
        
        return model
    
    def train(self) -> Union[nn.Module, Any]:
        """
        Train the model based on configuration.
        
        Returns:
            Trained model
        """
        # Load and prepare data
        X_train, X_val, X_test, y_train, y_val, y_test = self.pipeline.run_pipeline()
        
        # Train based on model type
        if self.config['model']['type'] == 'mlp':
            return self.train_deep_learning_model(X_train, X_val, X_test, y_train, y_val, y_test)
        elif self.config['model']['type'] == 'cnn':
            return self.train_deep_learning_model(X_train, X_val, X_test, y_train, y_val, y_test)
        elif self.config['model']['type'] == 'lstm':
            return self.train_deep_learning_model(X_train, X_val, X_test, y_train, y_val, y_test)
        else:
            return self.train_ml_model(X_train, X_val, X_test, y_train, y_val, y_test)
    
    def prepare_data(self, X_train: np.ndarray, X_val: np.ndarray, X_test: np.ndarray,
                    y_train: np.ndarray, y_val: np.ndarray, y_test: np.ndarray) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """
        Prepare data loaders for training.
        
        Args:
            X_train, X_val, X_test: Training, validation and test features
            y_train, y_val, y_test: Training, validation and test labels
            
        Returns:
            Training, validation and test data loaders
        """
        # Convert sparse matrices to dense if needed
        if scipy.sparse.issparse(X_train):
            X_train = X_train.toarray()
        if scipy.sparse.issparse(X_val):
            X_val = X_val.toarray()
        if scipy.sparse.issparse(X_test):
            X_test = X_test.toarray()
        
        # Convert pandas Series to numpy arrays if needed
        if isinstance(y_train, pd.Series):
            y_train = y_train.to_numpy()
        if isinstance(y_val, pd.Series):
            y_val = y_val.to_numpy()
        if isinstance(y_test, pd.Series):
            y_test = y_test.to_numpy()
        
        # Ensure correct shape for labels
        y_train = y_train.reshape(-1, 1)
        y_val = y_val.reshape(-1, 1)
        y_test = y_test.reshape(-1, 1)
        
        # Convert to PyTorch tensors
        X_train = torch.FloatTensor(X_train)
        y_train = torch.FloatTensor(y_train)
        X_val = torch.FloatTensor(X_val)
        y_val = torch.FloatTensor(y_val)
        X_test = torch.FloatTensor(X_test)
        y_test = torch.FloatTensor(y_test)
        
        # Create datasets
        train_dataset = TensorDataset(X_train, y_train)
        val_dataset = TensorDataset(X_val, y_val)
        test_dataset = TensorDataset(X_test, y_test)
        
        # Create data loaders
        batch_size = self.config['training']['batch_size']
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        return train_loader, val_loader, test_loader
    
    def train_epoch(self, model: nn.Module, train_loader: DataLoader, 
                   criterion: nn.Module, optimizer: optim.Optimizer) -> float:
        """
        Train for one epoch.
        
        Args:
            model: PyTorch model
            train_loader: Training data loader
            criterion: Loss function
            optimizer: Optimizer
            
        Returns:
            Average training loss
        """
        model.train()
        total_loss = 0
        
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
            
            # Forward pass
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            
            # Backward pass and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        return total_loss / len(train_loader)
    
    def evaluate(self, model: nn.Module, test_loader: DataLoader, criterion: nn.Module) -> Tuple[float, Dict[str, float]]:
        """
        Evaluate model on test data.
        
        Args:
            model: PyTorch model
            test_loader: Test data loader
            criterion: Loss function
            
        Returns:
            Tuple of (loss, metrics dictionary)
        """
        model.eval()
        total_loss = 0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                total_loss += loss.item()
                
                preds = (outputs >= 0.5).float()
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        avg_loss = total_loss / len(test_loader)
        metrics = FraudMetrics.calculate_metrics(
            np.array(all_labels),
            np.array(all_preds)
        )
        
        return avg_loss, metrics

def main():
    """Main training function."""
    # Load config
    pipeline = BankingDataPipeline()
    config = pipeline.config
    
    # Train model
    trainer = ModelTrainer(config)
    model = trainer.train()
    
    logger.info("Training completed successfully")
    return model

if __name__ == "__main__":
    main()
