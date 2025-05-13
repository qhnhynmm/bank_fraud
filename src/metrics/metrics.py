import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from typing import Dict, Union, List

class FraudMetrics:
    @staticmethod
    def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray = None) -> Dict[str, float]:
        """
        Calculate fraud detection metrics.
        
        Args:
            y_true: Ground truth labels (0 or 1)
            y_pred: Predicted labels (0 or 1)
            y_prob: Predicted probabilities (optional)
            
        Returns:
            Dictionary containing the metrics
        """
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred),
            'recall': recall_score(y_true, y_pred),
            'f1': f1_score(y_true, y_pred)
        }
        
        return metrics
    
    @staticmethod
    def print_metrics(metrics: Dict[str, float]) -> None:
        """
        Print metrics in a formatted way.
        
        Args:
            metrics: Dictionary of metric names and values
        """
        print("\nModel Performance Metrics:")
        print("-" * 30)
        for metric_name, value in metrics.items():
            print(f"{metric_name.capitalize():>10}: {value:.4f}")
        print("-" * 30)

    @staticmethod
    def get_metric_names() -> List[str]:
        """
        Get list of available metrics.
        
        Returns:
            List of metric names
        """
        return ['accuracy', 'precision', 'recall', 'f1']