"""
Basic machine learning models for fraud detection.
"""

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)

class ModelFactory:
    """Factory class for creating ML models."""
    
    @staticmethod
    def create_model(model_name: str, model_params: Dict[str, Any]):
        """
        Create a model instance based on model name and parameters.
        
        Args:
            model_name: Name of the model to create
            model_params: Model parameters
            
        Returns:
            Initialized model instance
        """
        models = {
            'random_forest': RandomForestClassifier,
            'gradient_boosting': GradientBoostingClassifier,
            'logistic_regression': LogisticRegression,
            'svm': SVC,
            'decision_tree': DecisionTreeClassifier,
            'xgboost': XGBClassifier,
            'lightgbm': LGBMClassifier
        }
        
        if model_name not in models:
            raise ValueError(f"Model {model_name} not supported. Available models: {list(models.keys())}")
        
        logger.info(f"Creating {model_name} model with parameters: {model_params}")
        return models[model_name](**model_params)

class RandomForestModel:
    """Random Forest model wrapper."""
    
    @staticmethod
    def get_default_params():
        return {
            'n_estimators': 100,
            'max_depth': 10,
            'min_samples_split': 2,
            'min_samples_leaf': 1,
            'random_state': 42
        }

class GradientBoostingModel:
    """Gradient Boosting model wrapper."""
    
    @staticmethod
    def get_default_params():
        return {
            'n_estimators': 100,
            'learning_rate': 0.1,
            'max_depth': 3,
            'min_samples_split': 2,
            'random_state': 42
        }

class LogisticRegressionModel:
    """Logistic Regression model wrapper."""
    
    @staticmethod
    def get_default_params():
        return {
            'C': 1.0,
            'max_iter': 1000,
            'class_weight': 'balanced',
            'random_state': 42
        }

class SVMModel:
    """Support Vector Machine model wrapper."""
    
    @staticmethod
    def get_default_params():
        return {
            'C': 1.0,
            'kernel': 'rbf',
            'probability': True,
            'random_state': 42
        }

class XGBoostModel:
    """XGBoost model wrapper."""
    
    @staticmethod
    def get_default_params():
        return {
            'n_estimators': 100,
            'learning_rate': 0.1,
            'max_depth': 6,
            'min_child_weight': 1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': 42
        }

class LightGBMModel:
    """LightGBM model wrapper."""
    
    @staticmethod
    def get_default_params():
        return {
            'n_estimators': 100,
            'learning_rate': 0.1,
            'max_depth': 6,
            'num_leaves': 31,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'random_state': 42
        } 