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
    def create_model(model_name: str, **kwargs):
        """
        Create a model instance based on model name and parameters.
        
        Args:
            model_name: Name of the model to create
            **kwargs: Model parameters (optional)
            
        Returns:
            Initialized model instance
        """
        if model_name == 'random_forest':
            return RandomForestModel.create(**kwargs)
        elif model_name == 'gradient_boosting':
            return GradientBoostingModel.create(**kwargs)
        elif model_name == 'logistic_regression':
            return LogisticRegressionModel.create(**kwargs)
        elif model_name == 'svm':
            return SVMModel.create(**kwargs)
        elif model_name == 'xgboost':
            return XGBoostModel.create(**kwargs)
        elif model_name == 'lightgbm':
            return LightGBMModel.create(**kwargs)
        else:
            raise ValueError(f"Model {model_name} not supported. Available models: random_forest, gradient_boosting, logistic_regression, svm, xgboost, lightgbm")

class RandomForestModel:
    """Random Forest model wrapper."""
    
    @staticmethod
    def create(n_estimators=100, max_depth=10, min_samples_split=2, 
               min_samples_leaf=1, max_features='sqrt', 
               class_weight='balanced', random_state=42, **kwargs):
        """
        Create a Random Forest model with specified parameters.
        
        Args:
            n_estimators: Number of trees in the forest (default: 100)
            max_depth: Maximum depth of the trees (default: 10)
            min_samples_split: Minimum samples required to split a node (default: 2)
            min_samples_leaf: Minimum samples required in a leaf node (default: 1)
            max_features: Number of features to consider for best split (default: 'sqrt')
            class_weight: Weights associated with classes (default: 'balanced')
            random_state: Random seed for reproducibility (default: 42)
            **kwargs: Additional parameters to pass to RandomForestClassifier
            
        Returns:
            RandomForestClassifier instance
        """
        params = {
            'n_estimators': n_estimators,
            'max_depth': max_depth,
            'min_samples_split': min_samples_split,
            'min_samples_leaf': min_samples_leaf,
            'max_features': max_features,
            'class_weight': class_weight,
            'random_state': random_state,
            **kwargs
        }
        logger.info(f"Creating RandomForest model with parameters: {params}")
        return RandomForestClassifier(**params)

class GradientBoostingModel:
    """Gradient Boosting model wrapper."""
    
    @staticmethod
    def create(n_estimators=100, learning_rate=0.1, max_depth=3, 
               min_samples_split=2, random_state=42, **kwargs):
        """
        Create a Gradient Boosting model with specified parameters.
        
        Args:
            n_estimators: Number of boosting stages (default: 100)
            learning_rate: Shrinks the contribution of each tree (default: 0.1)
            max_depth: Maximum depth of the trees (default: 3)
            min_samples_split: Minimum samples required to split a node (default: 2)
            random_state: Random seed for reproducibility (default: 42)
            **kwargs: Additional parameters to pass to GradientBoostingClassifier
            
        Returns:
            GradientBoostingClassifier instance
        """
        params = {
            'n_estimators': n_estimators,
            'learning_rate': learning_rate,
            'max_depth': max_depth,
            'min_samples_split': min_samples_split,
            'random_state': random_state,
            **kwargs
        }
        logger.info(f"Creating GradientBoosting model with parameters: {params}")
        return GradientBoostingClassifier(**params)

class LogisticRegressionModel:
    """Logistic Regression model wrapper."""
    
    @staticmethod
    def create(C=1.0, max_iter=1000, class_weight='balanced', 
               random_state=42, **kwargs):
        """
        Create a Logistic Regression model with specified parameters.
        
        Args:
            C: Inverse of regularization strength (default: 1.0)
            max_iter: Maximum number of iterations (default: 1000)
            class_weight: Weights associated with classes (default: 'balanced')
            random_state: Random seed for reproducibility (default: 42)
            **kwargs: Additional parameters to pass to LogisticRegression
            
        Returns:
            LogisticRegression instance
        """
        params = {
            'C': C,
            'max_iter': max_iter,
            'class_weight': class_weight,
            'random_state': random_state,
            **kwargs
        }
        logger.info(f"Creating LogisticRegression model with parameters: {params}")
        return LogisticRegression(**params)

class SVMModel:
    """Support Vector Machine model wrapper."""
    
    @staticmethod
    def create(C=1.0, kernel='rbf', probability=True, random_state=42, **kwargs):
        """
        Create a Support Vector Machine model with specified parameters.
        
        Args:
            C: Regularization parameter (default: 1.0)
            kernel: Kernel type (default: 'rbf')
            probability: Enable probability estimates (default: True)
            random_state: Random seed for reproducibility (default: 42)
            **kwargs: Additional parameters to pass to SVC
            
        Returns:
            SVC instance
        """
        params = {
            'C': C,
            'kernel': kernel,
            'probability': probability,
            'random_state': random_state,
            **kwargs
        }
        logger.info(f"Creating SVM model with parameters: {params}")
        return SVC(**params)

class XGBoostModel:
    """XGBoost model wrapper."""
    
    @staticmethod
    def create(n_estimators=100, learning_rate=0.1, max_depth=6, 
               min_child_weight=1, subsample=0.8, colsample_bytree=0.8, 
               gamma=0, scale_pos_weight=1, random_state=42, **kwargs):
        """
        Create an XGBoost model with specified parameters.
        
        Args:
            n_estimators: Number of boosting rounds (default: 100)
            learning_rate: Step size shrinkage (default: 0.1)
            max_depth: Maximum depth of a tree (default: 6)
            min_child_weight: Minimum sum of instance weight needed in a child (default: 1)
            subsample: Subsample ratio of the training instances (default: 0.8)
            colsample_bytree: Subsample ratio of columns when constructing each tree (default: 0.8)
            gamma: Minimum loss reduction required to make a further partition (default: 0)
            scale_pos_weight: Controls the balance of positive and negative weights (default: 1)
            random_state: Random seed for reproducibility (default: 42)
            **kwargs: Additional parameters to pass to XGBClassifier
            
        Returns:
            XGBClassifier instance
        """
        params = {
            'n_estimators': n_estimators,
            'learning_rate': learning_rate,
            'max_depth': max_depth,
            'min_child_weight': min_child_weight,
            'subsample': subsample,
            'colsample_bytree': colsample_bytree,
            'gamma': gamma,
            'scale_pos_weight': scale_pos_weight,
            'random_state': random_state,
            **kwargs
        }
        logger.info(f"Creating XGBoost model with parameters: {params}")
        return XGBClassifier(**params)

class LightGBMModel:
    """LightGBM model wrapper."""
    
    @staticmethod
    def create(n_estimators=100, learning_rate=0.1, max_depth=6, 
               num_leaves=31, feature_fraction=0.8, bagging_fraction=0.8, 
               bagging_freq=5, min_child_samples=20, scale_pos_weight=1, 
               random_state=42, **kwargs):
        """
        Create a LightGBM model with specified parameters.
        
        Args:
            n_estimators: Number of boosting iterations (default: 100)
            learning_rate: Boosting learning rate (default: 0.1)
            max_depth: Maximum tree depth (default: 6)
            num_leaves: Maximum tree leaves for base learners (default: 31)
            feature_fraction: LightGBM will randomly select a subset of features on each iteration (default: 0.8)
            bagging_fraction: Like feature_fraction, but for data (default: 0.8)
            bagging_freq: Frequency for bagging (default: 5)
            min_child_samples: Minimum number of data needed in a leaf (default: 20)
            scale_pos_weight: Weight of positive class in binary classification (default: 1)
            random_state: Random seed for reproducibility (default: 42)
            **kwargs: Additional parameters to pass to LGBMClassifier
            
        Returns:
            LGBMClassifier instance
        """
        params = {
            'n_estimators': n_estimators,
            'learning_rate': learning_rate,
            'max_depth': max_depth,
            'num_leaves': num_leaves,
            'feature_fraction': feature_fraction,
            'bagging_fraction': bagging_fraction,
            'bagging_freq': bagging_freq,
            'min_child_samples': min_child_samples,
            'scale_pos_weight': scale_pos_weight,
            'random_state': random_state,
            **kwargs
        }
        logger.info(f"Creating LightGBM model with parameters: {params}")
        return LGBMClassifier(**params) 