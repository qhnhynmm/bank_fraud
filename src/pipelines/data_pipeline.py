"""
Data pipeline module that combines data loading and preprocessing steps.
"""

import os
import yaml
import joblib
from typing import Dict, Tuple, Union
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
import logging

from src.data_utils.load_data import load_data, explore_data
from src.data_utils.data_processing import preprocess_data
logger = logging.getLogger(__name__)

class BankingDataPipeline:
    """Pipeline for loading and processing banking fraud data."""
    
    def __init__(self, config_path: str = None):
        """
        Initialize the pipeline with configuration.
        
        Args:
            config_path: Path to the configuration file. If None, uses default path.
        """
        if config_path is None:
            # Get the current file's directory
            current_dir = os.path.dirname(os.path.abspath(__file__))
            # Navigate to config file
            config_path = os.path.join(os.path.dirname(current_dir), 'config', 'config.yaml')
        
        self.config = self._load_config(config_path)
        self.preprocessor = None
        
        # Setup logging based on config
        if 'logging' in self.config:
            os.makedirs(os.path.dirname(self.config['logging']['file']), exist_ok=True)
            logging.basicConfig(
                level=self.config['logging']['level'],
                format=self.config['logging']['format'],
                filename=self.config['logging']['file']
            )
    
    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from yaml file."""
        logger.info(f"Loading config from {config_path}")
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            logger.info("Config loaded successfully")
            return config
        except Exception as e:
            logger.error(f"Error loading config: {e}")
            raise
    
    def create_preprocessing_pipeline(self, numerical_cols, categorical_cols):
        """
        Create the preprocessing pipeline.
        
        Args:
            numerical_cols: List of numerical column names
            categorical_cols: List of categorical column names
        """
        numerical_transformer = Pipeline(steps=[
            ('scaler', StandardScaler())
        ])
        
        categorical_transformer = Pipeline(steps=[
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ])
        
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numerical_transformer, numerical_cols),
                ('cat', categorical_transformer, categorical_cols)
            ]
        )
        
        return preprocessor
    
    def create_balancing_pipeline(self):
        """Create the data balancing pipeline."""
        over = SMOTE(
            sampling_strategy=self.config['preprocessing']['smote_ratio'],
            random_state=self.config['preprocessing']['random_state']
        )
        under = RandomUnderSampler(
            sampling_strategy=self.config['preprocessing']['undersampling_ratio'],
            random_state=self.config['preprocessing']['random_state']
        )
        
        return ImbPipeline([
            ('oversample', over),
            ('undersample', under)
        ])
    
    def run_pipeline(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Run the complete data pipeline.
        
        Returns:
            Tuple containing (X_train, X_val, X_test, y_train, y_val, y_test)
        """
        logger.info("Starting data pipeline...")
        
        # Load data
        data = load_data(self.config)
        
        # Explore data
        explore_data(data)
        
        # Preprocess data
        X_train, X_val, X_test, y_train, y_val, y_test, self.preprocessor = preprocess_data(
            data, 
            target_col=self.config['data']['target_column']
        )
        
        # Save preprocessor for inference
        self._save_preprocessor()
        
        # Log data shapes and class distributions
        logger.info("Data split summary:")
        logger.info(f"Training set shape: {X_train.shape}, Fraud ratio: {np.mean(y_train):.4f}")
        logger.info(f"Validation set shape: {X_val.shape}, Fraud ratio: {np.mean(y_val):.4f}")
        logger.info(f"Test set shape: {X_test.shape}, Fraud ratio: {np.mean(y_test):.4f}")
        
        logger.info("Data pipeline completed successfully")
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def _save_preprocessor(self):
        """Save the preprocessor for later use in inference."""
        output_dir = self.config['data']['processed_dir']
        os.makedirs(output_dir, exist_ok=True)
        
        preprocessor_path = os.path.join(output_dir, 'preprocessor.joblib')
        joblib.dump(self.preprocessor, preprocessor_path)
        logger.info(f"Saved preprocessor to {preprocessor_path}")
    
    def load_preprocessor(self):
        """Load the saved preprocessor."""
        preprocessor_path = os.path.join(
            self.config['data']['processed_dir'],
            'preprocessor.joblib'
        )
        self.preprocessor = joblib.load(preprocessor_path)
        return self.preprocessor
    
    def get_feature_dim(self) -> int:
        """
        Get the dimension of features after preprocessing.
        
        Returns:
            int: Feature dimension
        """
        # Try to load preprocessor if not already loaded
        if self.preprocessor is None:
            try:
                self.load_preprocessor()
            except Exception as e:
                logger.warning(f"Could not load preprocessor: {e}")
                # Estimate based on config
                return 52  # Reasonable default value based on our data
        
        # For ColumnTransformer, we need to check each transformer
        if hasattr(self.preprocessor, 'transformers_'):
            # Get dimensions from each transformer
            total_dims = 0
            for name, transformer, cols in self.preprocessor.transformers_:
                if name == 'cat' and hasattr(transformer, 'named_steps') and 'onehot' in transformer.named_steps:
                    # For OneHotEncoder, get the number of output features
                    onehot = transformer.named_steps['onehot']
                    if hasattr(onehot, 'categories_'):
                        total_dims += sum(len(c) for c in onehot.categories_)
                elif name == 'num':
                    # For numerical features, just count them
                    total_dims += len(cols)
            
            if total_dims > 0:
                logger.info(f"Feature dimension from preprocessor: {total_dims}")
                return total_dims
        
        # If we can't determine the exact dimension, try to infer it from a sample transaction
        try:
            # Create a sample transaction manually instead of using ModelInference
            sample_data = {
                'Transaction ID': '#9999 999',
                'Date': '14-Oct-20',
                'Day of Week': 'Wednesday',
                'Time': 12,
                'Type of Card': 'Visa',
                'Entry Mode': 'PIN',
                'Amount': 250,
                'Type of Transaction': 'POS',
                'Merchant Group': 'Electronics',
                'Country of Transaction': 'United Kingdom',
                'Shipping Address': 'United Kingdom',
                'Country of Residence': 'United Kingdom',
                'Gender': 'M',
                'Age': 35.0,
                'Bank': 'Barclays'
            }
            sample_df = pd.DataFrame([sample_data])
            
            # Preprocess it
            processed = self.preprocess_inference_data(sample_df)
            
            # Get dimension from processed data shape
            import scipy.sparse
            if scipy.sparse.issparse(processed):
                feature_dim = processed.shape[1]
            else:
                feature_dim = processed.shape[1] if len(processed.shape) > 1 else processed.shape[0]
                
            logger.info(f"Feature dimension from sample data: {feature_dim}")
            return feature_dim
        except Exception as e:
            logger.warning(f"Could not determine feature dimension from sample: {e}")
        
        # Default fallback
        logger.warning("Could not determine feature dimension precisely, using estimate of 52")
        return 52  # Reasonable default value based on our observed data shape
    
    def preprocess_inference_data(self, data: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        Preprocess data for inference.
        
        Args:
            data: Input data (DataFrame or array)
            
        Returns:
            Preprocessed features as numpy array
        """
        # Ensure we have a preprocessor
        if self.preprocessor is None:
            try:
                self.load_preprocessor()
            except Exception as e:
                logger.error(f"Could not load preprocessor: {e}")
                # If we can't load the preprocessor, try to prepare data without it
                logger.info("Attempting to preprocess data without preprocessor...")
                return self._preprocess_without_preprocessor(data)
        
        # Convert to DataFrame if it's a numpy array
        if isinstance(data, np.ndarray):
            try:
                # Try to infer column names from preprocessor if available
                if hasattr(self.preprocessor, 'feature_names_in_'):
                    column_names = self.preprocessor.feature_names_in_
                    # Check if dimensions match
                    if data.shape[1] != len(column_names):
                        logger.warning(f"Input data has {data.shape[1]} columns but preprocessor expects {len(column_names)}. Using generic column names.")
                        column_names = [f'feature_{i}' for i in range(data.shape[1])]
                else:
                    # Use generic column names
                    column_names = [f'feature_{i}' for i in range(data.shape[1])]
                
                data = pd.DataFrame(data, columns=column_names)
            except Exception as e:
                logger.error(f"Error converting array to DataFrame: {e}")
                # Return original data if conversion fails
                return data
        
        # Copy to avoid modifying the original
        data = data.copy()
        
        # Log original data shape
        logger.info(f"Original inference data shape: {data.shape}")
        
        try:
            # Basic preprocessing similar to what we do in training
            # Prepare Date column if it exists
            if 'Date' in data.columns:
                try:
                    data['Date'] = pd.to_datetime(data['Date'], format='%d-%b-%y')
                except Exception as e:
                    logger.warning(f"Could not convert Date column: {e}")
            
            # Convert Amount to float if it's a string with currency symbol
            if 'Amount' in data.columns and isinstance(data['Amount'].iloc[0], str):
                try:
                    data['Amount'] = data['Amount'].replace({'£': '', ',': ''}, regex=True).astype(float)
                    logger.info("Converted Amount to numeric")
                except Exception as e:
                    logger.warning(f"Could not convert Amount column: {e}")
            
            # Feature engineering - derive standard features
            if 'Day of Week' in data.columns and isinstance(data['Day of Week'].iloc[0], str):
                try:
                    # Convert text days to numbers
                    day_mapping = {
                        'Monday': 1, 'Tuesday': 2, 'Wednesday': 3, 'Thursday': 4, 
                        'Friday': 5, 'Saturday': 6, 'Sunday': 7
                    }
                    data['Day of Week'] = data['Day of Week'].map(day_mapping)
                    logger.info("Converted Day of Week to numeric")
                except Exception as e:
                    logger.warning(f"Could not convert Day of Week column: {e}")
            
            # Create Is_Weekend column if it doesn't exist
            if 'Is_Weekend' not in data.columns and 'Day of Week' in data.columns:
                try:
                    data['Is_Weekend'] = (data['Day of Week'].isin([6, 7])).astype(int)
                    logger.info("Created Is_Weekend feature")
                except Exception as e:
                    logger.warning(f"Could not create Is_Weekend feature: {e}")
            
            # Create Amount_Log if it doesn't exist
            if 'Amount_Log' not in data.columns and 'Amount' in data.columns:
                try:
                    data['Amount_Log'] = np.log1p(data['Amount'])
                    logger.info("Created Amount_Log feature")
                except Exception as e:
                    logger.warning(f"Could not create Amount_Log feature: {e}")
            
            # Apply preprocessor for consistent transformation
            try:
                processed_data = self.preprocessor.transform(data)
                logger.info(f"Preprocessed data shape: {processed_data.shape}")
                return processed_data
            except Exception as e:
                logger.error(f"Error preprocessing inference data with preprocessor: {e}")
                # If preprocessor transform fails, try a simpler approach
                return self._preprocess_without_preprocessor(data)
            
        except Exception as e:
            logger.error(f"Error preprocessing inference data: {e}")
            raise
    
    def _preprocess_without_preprocessor(self, data: pd.DataFrame) -> np.ndarray:
        """
        Preprocess data without using the saved preprocessor (fallback method).
        
        Args:
            data: Input data (DataFrame)
            
        Returns:
            Preprocessed features as numpy array
        """
        logger.info("Using fallback preprocessing method")
        
        try:
            # Make a copy to avoid modifying original
            df = data.copy()
            
            # Drop non-numeric columns that typically don't help in prediction
            cols_to_drop = ['Transaction ID', 'Date']
            for col in cols_to_drop:
                if col in df.columns:
                    df = df.drop(col, axis=1)
            
            # Drop target column if present
            if 'Fraud' in df.columns:
                df = df.drop('Fraud', axis=1)
            
            # Convert categorical variables to dummy variables
            categorical_cols = df.select_dtypes(include=['object']).columns
            if len(categorical_cols) > 0:
                df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
                logger.info(f"Converted {len(categorical_cols)} categorical columns to dummies")
            
            # Convert to numeric where possible
            for col in df.columns:
                if df[col].dtype == 'object':
                    try:
                        df[col] = pd.to_numeric(df[col])
                    except:
                        # If conversion fails, drop the column
                        df = df.drop(col, axis=1)
            
            # Make sure all data is numeric
            numeric_df = df.select_dtypes(include=['number'])
            
            # Fill missing values
            numeric_df = numeric_df.fillna(0)
            
            # Log the final shape
            logger.info(f"Fallback preprocessing result shape: {numeric_df.shape}")
            
            return numeric_df.values
            
        except Exception as e:
            logger.error(f"Fallback preprocessing failed: {e}")
            # As a last resort, just return dummy data
            logger.warning("Returning dummy data as preprocessing failed")
            return np.zeros((data.shape[0], 10))

def run_data_pipeline(config_path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Convenience function to run the data pipeline.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Tuple containing (X_train, X_val, X_test, y_train, y_val, y_test)
    """
    pipeline = BankingDataPipeline(config_path)
    return pipeline.run_pipeline()

if __name__ == "__main__":
    # Example usage
    pipeline = BankingDataPipeline()  # Will use default config path
    X_train, X_val, X_test, y_train, y_val, y_test = pipeline.run_pipeline() 