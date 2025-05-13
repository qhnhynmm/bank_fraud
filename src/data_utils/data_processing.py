import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline as ImbPipeline
import logging
from src.data_utils.load_data import load_data, explore_data
from sklearn.model_selection import StratifiedShuffleSplit

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def balance_data(X, y):
    """
    Balance dataset using SMOTE and undersampling
    
    Args:
        X: Features
        y: Target variable
        
    Returns:
        X_resampled, y_resampled: Balanced dataset
    """
    # Combine SMOTE and undersampling
    over = SMOTE(sampling_strategy=0.1)  # Bring minority class up to 10% of majority
    under = RandomUnderSampler(sampling_strategy=0.5)  # Reduce majority class to have 2:1 ratio
    
    steps = [('o', over), ('u', under)]
    pipeline = ImbPipeline(steps=steps)
    
    X_res, y_res = pipeline.fit_resample(X, y)
    return X_res, y_res

def preprocess_data(data, target_col='Fraud'):
    """
    Preprocess the banking data for model training.
    
    Args:
        data (pd.DataFrame): The raw data
        target_col (str): Name of the target column
        
    Returns:
        tuple: Preprocessed data (X_train, X_val, X_test, y_train, y_val, y_test, preprocessor)
    """
    logger.info("Preprocessing data...")
    
    # Create a copy of the data to avoid modifying the original
    data = data.copy()
    
    # Verify target column exists and contains expected values
    if target_col not in data.columns:
        raise ValueError(f"Target column '{target_col}' not found in data")
    
    # Check target distribution before processing
    fraud_count = data[target_col].sum()
    total_count = len(data)
    logger.info(f"Initial fraud ratio: {fraud_count/total_count:.4f} ({fraud_count} fraudulent out of {total_count} transactions)")
    
    # Convert target to numeric if needed
    if data[target_col].dtype == 'object':
        data[target_col] = data[target_col].map({'Yes': 1, 'No': 0, True: 1, False: 0, 1: 1, 0: 0})
        logger.info("Converted target column to numeric")
    
    # Convert Date column and extract time-based features
    data['Date'] = pd.to_datetime(data['Date'], format='%d-%b-%y')
    data['Day of Week'] = data['Date'].dt.weekday + 1
    data['Hour'] = data['Date'].dt.hour
    data['Is_Weekend'] = data['Day of Week'].isin([6, 7]).astype(int)
    
    # Convert Amount and create amount-based features
    data['Amount'] = data['Amount'].replace({'£': '', ',': ''}, regex=True).astype(float)
    data['Amount_Log'] = np.log1p(data['Amount'])
    
    # Handle missing values
    numerical_cols = data.select_dtypes(include=['int64', 'float64']).columns
    categorical_cols = data.select_dtypes(include=['object']).columns
    
    # Fill numerical missing values
    for col in numerical_cols:
        data[col] = data[col].fillna(data[col].median())
    
    # Fill categorical missing values
    for col in categorical_cols:
        data[col] = data[col].fillna(data[col].mode()[0])
    
    # Handle outliers in numerical columns
    for col in numerical_cols:
        if col != target_col:  # Don't modify the target column
            Q1 = data[col].quantile(0.25)
            Q3 = data[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            data[col] = data[col].clip(lower_bound, upper_bound)
    
    # Drop unnecessary columns
    data = data.drop(['Transaction ID', 'Date'], axis=1)
    
    # Split features and target
    X = data.drop([target_col], axis=1)
    y = data[target_col]
    
    # Verify we still have fraud cases after preprocessing
    if len(y.unique()) != 2:
        raise ValueError(f"Lost fraud cases during preprocessing! Unique values in target: {y.unique()}")
    
    # Update categorical and numerical columns after feature engineering
    categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
    categorical_cols.extend(['Day of Week', 'Is_Weekend'])
    numerical_cols = [col for col in X.select_dtypes(include=['int64', 'float64']).columns if col not in categorical_cols]
    
    logger.info(f"Categorical columns: {categorical_cols}")
    logger.info(f"Numerical columns: {numerical_cols}")
    
    # Create preprocessing pipelines
    numerical_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())
    ])
    
    categorical_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    
    # Combine preprocessing steps
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_cols),
            ('cat', categorical_transformer, categorical_cols)
        ]
    )
    
    # Analyze class distribution
    fraud_indices = y[y == 1].index
    non_fraud_indices = y[y == 0].index
    
    logger.info(f"Total samples: {len(y)}")
    logger.info(f"Fraud samples: {len(fraud_indices)}")
    logger.info(f"Non-fraud samples: {len(non_fraud_indices)}")
    
    if len(fraud_indices) == 0:
        raise ValueError("No fraud cases found after preprocessing!")
    
    # Stratified split ensuring both classes are represented in all sets
    sss1 = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_val_idx, test_idx = next(sss1.split(X, y))
    
    X_temp, X_test = X.iloc[train_val_idx], X.iloc[test_idx]
    y_temp, y_test = y.iloc[train_val_idx], y.iloc[test_idx]
    
    # Second split: divide remaining data into train and validation
    sss2 = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, val_idx = next(sss2.split(X_temp, y_temp))
    
    X_train, X_val = X_temp.iloc[train_idx], X_temp.iloc[val_idx]
    y_train, y_val = y_temp.iloc[train_idx], y_temp.iloc[val_idx]
    
    # Log class distribution for each set
    logger.info("\nClass distribution in splits:")
    logger.info(f"Training set - Total: {len(y_train)}, Fraud ratio: {y_train.mean():.4f}")
    logger.info(f"Validation set - Total: {len(y_val)}, Fraud ratio: {y_val.mean():.4f}")
    logger.info(f"Test set - Total: {len(y_test)}, Fraud ratio: {y_test.mean():.4f}")
    
    # Apply preprocessing
    X_train_processed = preprocessor.fit_transform(X_train)
    X_val_processed = preprocessor.transform(X_val)
    X_test_processed = preprocessor.transform(X_test)
    
    # Balance only the training data
    logger.info("\nBalancing training data...")
    # Use SMOTE with more conservative ratios
    over = SMOTE(sampling_strategy=0.5, random_state=42)  # Bring minority class to 50% of majority
    under = RandomUnderSampler(sampling_strategy=0.8, random_state=42)  # Keep 80% of majority class
    
    # Apply balancing using imblearn pipeline
    balancing_pipeline = ImbPipeline([
        ('over', over),
        ('under', under)
    ])
    
    X_train_processed, y_train = balancing_pipeline.fit_resample(X_train_processed, y_train)
    logger.info(f"After balancing - Training set shape: {X_train_processed.shape}")
    logger.info(f"After balancing - Training set fraud ratio: {np.mean(y_train):.4f}")
    
    # Log final shapes
    logger.info("\nFinal dataset shapes:")
    logger.info(f"Training set: {X_train_processed.shape}")
    logger.info(f"Validation set: {X_val_processed.shape}")
    logger.info(f"Test set: {X_test_processed.shape}")
    
    return X_train_processed, X_val_processed, X_test_processed, y_train, y_val, y_test, preprocessor

def save_processed_data(X_train, X_test, y_train, y_test, output_dir):
    """
    Save the processed data to disk.
    
    Args:
        X_train, X_test, y_train, y_test: Processed data
        output_dir (str): Directory to save the data
    """
    logger.info(f"Saving processed data to {output_dir}")
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Save as numpy arrays
    np.save(os.path.join(output_dir, 'X_train.npy'), X_train)
    np.save(os.path.join(output_dir, 'X_test.npy'), X_test)
    np.save(os.path.join(output_dir, 'y_train.npy'), y_train)
    np.save(os.path.join(output_dir, 'y_test.npy'), y_test)
    
    logger.info("Processed data saved successfully")

def main():
    """Main function to run the preprocessing pipeline."""
    # Define file paths
    # Get the current script's directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # Navigate to the project root and then to the data directory
    root_dir = os.path.dirname(os.path.dirname(current_dir))  # Go up two levels
    
    input_file = os.path.join(root_dir, 'data', 'CreditCardData.csv')
    output_dir = os.path.join(root_dir, 'data', 'processed')
    
    logger.info(f"Looking for input file at: {input_file}")
    
    # Load and preprocess data
    data = load_data(input_file)
    explore_data(data)
    X_train, X_val, X_test, y_train, y_val, y_test, preprocessor = preprocess_data(data)
    save_processed_data(X_train, X_test, y_train, y_test, output_dir)
    
    logger.info("Preprocessing completed successfully")

if __name__ == "__main__":
    main() 