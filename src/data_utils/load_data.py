import pandas as pd
import logging
from typing import Dict

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_data(config: Dict):
    """
    Load data from CSV file specified in config.
    
    Args:
        config (Dict): Configuration dictionary containing data_path
        
    Returns:
        pd.DataFrame: Loaded data
        
    Raises:
        Exception: If there is an error loading the data
    """
    logger.info(f"Loading data from {config['data']['data_path']}")
    try:
        data = pd.read_csv(config['data']['data_path'])
        logger.info(f"Successfully loaded data with shape {data.shape}")
        return data
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        raise

def explore_data(data):
    """
    Explore the dataset and print summary statistics.
    
    Args:
        data (pd.DataFrame): The data to explore
    """
    logger.info("Exploring data...")
    logger.info(f"Data shape: {data.shape}")
    logger.info(f"Columns: {data.columns.tolist()}")
    
    # Check for missing values
    missing_values = data.isnull().sum()
    logger.info(f"Missing values per column:\n{missing_values}")
    
    # Basic statistics
    logger.info(f"Basic statistics:\n{data.describe()}")
    
    # Check target distribution
    if 'Fraud' in data.columns:
        logger.info(f"Target distribution:\n{data['Fraud'].value_counts()}")
        logger.info(f"Target distribution (%):\n{data['Fraud'].value_counts(normalize=True) * 100}")
