# dataloader.py
import pandas as pd
import os
import logging

logging.basicConfig(filename = "load_data.log",level=logging.INFO,format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

def load_cicddos2019(data_path='./data', sample_size=None):
     
    try:
         
        files = [f for f in os.listdir(data_path) if f.endswith('.csv')]
        
        if not files:
            logger.warning(f"No CSV files found in {data_path}. Using synthetic data.")
            return None
             
        data_file = os.path.join(data_path, files[0])
        logger.info(f"Loading dataset from {data_file}")
        
        df = pd.read_csv(data_file)
         
        if sample_size and sample_size < len(df):
            df = df.sample(sample_size, random_state=42)
            
        logger.info(f"Dataset loaded: {len(df)} records, {df.columns.size} features")
        return df
        
    except Exception as e:
        logger.error(f"Error loading dataset: {str(e)}")
        return None