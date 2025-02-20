# prediction.py
import os
import numpy as np
import pandas as pd
import tensorflow as tf
import joblib
from collections import deque
import logging
from datetime import datetime
logging.basicConfig(filename="prediction.log", level=logging.INFO,
                   format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class PredictionModel:
    def __init__(self, model_dir='./model', sequence_length=10):
        self.model_dir = model_dir
        self.sequence_length = sequence_length
        self.model = None
        self.scaler = None
        self.label_encoder = None
        self.feature_columns = None
        self.history = deque(maxlen=100)
        self.window = deque(maxlen=sequence_length)
        self.last_prediction_time = None
        self.prediction_threshold = 0.6
        # Load model and artifacts
        self.load_model()
        
    def load_model(self):
        logger.info("Loading TensorFlow model - this may take a moment...")
        try:
            # Load model
            model_path = os.path.join(self.model_dir, 'ddos_model.h5')
            self.model = tf.keras.models.load_model(model_path)
            
            # Load preprocessing artifacts
            self.scaler = joblib.load(os.path.join(self.model_dir, 'scaler.pkl'))
            self.label_encoder = joblib.load(os.path.join(self.model_dir, 'label_encoder.pkl'))
            self.feature_columns = joblib.load(os.path.join(self.model_dir, 'feature_columns.pkl'))
            
            logger.info("Model and artifacts loaded successfully")
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            logger.warning("Using default model configuration")
            
            # If loading fails, set up with defaults
            self.feature_columns = [
                'Source IP', 'Destination IP', 'Protocol',
                'Total Length of Fwd Packets', 'Fwd Packet Length Min',
                'Bwd IAT Mean', 'Flow IAT Min', 'Init_Win_bytes_forward',
                'Init_Win_bytes_backward', 'ACK Flag Count', 'SYN Flag Count',
                'FIN Flag Count', 'Flow Packets/s', 'Flow Bytes/s'
            ]
    
    def preprocess_data(self, traffic_data): 
        # Ensure dataframe format
        if not isinstance(traffic_data, pd.DataFrame):
            traffic_data = pd.DataFrame([traffic_data])
            
        ip_cols = [col for col in traffic_data.columns if 'ip' in col.lower()]
        for col in ip_cols:
            if col in traffic_data.columns: 
                traffic_data[col] = traffic_data[col].apply(
                    lambda x: int(hash(str(x)) % 1000000)
                )     
        missing_cols = set(self.feature_columns) - set(traffic_data.columns)
        for col in missing_cols:
            traffic_data[col] = 0  
            
        # Select only the features used by the model
        traffic_data = traffic_data[self.feature_columns]
        
        # Apply scaling if scaler is available
        if self.scaler:
            try:
                traffic_data = pd.DataFrame(
                    self.scaler.transform(traffic_data),
                    columns=self.feature_columns
                )
            except Exception as e:
                logger.warning(f"Scaling error: {str(e)}. Using unscaled data.")
        
        return traffic_data
    
    def predict(self, traffic_data): 
        if self.model is None:
                raise RuntimeError("Model not loaded")
            
        
        current_time = datetime.now()
         
        processed_data = self.preprocess_data(traffic_data)
         
        for _, row in processed_data.iterrows():
            self.window.append(row.values)
             
        if len(self.window) < self.sequence_length:
            return {"prediction": "Insufficient data", "probability": 0.0, "status": "Insufficient data"}
            
        # Create sequence for LSTM input
        sequence = np.array([list(self.window)])
        
        # Make prediction
        pred_probability = float(self.model.predict(sequence,verbose = 0)[0][0])
        pred_label = "DDoS Attack" if pred_probability > 0.6 else "Normal Traffic"
        status = "DDoS Attack Detected" if pred_probability > 0.6 else "Normal Traffic"
        
        # Store prediction in history
        self.history.append({
                    "timestamp": pd.Timestamp.now(),
                    "prediction": pred_label,
                    "probability": pred_probability,
                    "status": status
                        })
        
        return {
        "prediction": pred_label,
        "probability": pred_probability,
        "status": status
                }
        
    def get_history(self):
        """Return recent prediction history"""
        return list(self.history)