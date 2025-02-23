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
        self.prediction_threshold = 0.4
        self.load_model()
        
    def load_model(self):
        logger.info("Loading TensorFlow model - this may take a moment...")
        try:
            model_path = os.path.join(self.model_dir, 'ddos_model.h5')
            self.model = tf.keras.models.load_model(model_path)
            self.scaler = joblib.load(os.path.join(self.model_dir, 'scaler.pkl'))
            self.label_encoder = joblib.load(os.path.join(self.model_dir, 'label_encoder.pkl'))
            self.feature_columns = joblib.load(os.path.join(self.model_dir, 'feature_columns.pkl'))
            
            logger.info("Model and artifacts loaded successfully")
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            logger.warning("Using default model configuration")
            
            # If loading fails, set up with defaults
            self.feature_columns = [
                'Source IP', 'Destination IP', 'Protocol','Total Length of Fwd Packets', 'Fwd Packet Length Min',
                'Bwd IAT Mean', 'Flow IAT Min', 'Init_Win_bytes_forward','Init_Win_bytes_backward', 'ACK Flag Count', 
                'SYN Flag Count','FIN Flag Count', 'Flow Packets/s', 'Flow Bytes/s'
            ]
    
    def preprocess_data(self, data_df ): 

        # if not isinstance(traffic_data, pd.DataFrame):
        #     traffic_data = pd.DataFrame([traffic_data])
            
        # for col in ['Source IP', ' Destination IP']:
        #     if col in traffic_data.columns:
        #         traffic_data[col] = traffic_data[col].astype(str).apply(lambda x: int(hash(x) % 100000))
                    
        # missing_cols = set(self.feature_columns) - set(traffic_data.columns)
        # for col in missing_cols:
        #     traffic_data[col] = 0  
             
        # traffic_data = traffic_data[self.feature_columns]
        # traffic_data.replace([np.inf, -np.inf], np.nan, inplace=True)
        # traffic_data.dropna(inplace=True)
            
        # if self.scaler:
        #     try:
        #         traffic_data = pd.DataFrame(
        #             self.scaler.transform(traffic_data),
        #             columns=self.feature_columns
        #         )
        #     except Exception as e:
        #         logger.warning(f"Scaling error: {str(e)}. Using unscaled data.")
        
        df = data_df.copy() 
        drop_cols = ['Timestamp', 'Fwd Header Length']
        df.drop(columns=[col for col in drop_cols if col in df.columns], inplace=True)
         
        for col in ['Source IP', 'Destination IP']:
            if col in df.columns:
                df[col] = df[col].astype(str).apply(lambda x: int(hash(x) % 100000))  
        
        df = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
        
        missing_cols = set(self.feature_columns) - set(df.columns)
        for col in missing_cols:
            df[col] = 0
         
        df = df[self.feature_columns]
         
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df.fillna(0, inplace=True)
         
        if self.scaler:
            try:
                scaled_data = self.scaler.transform(df)
                df = pd.DataFrame(scaled_data, columns=self.feature_columns)
            except Exception as e:
                logger.error(f"Scaling error: {str(e)}")
                raise
        
        return df

    def predict(self, traffic_data): 
        if self.model is None:
                raise RuntimeError("Model not loaded")
        current_time = datetime.now()         
        processed_data = self.preprocess_data(data_df = traffic_data)
         
        for _, row in processed_data.iterrows():
            self.window.append(row.values)
             
        if len(self.window) < self.sequence_length:
            return {"prediction": "Insufficient data", "probability": 0.0, "status": "Insufficient data"}
             
        
        sequence = np.array([list(self.window)])
        adjusted_prob = float(self.model.predict(sequence, verbose=0)[0][0])
        
        # adjusted_prob = pred_probability
         
        pred_label = "DDoS Attack" if adjusted_prob > self.prediction_threshold else "Normal Traffic"
        status = "DDoS Attack Detected" if adjusted_prob > self.prediction_threshold else "Normal Traffic"
         
        self.history.append({
                        "timestamp": current_time,
                        "prediction": pred_label,
                        "probability": adjusted_prob,
                        "status": status
                        })
        
        return {
                "prediction": pred_label,
                "probability": adjusted_prob,
                "status": status
                }
        
    def get_history(self):         
        return list(self.history)
    
    