# train_model.py
import numpy as np
import pandas as pd
import joblib
import os,json
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tensorflow.keras import Sequential
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.layers import Dense, LSTM, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import logging
from dataloader import load_cicddos2019

# Configure logging
logging.basicConfig(
    filename='train_model.log',
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Set TensorFlow to grow GPU memory usage as needed
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        logger.error(f"GPU setup error: {str(e)}")

def prepare_data(data_df=None, data_path=None):
    """Prepare and preprocess the dataset for training"""
    if data_df is None and data_path:
        data_df = load_cicddos2019(data_path)
    
    if data_df is None:
        logger.warning("No data provided. Generating synthetic data.")
        return generate_synthetic_data()
    
    logger.info(f"Preprocessing dataset with {len(data_df)} records")
    
    # Handle timestamp column if it exists
    timestamp_col = None
    for col in data_df.columns:
        if 'time' in col.lower() or 'timestamp' in col.lower():
            timestamp_col = col
            data_df[timestamp_col] = pd.to_datetime(data_df[timestamp_col], errors='coerce')
            break
    
    # Drop unnecessary columns
    cols_to_drop = ['Flow ID'] if 'Flow ID' in data_df.columns else []
    # Add other unnecessary columns to drop
    data_df.drop(columns=cols_to_drop, errors='ignore', inplace=True)
    
    # Handle constant and NaN columns
    constant_cols = [col for col in data_df.columns if data_df[col].nunique() <= 1]
    data_df.drop(columns=constant_cols, inplace=True)
    data_df = data_df.replace([np.inf, -np.inf], np.nan)
    data_df = data_df.dropna(axis=1)
    
    # Find label column
    label_col = None
    for col in data_df.columns:
        if 'label' in col.lower() or 'attack' in col.lower() or 'class' in col.lower():
            label_col = col
            break
    
    if not label_col:
        logger.error("No label column found in dataset")
        raise ValueError("Could not identify the label column in the dataset")
    
    # Encode labels
    label_encoder = LabelEncoder()
    data_df[label_col] = label_encoder.fit_transform(data_df[label_col])
    
    # Find and encode IP columns
    ip_cols = [col for col in data_df.columns if 'ip' in col.lower() or 'addr' in col.lower()]
    if ip_cols:
        ip_encoder = LabelEncoder()
        all_ips = pd.concat([data_df[col] for col in ip_cols])
        ip_encoder.fit(all_ips)
        for col in ip_cols:
            data_df[col] = ip_encoder.transform(data_df[col])
    
    # Set timestamp as index if available
    if timestamp_col:
        data_df.set_index(timestamp_col, inplace=True)
    
    # Normalize numerical features
    numeric_cols = data_df.select_dtypes(include=['float64', 'int64']).columns
    numeric_cols = [col for col in numeric_cols if col != label_col]
    
    scaler = StandardScaler()
    data_df[numeric_cols] = scaler.fit_transform(data_df[numeric_cols])
    
    # Save feature columns for future reference
    feature_cols = list(data_df.columns)
    if label_col in feature_cols:
        feature_cols.remove(label_col)
    
    return data_df, scaler, label_encoder, feature_cols, label_col

def generate_synthetic_data():
    """Generate synthetic DDoS traffic data for demo purposes"""
    logger.info("Generating synthetic data for demo purposes")
    
    timestamps = pd.date_range(start='2023-01-01', periods=10000, freq='1S')
    np.random.seed(42)
    n_samples = len(timestamps)
    
    data = {
        'Timestamp': timestamps,
        'Source IP': np.random.randint(0, 1000, n_samples),
        'Destination IP': np.random.randint(0, 100, n_samples),
        'Protocol': np.random.randint(0, 5, n_samples),
        'Total Length of Fwd Packets': np.random.exponential(500, n_samples),
        'Fwd Packet Length Min': np.random.exponential(100, n_samples),
        'Bwd IAT Mean': np.random.exponential(0.01, n_samples),
        'Flow IAT Min': np.random.exponential(0.001, n_samples),
        'Init_Win_bytes_forward': np.random.normal(1000, 200, n_samples),
        'Init_Win_bytes_backward': np.random.normal(1000, 200, n_samples),
        'ACK Flag Count': np.random.binomial(1, 0.7, n_samples),
        'SYN Flag Count': np.random.binomial(1, 0.3, n_samples),
        'FIN Flag Count': np.random.binomial(1, 0.1, n_samples),
        'Flow Packets/s': np.random.exponential(10, n_samples),
        'Flow Bytes/s': np.random.exponential(5000, n_samples)
    }
    
    # Generate labels (0 for normal, 1 for attack)
    labels = np.zeros(n_samples)
    
    attack_times = [
        (1000, 1500),   # First attack period
        (3000, 3500),   # Second attack period
        (7000, 7500)    # Third attack period
    ]
    
    for start, end in attack_times:
        labels[start:end] = 1
        
        # Modify attack traffic patterns
        data['Flow Packets/s'][start:end] *= 10
        data['Flow Bytes/s'][start:end] *= 5
        data['SYN Flag Count'][start:end] = np.random.binomial(1, 0.9, end-start)
    
    data['Label'] = labels
    
    df = pd.DataFrame(data)
    df.set_index('Timestamp', inplace=True)
    
    scaler = StandardScaler()
    feature_cols = list(df.columns)
    feature_cols.remove('Label')
    scaler.fit(df[feature_cols])
    
    label_encoder = LabelEncoder()
    label_encoder.fit([0, 1])
    
    return df, scaler, label_encoder, feature_cols, 'Label'

def create_sequences(data, label_col, seq_length=10):
    """Create input sequences for LSTM training"""
    X, y = [], []
    
    features = data.drop(label_col, axis=1)
    target = data[label_col]
    
    features_array = features.values
    target_array = target.values
    
    for i in range(len(data) - seq_length):
        X.append(features_array[i:i + seq_length])
        y.append(target_array[i + seq_length])
    
    return np.array(X), np.array(y)

def evaluate_model(model, X, y):
    """Evaluate model performance and generate metrics"""
    # Make predictions
    y_pred_prob = model.predict(X)
    y_pred = (y_pred_prob > 0.5).astype(int)
    
    # Calculate metrics
    cm = confusion_matrix(y, y_pred)
    report = classification_report(y, y_pred, output_dict=True)
    
    # Visualize confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Normal', 'Attack'],
                yticklabels=['Normal', 'Attack'])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.savefig('model/confusion_matrix.png')
    
    # Log metrics
    metrics = {
        'accuracy': report['accuracy'],
        'precision': report['weighted avg']['precision'],
        'recall': report['weighted avg']['recall'],
        'f1-score': report['weighted avg']['f1-score'],
        'confusion_matrix': cm.tolist()
    }
    
    # Save metrics to file
    with open('model/evaluation_metrics.json', 'w') as f:
        json.dump(metrics, f, indent=4)
    
    logger.info(f"Model evaluation completed. Accuracy: {metrics['accuracy']:.4f}")
    return metrics


def train_lstm_model(data_path=None, data_df=None, seq_length=10, epochs=10, batch_size=64):
    """Train an LSTM model for DDoS detection"""
    model_dir = os.path.join(os.getcwd(), 'model')
    os.makedirs(model_dir, exist_ok=True)
    
    # Prepare the data
    data_df, scaler, label_encoder, feature_cols, label_col = prepare_data(data_df, data_path)
    logger.info(f"Data prepared: {data_df.shape[0]} records, {len(feature_cols)} features")
    
    # Create sequences for LSTM
    X, y = create_sequences(data_df, label_col, seq_length)
    logger.info(f"Sequences created: X shape={X.shape}, y shape={y.shape}")
    
    # Build the LSTM model
    model = Sequential([
        LSTM(128, return_sequences=True, input_shape=(X.shape[1], X.shape[2])),
        Dropout(0.2),
        LSTM(64, return_sequences=False),
        Dropout(0.1),
        Dense(32, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.AUC(), tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
    )
    
    # Set up callbacks for training
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True),
        ModelCheckpoint(os.path.join(model_dir, 'best_model.h5'), save_best_only=True)
    ]
    
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    # Train model
    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.2,
        callbacks=callbacks,
        verbose=1
    )
    
    # Evaluate model
    logger.info("Evaluating model performance...")
    metrics = evaluate_model(model, X_test, y_test)
    
    # Save model and artifacts
    model_path = os.path.join(model_dir, 'ddos_model.h5')
    scaler_path = os.path.join(model_dir, 'scaler.pkl')
    label_encoder_path = os.path.join(model_dir, 'label_encoder.pkl')
    feature_cols_path = os.path.join(model_dir, 'feature_columns.pkl')
    
    model.save(model_path)
    joblib.dump(scaler, scaler_path)
    joblib.dump(label_encoder, label_encoder_path)
    joblib.dump(feature_cols, feature_cols_path)
    
    logger.info(f"Model and artifacts saved in {model_dir}")
    
    return model, scaler, label_encoder, feature_cols, history

if __name__ == "__main__":
    # Look for dataset in standard locations
    data_path = './data'
    train_lstm_model(data_path=data_path, epochs=5, batch_size=64)