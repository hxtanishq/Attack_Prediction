# main.py
import os
import argparse
import threading
import logging
from server import run_server
from train_model import train_lstm_model

os.makedirs('model', exist_ok=True)
os.makedirs('logs', exist_ok=True)
logging.basicConfig(filename="main.log",level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
logger.info("Starting DDoS detection system...")

def check_model_exists():
    
    model_path = os.path.join('model', 'ddos_model.h5')
    if not os.path.exists(model_path):
        logger.info("No trained model found. Training now...")
        train_lstm_model()
    else:
        logger.info("Found existing model.")

def main(args):
    os.makedirs('model', exist_ok=True)
    os.makedirs('logs', exist_ok=True)
    
    if args.train:
        logger.info("Training model...")
        train_lstm_model(epochs=args.epochs, batch_size=args.batch_size)
    else:
        logger.info("Checking for existing model...")
        check_model_exists()
    
    logger.info(f"Starting DDoS detection server on port {args.port}")
    run_server(host=args.host, port=args.port)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='DDoS Detection System')
    parser.add_argument('--host', default='0.0.0.0', help='Host to run the server on')
    parser.add_argument('--port', type=int, default=5000, help='Port to run the server on')
    parser.add_argument('--train', action='store_true', help='Force training a new model')
    parser.add_argument('--epochs', type=int, default=5, help='Number of epochs for training')
    parser.add_argument('--batch-size', type=int, default=64, help='Batch size for training')
    
    args = parser.parse_args()
    main(args)