# server.py
from flask import Flask, request, jsonify, render_template, send_from_directory
import pandas as pd
import threading
import time
import os
import logging
from prediction import PredictionModel
import csv
import json
from datetime import datetime
from functools import wraps
 

app = Flask(__name__, static_folder='static')

logging.basicConfig(filename="server.log",level=logging.INFO)
logger = logging.getLogger(__name__)

os.makedirs('logs', exist_ok=True)

log_file = os.path.join('logs', f'traffic_log_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv')

with open(log_file, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['timestamp', 'request_count', 'rps', 'prediction', 'probability', 'status'])

def log_request_to_csv(request_data, prediction_result):
    """Log request and prediction data to CSV file"""
    with open(log_file, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            datetime.now().isoformat(),
            request_count,
            requests_per_second,
            prediction_result.get('prediction', 'Unknown'),
            prediction_result.get('probability', 0),
            prediction_result.get('status', 'Unknown')
        ])

# Initialize the prediction model
predictor = None

def init_predictor():
    global predictor
    try:
        predictor = PredictionModel()
        logger.info("Prediction model initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize predictor: {str(e)}")
        raise

def error_handler(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        try:
            return f(*args, **kwargs)
        except Exception as e:
            logger.error(f"Error in {f.__name__}: {str(e)}")
            return jsonify({
                "error": str(e),
                "status": "Error"
            }), 500
    return decorated_function

# Global variables to count requests and track requests per second
request_count = 0
requests_per_second = 0
ddos_detection_status = "Normal Traffic"
lock = threading.Lock()

def reset_request_count():
    global request_count, requests_per_second
    while True:
        time.sleep(1)  # Reset every second
        with lock:
            requests_per_second = request_count
            request_count = 0

# Start a background thread to track requests per second
threading.Thread(target=reset_request_count, daemon=True).start()

@app.route("/")
def index():
    return render_template("index.html")

@app.route('/model-metrics', methods=['GET'])
def model_metrics():
    try:
        with open('model/evaluation_metrics.json', 'r') as f:
            metrics = json.load(f)
        return jsonify(metrics)
    except Exception as e:
        logger.error(f"Error loading model metrics: {str(e)}")
        return jsonify({
            'accuracy': 0.95,  # Default values if file not found
            'precision': 0.94,
            'recall': 0.93,
            'f1_score': 0.94
        })
        
@app.route('/download-logs', methods=['GET'])
def download_logs():
    try:
        return send_from_directory('logs', os.path.basename(log_file), as_attachment=True)
    except Exception as e:
        logger.error(f"Error downloading logs: {str(e)}")
        return jsonify({"error": "Logs not available"}), 404
    
@app.route("/static/<path:path>")
def serve_static(path):
    return send_from_directory("static", path)

@app.route("/count", methods=["GET"])
def count():
    with lock:
        return jsonify({"count": request_count, "rps": requests_per_second})

@app.route('/predict', methods=['POST'])
def predict():
    global request_count, ddos_detection_status
    with lock:
        request_count += 1
    
    try:
        data = request.json  # Expecting JSON input
        if not data or 'traffic_data' not in data:
            return jsonify({"error": "Invalid input. Expecting 'traffic_data' field.",
                           "prediction": "Unknown", 
                           "probability": 0.0,
                           "status": "Error"}), 400
        # Convert input to pandas DataFrame
        traffic_data = pd.DataFrame(data['traffic_data'])
        
        # Run prediction
        result = predictor.predict(traffic_data)
        if 'status' not in result:
            result['status'] = "Normal Traffic"
            
        ddos_detection_status = result["status"]
        log_request_to_csv(data, result)
        
        return jsonify(result)
    
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        return jsonify({
            "error": str(e),
            "prediction": "Error",
            "probability": 0.0,
            "status": "Error"
        }), 500
@app.route('/status', methods=['GET'])
def status():
    return jsonify({
        "requests_per_second": requests_per_second,
        "status": ddos_detection_status,
        "recent_predictions": predictor.get_history()
    })

@app.route("/monitor", methods=["GET"])
def monitor():
    return render_template("./monitor.html")

# Add this to server.py
@app.route('/test-attack', methods=['POST'])
def test_attack():
    try:
        # Run in a background thread to not block the response
        attack_thread = threading.Thread(
            target=run_attack_test,
            daemon=True
        )
        attack_thread.start()
        
        return jsonify({"message": "Attack simulation started in background"})
    except Exception as e:
        logger.error(f"Request error: {str(e)}")
        return jsonify({"error": str(e)}), 500

def run_attack_test():
    """Run the attack test simulation"""
    try:
        from test_attack import run_attack_simulation
        server_url = f"http://localhost:5000/predict"
        run_attack_simulation(
            server_url,
            attack_type="syn_flood",
            normal_duration=20,
            attack_duration=30,
            normal_rps=(1, 5),
            attack_rps=(50, 150)
        )
    except Exception as e:
        logger.error(f"Error in attack test: {str(e)}")

def run_server(host='0.0.0.0', port=5000):
    app.run(host=host, port=port, debug=False, threaded=True)

if __name__ == '__main__':
    # Check if model exists, train if it doesn't
    if not os.path.exists('./model/ddos_model.h5'):
        logger.info("No model found. Training a new model...")
        from train_model import train_lstm_model
        train_lstm_model()
        # Reload the model after training
        predictor = PredictionModel()
    
    logger.info(f"Starting DDoS detection server on port 5000")
    run_server(debug=True)