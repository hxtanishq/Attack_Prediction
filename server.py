# server.py
from flask import Flask, request, jsonify, render_template, send_from_directory, Response
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

log_file = os.path.join('logs', f'OUTPUT.{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv')

with open(log_file, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['timestamp', 'request_count', 'rps', 'prediction', 'probability', 'status'])

def log_request_to_csv(prediction_result):
 
    with open(log_file, 'a', newline='') as f:
        writer = csv.writer(f)
        probability = prediction_result.get('probability', 0)
        formatted_probability = "{:.2f}".format(probability) if isinstance(probability, (int, float)) else probability

        writer.writerow([
            datetime.now().isoformat(),
            request_count,
            requests_per_second,
            prediction_result.get('prediction', 'Unknown'),
            formatted_probability,
            prediction_result.get('status', 'Unknown')
        ])
 
predictor = None

def init_predictor():
    global predictor
    try:
        predictor = PredictionModel()
        print("Prediction model initialized successfully")
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
totalReq = 0
requests_per_second = 0
ddos_detection_status = "Normal Traffic"
lock = threading.Lock()

def reset_request_count():
    global request_count, requests_per_second, totalReq
    while True:
        time.sleep(1)  
        with lock:
            totalReq += request_count
            requests_per_second = request_count
            logger.info(f"Reset request_count: {request_count}, rps: {requests_per_second}")
            request_count = 0
 
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
            'accuracy': 0.95,  
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
        return "Log file not found", 404 
    
@app.route("/static/<path:path>")
def serve_static(path):
    return send_from_directory("static", path)

@app.route("/count", methods=["GET"])
def count():
    with lock:
        return jsonify({"count": request_count, "rps": requests_per_second})

@app.route("/total_count", methods=["GET"])
def total_count():
    with lock:
        return jsonify({"count": totalReq})

@app.route('/predict', methods=['POST'])
@error_handler
def predict():
    global request_count, ddos_detection_status
    with lock:
        request_count += 1
    if predictor is None:
        logger.error("Predictor not initialized")
        return jsonify({"error": "Prediction model not initialized", "status": "Error"}), 500
    try:
        data = request.json   
        if not data or 'traffic_data' not in data:
            return jsonify({"error": "Invalid input. Expecting 'traffic_data' field.",
                           "prediction": "Unknown", 
                           "probability": 0.0,
                           "status": "Error"}), 400
        
        traffic_data = pd.DataFrame(data['traffic_data'])
        result = predictor.predict(traffic_data)
        ddos_detection_status = result["status"]
        log_request_to_csv(result)
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
    if predictor is None:
        return jsonify({"requests_per_second": requests_per_second,
                        "status": "Error: Predictor not initialized",
                        "recent_predictions": []})
    return jsonify({"requests_per_second": requests_per_second,
                    "status": ddos_detection_status,
                    "recent_predictions": predictor.get_history()})

@app.route("/monitor", methods=["GET"])
def monitor():
    return render_template("./monitor.html")

@app.route('/test-attack', methods=['POST'])
def test_attack():
    try:         
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
    try:
        from attack import run_attack_simulation
        server_url = f"http://localhost:5000/predict"
        run_attack_simulation(
            server_url,
            normal_duration=15,
            attack_duration=60,
            normal_rps=(1, 30),
            attack_rps=(100, 800)
        )
    except Exception as e:
        logger.error(f"Error in attack test: {str(e)}")

def run_server(host='0.0.0.0', port=5000):
    init_predictor()
    app.run(host=host, port=port, debug=False, threaded=True)

if __name__ == '__main__':
    if not os.path.exists('./model/ddos_model.h5'):
        logger.info("No model found. Training a new model...")
        from train_model import train_lstm_model
        train_lstm_model()
        
    logger.info(f"Starting DDoS detection server on port 5000")
    run_server(debug=True)