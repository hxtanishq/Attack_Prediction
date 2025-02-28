import requests
import time
import numpy as np
import argparse
import logging
from datetime import datetime
import pandas as pd

logging.basicConfig(filename="attack.log", level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

try:
    dataset = pd.read_csv('data/data.csv')   
except FileNotFoundError:
    logger.error("Dataset file 'data.csv' not found. Please provide the correct path.")
    exit()

def send_traffic_from_dataset(url, data_source, duration=60, rps_min=10, rps_max=200):
    start_time = time.time()
    request_count = 0
    num_rows = len(data_source)

    try:
        while time.time() - start_time < duration:
            rps = np.random.randint(rps_min, rps_max)

            for _ in range(rps):
                if num_rows > 0:
                    row_index = np.random.randint(0, num_rows)
                    row_series = data_source.iloc[row_index]   
                    row_series = row_series.drop(labels=[" Label"], errors='ignore') 

                    row_data = row_series.to_dict()  

                    row_data = {k: float(v) if isinstance(v, (np.float32, np.float64)) else
                                int(v) if isinstance(v, (np.int32, np.int64)) else v
                                for k, v in row_data.items()}

                    data = {"traffic_data": [row_data]}

                    try:
                        response = requests.post(url, json=data, timeout=5)
                        request_count += 1

                        if request_count % 100 == 0:
                            logger.info(f"Sent {request_count} requests. Latest response: {response.status_code}")

                    except requests.exceptions.RequestException as e:
                        logger.error(f"Request error: {str(e)}")
                else:
                    logger.warning("No data rows available in the provided data source.")
                    break

            time.sleep(1)

    except KeyboardInterrupt:
        logger.info("Traffic generation interrupted by user")

    logger.info(f"Traffic generation completed. Sent {request_count} requests in {time.time() - start_time:.2f} seconds")
    return request_count


def run_attack_simulation(url, attack_duration=60, normal_duration=10, normal_rps=(10, 30), attack_rps=(100, 500)):
    attack_data = dataset[dataset[' Label'] == 1]  
    normal_data = dataset[dataset[' Label'] == 0]

    logger.info(f"Starting normal traffic phase ({normal_duration}s)")
    normal_requests = send_traffic_from_dataset(url, data_source=normal_data, duration=normal_duration, rps_min=normal_rps[0], rps_max=normal_rps[1])

    logger.info(f"Starting attack traffic phase ({attack_duration}s)")
    attack_requests = send_traffic_from_dataset(url, data_source=attack_data, duration=attack_duration, rps_min=attack_rps[0], rps_max=attack_rps[1])

    logger.info(f"Simulation complete. Normal requests: {normal_requests}, Attack requests: {attack_requests}")

    logger.info(f"Returning to normal traffic phase")
    send_traffic_from_dataset(url, data_source=normal_data, duration=normal_duration, rps_min=normal_rps[0], rps_max=normal_rps[1])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='DDoS Attack Testing Tool')
    parser.add_argument('--url', default="http://127.0.0.1:5000/predict", help='Target URL')
    parser.add_argument('--normal-time', type=int, default=30, help='Duration of normal traffic (seconds)')
    parser.add_argument('--attack-time', type=int, default=30, help='Duration of attack traffic (seconds)')
    parser.add_argument('--normal-rps', type=str, default="1,20", help='Normal RPS range (min,max)')
    parser.add_argument('--attack-rps', type=str, default="100,700", help='Attack RPS range (min,max)')

    args = parser.parse_args()

    normal_rps = tuple(map(int, args.normal_rps.split(',')))
    attack_rps = tuple(map(int, args.attack_rps.split(',')))

    run_attack_simulation(
        args.url,
        args.attack_time,
        args.normal_time,
        normal_rps,
        attack_rps
    )
    
    
