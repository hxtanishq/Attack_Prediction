# test_attack.py
import requests
import time
import numpy as np
import argparse
import logging

logging.basicConfig(filename="attack.log",level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def generate_normal_traffic(src_ip=None):
    
    if src_ip is None:
        src_ip = f"192.168.{np.random.randint(1, 255)}.{np.random.randint(1, 255)}"
        
    traffic_data = {
        'Source IP': src_ip,
        'Destination IP':  f"10.0.{np.random.randint(1, 255)}.{np.random.randint(1, 255)}",
        "Protocol": np.random.choice([6, 17, 1]),  # TCP, UDP, ICMP
        "Total Length of Fwd Packets": np.random.randint(60, 1500),
        "Fwd Packet Length Min": np.random.randint(20, 60),
        "Bwd IAT Mean": np.random.uniform(0.01, 0.5),
        "Flow IAT Min": np.random.uniform(0.001, 0.1),
        "Init_Win_bytes_forward": np.random.randint(1024, 65535),
        "Init_Win_bytes_backward": np.random.randint(1024, 65535),
        "ACK Flag Count": np.random.binomial(1, 0.7),
        "SYN Flag Count": np.random.binomial(1, 0.3),
        "FIN Flag Count": np.random.binomial(1, 0.1),
        "Flow Packets/s": np.random.uniform(1, 100),
        "Flow Bytes/s": np.random.uniform(100, 1000)
    }
    
    # Convert all NumPy types to native Python types
    return {k: float(v) if isinstance(v, (np.float32, np.float64)) else 
              int(v) if isinstance(v, (np.int32, np.int64)) else v 
              for k, v in traffic_data.items()}

def generate_attack_traffic(attack_type="http_flood" , src_ip=None):
    if src_ip is None:
        src_ip = f"{np.random.randint(1, 255)}.{np.random.randint(1, 255)}.{np.random.randint(1, 255)}.{np.random.randint(1, 255)}"
    
    base_traffic = generate_normal_traffic(src_ip)
    
    if attack_type == "syn_flood":
         
        base_traffic.update({
            "SYN Flag Count": 1,
            "ACK Flag Count": 0,
            "FIN Flag Count": 0,
            "Flow Packets/s": np.random.uniform(500, 5000),
            "Flow Bytes/s": np.random.uniform(50000, 500000)
        })
    elif attack_type == "http_flood":
        
        base_traffic.update({
            "Protocol": 6,  # TCP
            "ACK Flag Count": 1,
            "SYN Flag Count": 0,
            "Flow Packets/s": np.random.uniform(500, 2000),
            "Flow Bytes/s": np.random.uniform(10000, 100000),
            "Total Length of Fwd Packets": np.random.randint(1000, 10000)
        })
    elif attack_type == "udp_flood":
         
        base_traffic.update({
            "Protocol": 17,  # UDP
            "Flow Packets/s": np.random.uniform(1000, 10000),
            "Flow Bytes/s": np.random.uniform(50000, 500000)
        })
        
    return {k: float(v) if isinstance(v, (np.float32, np.float64)) else 
              int(v) if isinstance(v, (np.int32, np.int64)) else v 
              for k, v in base_traffic.items()}

def send_traffic(url, is_attack=False, attack_type="http_flood", duration=60, rps_min=10, rps_max=100):
   
    start_time = time.time()
    request_count = 0
    
    
    src_ip = f"192.168.{np.random.randint(1, 255)}.{np.random.randint(1, 255)}"
    
    try:
        while time.time() - start_time < duration:
            # Determine how many requests to send in this batch
            rps = np.random.randint(rps_min, rps_max)
            
            for _ in range(rps):
                if is_attack:
                    traffic_data = generate_attack_traffic(attack_type, src_ip)
                else:
                    traffic_data = generate_normal_traffic(src_ip)
                
                data = {"traffic_data": [traffic_data]}
                
                try:
                    response = requests.post(url, json=data, timeout=5)
                    request_count += 1
                    
                    if request_count % 100 == 0:
                        logger.info(f"Sent {request_count} requests. Latest response: {response.status_code}")
                        
                except requests.exceptions.RequestException as e:
                    logger.error(f"Request error: {str(e)}")
                    
            # Sleep to maintain the desired RPS
            time.sleep(0.5)
            
    except KeyboardInterrupt:
        logger.info("Traffic generation interrupted by user")
        
    logger.info(f"Traffic generation completed. Sent {request_count} requests in {time.time() - start_time:.2f} seconds")
    return request_count

def run_attack_simulation(url, attack_type="syn_flood", normal_duration=10, attack_duration=50, 
                        normal_rps=(10, 20), attack_rps=(100, 400)):
     
    logger.info(f"Starting traffic simulation - Normal phase ({normal_duration}s)")
    normal_requests = send_traffic(
        url, 
        is_attack=False, 
        duration=normal_duration,
        rps_min=normal_rps[0],
        rps_max=normal_rps[1]
    )
    
    logger.info(f"Starting attack phase - {attack_type} ({attack_duration}s)")
    attack_requests = send_traffic(
        url, 
        is_attack=True,
        attack_type=attack_type,
        duration=attack_duration,
        rps_min=attack_rps[0],
        rps_max=attack_rps[1]
    )
    
    logger.info(f"Simulation complete. Normal requests: {normal_requests}, Attack requests: {attack_requests}")
    
    # Return to normal traffic
    logger.info(f"Returning to normal traffic")
    send_traffic(
        url, 
        is_attack=False, 
        duration=normal_duration,
        rps_min=normal_rps[0],
        rps_max=normal_rps[1]
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='DDoS Attack Testing Tool')
    parser.add_argument('--url', default="http://127.0.0.1:5000/predict", help='Target URL')
    parser.add_argument('--type', default="syn_flood", choices=["syn_flood", "http_flood", "udp_flood"], 
                       help='Attack type')
    parser.add_argument('--normal-time', type=int, default=30, help='Duration of normal traffic (seconds)')
    parser.add_argument('--attack-time', type=int, default=30, help='Duration of attack traffic (seconds)')
    parser.add_argument('--normal-rps', type=str, default="1,5", help='Normal RPS range (min,max)')
    parser.add_argument('--attack-rps', type=str, default="50,200", help='Attack RPS range (min,max)')
    
    args = parser.parse_args()
     
    normal_rps = tuple(map(int, args.normal_rps.split(',')))
    attack_rps = tuple(map(int, args.attack_rps.split(',')))
    
    run_attack_simulation(
        args.url,
        attack_type=args.type,
        normal_duration=args.normal_time,
        attack_duration=args.attack_time,
        normal_rps=normal_rps,
        attack_rps=attack_rps
    )