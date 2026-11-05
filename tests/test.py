import json
import urllib.request
import urllib.error

def run_test(scenario_name, payload):
    url = "http://127.0.0.1:5000/api/predict"
    
    print(f"\n{'='*50}")
    print(f"SCENARIO: {scenario_name}")
    print(f"{'='*50}")
    print("REQUEST PAYLOAD:")
    print(json.dumps(payload, indent=2))
    print("-" * 50)
    
    req = urllib.request.Request(
        url, 
        data=json.dumps(payload).encode('utf-8'), 
        headers={'Content-Type': 'application/json'}
    )
    
    try:
        with urllib.request.urlopen(req) as response:
            res_data = json.loads(response.read().decode('utf-8'))
            print(f"STATUS CODE : {response.getcode()}")
            print("API RESPONSE:")
            print(json.dumps(res_data, indent=2))
    except urllib.error.HTTPError as e:
        print(f"HTTP ERROR  : {e.code}")
        body_bytes = e.read()          # ← READ ONCE
        if not body_bytes:
            print("ERROR BODY  : (empty)")
        else:
            try:
                error_data = json.loads(body_bytes.decode('utf-8'))
                print("ERROR BODY  :")
                print(json.dumps(error_data, indent=2))
            except Exception:
                print("ERROR BODY  :")
                print(body_bytes.decode('utf-8', errors='replace'))
    except urllib.error.URLError as e:
        print(f"ERROR: {e.reason}")
        print("-> Make sure your Flask server is running on another terminal (python app/main.py)")

if __name__ == "__main__":
    # Scenario 1: Normal Student Payment
    legit_payload = {
        "device_student_count_24h": 1,
        "page_dwell_time_seconds": 145.5,
        "is_high_risk_asn": 0,
        "failed_attempts_1h": 0,
        "session_hardware_mismatch": 0,
        "is_off_peak_hour": 0
    }
    
    # Scenario 2: Suspicious Bot/Fraud Attempt
    fraud_payload = {
        "device_student_count_24h": 5,
        "page_dwell_time_seconds": 1.2,
        "is_high_risk_asn": 1,
        "failed_attempts_1h": 4,
        "session_hardware_mismatch": 1,
        "is_off_peak_hour": 1
    }

    print("Testing Fraud Detection API...")
    run_test("Legitimate Student Transaction", legit_payload)
    run_test("High-Risk Fraudulent Transaction", fraud_payload)