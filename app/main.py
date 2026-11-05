# /app/main.py

import os
import sys

# Ensure the root project directory is in the Python path when executed directly
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from flask import Flask, request, jsonify
from flask_cors import CORS
from app.model_engine import FraudDetectionEngine

app = Flask(__name__)
# Enable CORS so the Next.js frontend (running on port 3000) can make POST requests
CORS(app, resources={r"/api/*": {"origins": "*"}})

engine = FraudDetectionEngine()

@app.route('/api/health', methods=['GET'])
def health_check():
  model_status = "Loaded" if engine.load_model() else "Missing Binary"
  return jsonify({
    "service": "UNIZIK Fraud Detection API",
    "status": "Active",
    "model_status": model_status
  }), 200

@app.route('/api/predict', methods=['POST'])
def predict_transaction():
  if not request.is_json:
    return jsonify({
      "status": "error",
      "message": "Invalid request format. Payload must be Content-Type: application/json."
    }), 400

  payload = request.get_json()
  result = engine.predict(payload)

  if result.get("status") == "error":
    return jsonify(result), 500

  return jsonify(result), 200

if __name__ == '__main__':
  # Run server on localhost (127.0.0.1)
  app.run(host='127.0.0.1', port=5000, debug=True)