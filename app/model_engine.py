# app/model_engine.py

import os
import math
import joblib
import pandas as pd
from app.config import Config


class FraudDetectionEngine:
  def __init__(self):
    self.model = None
    self.feature_columns = [
      'device_student_count_24h',
      'page_dwell_time_seconds',
      'is_high_risk_asn',
      'failed_attempts_1h',
      'session_hardware_mismatch',
      'is_off_peak_hour'
    ]
    self.load_model()

  def load_model(self) -> bool:
    """Loads the trained model from Config.MODEL_PATH."""
    if os.path.exists(Config.MODEL_PATH):
      try:
        self.model = joblib.load(Config.MODEL_PATH)
        return True
      except Exception as e:
        print(f"Failed to load model binary: {e}")
        return False
    return False

  def _safe_float(self, value, default: float, min_val: float = None, max_val: float = None) -> float:
    """Casts to float; returns default for invalid values; clamps to optional bounds."""
    if value is None or value == "":
      return default
    try:
      val = float(value)
      # Reject NaN or Infinity
      if math.isnan(val) or math.isinf(val):
        return default
      # Clamp to optional bounds if provided
      if min_val is not None and val < min_val:
        val = min_val
      if max_val is not None and val > max_val:
        val = max_val
      return val
    except (ValueError, TypeError):
      return default

  def _safe_int(self, value, default: int, min_val: int = None, max_val: int = None) -> int:
    """Casts to int via _safe_float with optional bounds."""
    val = self._safe_float(value, float(default), float(min_val) if min_val is not None else None, float(max_val) if max_val is not None else None)
    return int(val)

  def extract_features(self, payload: dict) -> list:
    """Extracts the 6 feature values"""
    device_count = self._safe_int(payload.get('device_student_count_24h'), default=0, min_val=0)

    dwell_time = self._safe_float(payload.get('page_dwell_time_seconds'), default=0, min_val=0)

    # Binary flags must be 0 or 1
    high_risk_asn = 1 if str(payload.get('is_high_risk_asn', 0)).lower() in ['1', 'true', 'yes'] else 0

    failed_attempts = self._safe_int(payload.get('failed_attempts_1h'), default=0, min_val=0)

    hardware_mismatch = 1 if str(payload.get('session_hardware_mismatch', 0)).lower() in ['1', 'true', 'yes'] else 0

    off_peak = 1 if str(payload.get('is_off_peak_hour', 0)).lower() in ['1', 'true', 'yes'] else 0

    return [device_count, dwell_time, high_risk_asn, failed_attempts, hardware_mismatch, off_peak]

  def _generate_explanation(self, features: list, prediction: int) -> str:
    """Reports the tree path the transaction took through the fitted model."""
    verdict_text = "FRAUDULENT" if prediction == 1 else "LEGITIMATE"

    labels = {
      'device_student_count_24h': 'Distinct students on device (24h)',
      'page_dwell_time_seconds': 'Page dwell time',
      'is_high_risk_asn': 'ISP not in MTN/Glo/Airtel/9Mobile/Etisalat flag',
      'failed_attempts_1h': 'Failed attempts in last hour',
      'session_hardware_mismatch': 'Device fingerprint mismatch flag',
      'is_off_peak_hour': 'Transaction in off-peak WAT window (1:00-4:30 AM) flag',
    }

    units = {'page_dwell_time_seconds': 's'}

    try:
      tree = self.model.tree_
      node = 0
      lines = []
      step = 1

      while tree.children_left[node] != tree.children_right[node]:
        idx = tree.feature[node]
        threshold = float(tree.threshold[node])
        actual = features[idx]
        col = self.feature_columns[idx]
        unit = units.get(col, '')

        if actual <= threshold:
          branch = 'left'
          next_node = tree.children_left[node]
        else:
          branch = 'right'
          next_node = tree.children_right[node]

        lines.append(
          f"{step}. {labels[col]} = {actual}{unit} "
          f"(threshold {threshold:.3f}{unit}, {branch})"
        )

        node = next_node
        step += 1

      if not lines:
        return f"Model classified transaction as {verdict_text}. Tree has no split nodes."

      return (
        "Tree path (root → leaf):\n"
        + "\n".join(lines)
        + f"\nLeaf {node} → {verdict_text}"
      )
    except Exception as e:
      return f"Model classified transaction as {verdict_text}. Could not extract tree path: {e}"

  def predict(self, payload: dict) -> dict:
    """Predicts verdict and returns verdict, confidence, and explanation."""
    if not self.model:
      if not self.load_model():
        return {
          "status": "error",
          "message": "Model not loaded."
        }

    try:
      features = self.extract_features(payload)
      print(f"[ML] Features: {features}")

      feature_df = pd.DataFrame([features], columns=self.feature_columns)

      prediction = int(self.model.predict(feature_df)[0])
      probabilities = self.model.predict_proba(feature_df)[0]
      confidence = float(probabilities[prediction])

      verdict = "FRAUDULENT" if prediction == 1 else "LEGITIMATE"
      explanation = self._generate_explanation(features, prediction)

      return {
        "status": "success",
        "verdict": verdict,
        "prediction": prediction,
        "confidence": round(confidence, 4),
        "explanation": explanation
      }
    except Exception as e:
      return {
        "status": "error",
        "message": str(e)
      }