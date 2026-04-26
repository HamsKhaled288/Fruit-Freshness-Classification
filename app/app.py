"""
app.py  —  Fruit Freshness Detection Flask Web App
---------------------------------------------------
Routes:
  GET  /          → renders index.html
  POST /predict   → accepts uploaded image, returns JSON prediction

Usage (from project root):
  python app/app.py
"""

import os
import io
import numpy as np
from flask import Flask, request, jsonify, render_template
from tensorflow.keras.models import load_model
from PIL import Image

# ── Configuration ──────────────────────────────────────────────────────
ROOT       = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(ROOT, "models", "transfer_model.h5")
IMG_SIZE   = (224, 224)

# Class index → label mapping.
# Keras flow_from_directory sorts classes alphabetically:
# 0 = fresh, 1 = rotten
CLASS_NAMES = {0: "Fresh", 1: "Rotten"}
# ──────────────────────────────────────────────────────────────────────

app = Flask(__name__,
            template_folder="templates",
            static_folder="static")

# Load model once at startup
print(f"  Loading model from: {MODEL_PATH}")
model = load_model(MODEL_PATH)
print("  ✔ Model loaded successfully.")


def preprocess_image(file_bytes: bytes) -> np.ndarray:
    """Open image bytes, resize to 224×224, normalize to [0,1], add batch dim."""
    img = Image.open(io.BytesIO(file_bytes)).convert("RGB")
    img = img.resize(IMG_SIZE)
    arr = np.array(img, dtype=np.float32) / 255.0   # normalize
    arr = np.expand_dims(arr, axis=0)                # shape (1, 224, 224, 3)
    return arr


@app.route("/", methods=["GET"])
def index():
    """Serve the main upload page."""
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    """
    Accept a multipart/form-data POST with field 'image'.
    Returns JSON: {"label": "Fresh"|"Rotten", "confidence": float}
    """
    if "image" not in request.files:
        return jsonify({"error": "No image file provided. Use field name 'image'."}), 400

    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "Empty filename received."}), 400

    try:
        file_bytes = file.read()
        img_array  = preprocess_image(file_bytes)
    except Exception as e:
        return jsonify({"error": f"Could not read or process the image: {str(e)}"}), 400

    # Run inference
    prob = float(model.predict(img_array, verbose=0)[0][0])

    # prob ≥ 0.5 → rotten (class index 1), else fresh (class index 0)
    pred_index  = 1 if prob >= 0.5 else 0
    label       = CLASS_NAMES[pred_index]
    confidence  = prob if pred_index == 1 else (1.0 - prob)
    confidence  = round(confidence * 100, 2)   # as percentage

    return jsonify({"label": label, "confidence": confidence})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
