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
ROOT          = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CNN_PATH  = os.path.join(ROOT, "models", "CNN_model.h5")
TRANSFER_PATH = os.path.join(ROOT, "models", "transfer_model.h5")
IMG_SIZE      = (224, 224)

# Class index → label mapping.
# Keras flow_from_directory sorts classes alphabetically:
# 0 = fresh, 1 = rotten
CLASS_NAMES = {0: "Fresh", 1: "Rotten"}
# ──────────────────────────────────────────────────────────────────────

app = Flask(__name__,
            template_folder="templates",
            static_folder="static")

# Load models once at startup
print(f"Loading CNN model from: {CNN_PATH}")
cnn_model = load_model(CNN_PATH)
print("CNN model loaded successfully.")

print(f"Loading transfer model from: {TRANSFER_PATH}")
transfer_model = load_model(TRANSFER_PATH)
print("Transfer model loaded successfully.")


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
    Returns JSON with predictions from both models.
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

    # Run inference — CNN Model
    prob_c = float(cnn_model.predict(img_array, verbose=0)[0][0])
    idx_c  = 1 if prob_c >= 0.5 else 0
    res_c  = {
        "label": CLASS_NAMES[idx_c],
        "confidence": round((prob_c if idx_c == 1 else 1.0 - prob_c) * 100, 2)
    }

    # Run inference — Transfer Model
    prob_t = float(transfer_model.predict(img_array, verbose=0)[0][0])
    idx_t  = 1 if prob_t >= 0.5 else 0
    res_t  = {
        "label": CLASS_NAMES[idx_t],
        "confidence": round((prob_t if idx_t == 1 else 1.0 - prob_t) * 100, 2)
    }

    return jsonify({
        "CNN": res_c,
        "transfer": res_t
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
