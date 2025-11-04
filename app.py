import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import numpy as np
from flask import Flask, request, jsonify, render_template
from PIL import Image
import json
import mysql.connector
import os

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# ---- Load Model + Labels ----
try:
    model = load_model("injury_classifier.h5")
    with open("labels.json", "r") as f:
        labels = json.load(f)
    print("✅ Model & labels loaded.")
except Exception as e:
    print(f"❌ MODEL/JSON LOAD FAILED: {e}")
    model = None
    labels = {}

# ---- MySQL Connection ----
try:
    db = mysql.connector.connect(
        host="localhost",
        user="root",
        password="abhra",
        database="injury_ai"
    )
    cursor = db.cursor()
    print("✅ MySQL Connected.")
except Exception as e:
    print(f"❌ DB CONNECTION FAILED: {e}")
    db = None
    cursor = None

# ---- Image Preprocessing ----
def preprocess_image(img):
    if img.mode != "RGB":
        img = img.convert("RGB")
    img = img.resize((224, 224))
    img = img_to_array(img) / 255.0
    img = np.expand_dims(img, axis=0)
    return img

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        if not model:
            return jsonify({"error": "Model not loaded"}), 500

        file = request.files.get("imageFile")
        if not file:
            return jsonify({"error": "No file input"}), 400

        filename = file.filename
        if filename == "":
            return jsonify({"error": "Empty filename"}), 400

        save_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file.save(save_path)

        img = Image.open(save_path)
        img = preprocess_image(img)

        preds = model.predict(img)
        class_idx = np.argmax(preds)
        confidence = float(np.max(preds))
        label = labels[str(class_idx)]

        # ---- Insert into DB ----
        if db and cursor:
            try:
                sql = "INSERT INTO predictions (injury_type, confidence, image_path) VALUES (%s, %s, %s)"
                cursor.execute(sql, (label, confidence, save_path))
                db.commit()
            except Exception as db_err:
                print(f"❌ DB INSERT ERROR: {db_err}")
        else:
            print("⚠️ DB not connected, skipping insert.")

        return jsonify({
            "injury_type": label,
            "confidence": confidence
        })

    except Exception as e:
        print(f"❌ BACKEND ERROR: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
