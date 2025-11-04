"""
AI + DBMS Project: Advanced Image Injury Classifier with DBMS Integration
-------------------------------------------------------------------------

Features:
1. Loads trained model + label map
2. Computes SHA256 hash to avoid duplicate image entries in DB
3. Inserts advanced analytics:
    - top-3 predictions
    - inference time
    - model version tracking
    - user_id storage (future login system ready)
4. Separate feedback storage for human review loop
"""

import numpy as np
import time
import hashlib
import json
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import load_model
import mysql.connector

# ------------------ CONFIG ------------------

MODEL_PATH = "injury_classifier.h5"
LABELS_PATH = "labels.json"
MODEL_VERSION = "v1.0"

DB_CONFIG = {
    "host": "localhost",
    "user": "root",
    "password": "abhra",
    "database": "injury_ai"
}

# Fixed user for now, extend later
USER_ID = 1

# ------------------ LOAD MODEL & LABELS ------------------

with open(LABELS_PATH, "r") as f:
    labels = json.load(f)

model = load_model(MODEL_PATH)

# ------------------ CONNECT DB ------------------

db = mysql.connector.connect(**DB_CONFIG)
cursor = db.cursor()

# ------------------ UTILITIES ------------------

def compute_image_hash(img_path):
    """ Compute SHA256 hash for deduplication """
    with open(img_path, "rb") as img_file:
        return hashlib.sha256(img_file.read()).hexdigest()


def check_duplicate(image_hash):
    cursor.execute("SELECT id FROM predictions WHERE image_hash=%s", (image_hash,))
    return cursor.fetchone()


# ------------------ INFERENCE FUNCTION ------------------

def predict_and_store(img_path):
    image_hash = compute_image_hash(img_path)

    # Deduplication check
    existing = check_duplicate(image_hash)
    if existing:
        print(f"❌ Duplicate image detected. Existing prediction ID: {existing[0]}")
        return

    # Load image
    img = load_img(img_path, target_size=(224,224))
    img_arr = img_to_array(img) / 255.0
    img_arr = np.expand_dims(img_arr, axis=0)

    start = time.time()
    preds = model.predict(img_arr)
    end = time.time()

    inference_time = round((end - start) * 1000, 2)  # ms

    # Sort predictions to get top-3
    top_indices = preds[0].argsort()[-3:][::-1]

    # Extract Top-3
    top3 = [(labels[str(i)], float(preds[0][i])) for i in top_indices]

    # DB Insert
    sql = """
    INSERT INTO predictions (
        user_id, image_path, image_hash,
        top1_label, top1_conf,
        top2_label, top2_conf,
        top3_label, top3_conf,
        model_version, inference_time_ms
    ) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
    """

    values = (
        USER_ID, img_path, image_hash,
        top3[0][0], top3[0][1],
        top3[1][0], top3[1][1],
        top3[2][0], top3[2][1],
        MODEL_VERSION, inference_time
    )

    cursor.execute(sql, values)
    db.commit()

    print("\n✅ Prediction Stored Successfully")
    print(f"Top-1: {top3[0][0]} ({top3[0][1]:.2f})")
    print(f"Top-2: {top3[1][0]} ({top3[1][1]:.2f})")
    print(f"Top-3: {top3[2][0]} ({top3[2][1]:.2f})")
    print(f"Model: {MODEL_VERSION}, Inference Time: {inference_time} ms")


# ------------------ FEEDBACK INPUT FUNCTION ------------------

def submit_feedback(prediction_id, true_label, user_feedback):
    sql = """INSERT INTO feedback (prediction_id, user_id, correct_label, user_feedback)
             VALUES (%s, %s, %s, %s)"""
    cursor.execute(sql, (prediction_id, USER_ID, true_label, user_feedback))
    db.commit()
    print("✅ Feedback stored")


# ------------------ Example Run ------------------

if __name__ == "__main__":
    predict_and_store("dataset/wounds/fusc_0098.png")
    # submit_feedback(1, "laceration", "correct")
