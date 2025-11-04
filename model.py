"""
AI + DBMS Project: Injury Image Classifier - Training Script
-----------------------------------------------------------
This script trains a Convolutional Neural Network (CNN) model 
to classify medical injury images (e.g., burns, wounds, cuts, etc.).

Steps:
1. Load dataset from folders (structured as /class_name/images).
2. Preprocess images (resize, normalize).
3. Build a CNN model .
4. Train and validate the model.
5. Evaluate model with comprehensive metrics.
6. Save trained model and class labels for later use.
"""

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
import json
import numpy as np
from sklearn.metrics import (
    accuracy_score, 
    precision_score, 
    recall_score, 
    f1_score, 
    roc_auc_score, 
    confusion_matrix
)

# =====================
# Data Preprocessing
# =====================
datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,  # 80% train, 20% validation
    rotation_range=15,
    zoom_range=0.1,
    horizontal_flip=True
)

train_gen = datagen.flow_from_directory(
    'dataset/',
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    subset='training'
)

val_gen = datagen.flow_from_directory(
    'dataset/',
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    subset='validation',
    shuffle=False  # Important for evaluation: do not shuffle
)

# =====================
# Model Building
# =====================
base_model = MobileNetV2(weights="imagenet", include_top=False, input_shape=(224,224,3))

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation="relu")(x)
predictions = Dense(train_gen.num_classes, activation="softmax")(x)

model = Model(inputs=base_model.input, outputs=predictions)

# Freeze base layers
for layer in base_model.layers:
    layer.trainable = False

model.compile(optimizer=Adam(learning_rate=0.0001), 
              loss="categorical_crossentropy", 
              metrics=["accuracy"])

# =====================
# Training
# =====================
history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=3 
)

# =====================
# Evaluation
# =====================
print("\nEvaluating model on validation data...")

# Reset generator before prediction (optional but good practice)
val_gen.reset()

# Get predictions (probabilities)
# Note: Steps = total_samples / batch_size
y_pred_probs = model.predict(val_gen, steps=int(np.ceil(val_gen.samples / val_gen.batch_size)))

# Get predicted classes (indices)
y_pred_classes = np.argmax(y_pred_probs, axis=1)

# Get true classes
y_true_classes = val_gen.classes

# Get true classes in one-hot format for ROC-AUC
y_true_one_hot = to_categorical(y_true_classes, num_classes=train_gen.num_classes)

# Calculate metrics
# Using 'macro' average for multi-class, which treats all classes equally.
# Added zero_division=0 to handle cases where a class has no predictions.

accuracy = accuracy_score(y_true_classes, y_pred_classes)
precision = precision_score(y_true_classes, y_pred_classes, average='macro', zero_division=0)
recall = recall_score(y_true_classes, y_pred_classes, average='macro', zero_division=0)  # Sensitivity
f1 = f1_score(y_true_classes, y_pred_classes, average='macro', zero_division=0)

# ==================================
# MODIFIED SECTION
# ==================================
# Wrap ROC-AUC calculation in a try-except block to handle cases
# where a class is missing from the validation set.
try:
    roc_auc = roc_auc_score(y_true_one_hot, y_pred_probs, average='macro', multi_class='ovr')
except ValueError as e:
    print(f"\n⚠️  Could not calculate ROC-AUC score: {e}")
    print("    This likely means one class was missing from the validation data.")
    roc_auc = 0.0  # Set to 0.0 as a placeholder
# ==================================
# END OF MODIFIED SECTION
# ==================================


# Calculate Specificity (Macro)
cm = confusion_matrix(y_true_classes, y_pred_classes)
fp = cm.sum(axis=0) - np.diag(cm)
fn = cm.sum(axis=1) - np.diag(cm)
tp = np.diag(cm)
tn = cm.sum() - (fp + fn + tp)

# Specificity = TN / (TN + FP)
# Add a small epsilon (1e-6) to avoid division by zero
specificity_per_class = tn / (tn + fp + 1e-6)
specificity = np.mean(specificity_per_class)


print("\n--- Model Evaluation Metrics ---")
print(f"Total Accuracy:       {accuracy:.4f}")
print(f"Precision (Macro):    {precision:.4f}")
print(f"Recall (Macro):       {recall:.4f}")
print(f"Sensitivity (Macro):  {recall:.4f} (Same as Recall)")
print(f"F1 Score (Macro):     {f1:.4f}")
print(f"Specificity (Macro):  {specificity:.4f}")
print(f"ROC-AUC (Macro OVR):  {roc_auc:.4f}")
print("--------------------------------\n")


# =====================
# Save Model and Labels
# =====================
model.save("injury_classifier.h5")
print("✅ Model saved as injury_classifier.h5")

# After training is done
labels = train_gen.class_indices
labels = {v: k for k, v in labels.items()}  # reverse mapping (0: "burn", 1: "wound"...)

with open("labels.json", "w") as f:
    json.dump(labels, f)
print("✅ Labels saved as labels.json")