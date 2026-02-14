import os
import numpy as np
import sqlite3
import json
from datetime import datetime
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.resnet50 import preprocess_input
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

# =====================================================
# CONFIGURATION
# =====================================================
MODEL_PATH = "model/resnet_model.h5"
DATASET_PATH = "dataset"
IMG_SIZE = (224, 224)
BATCH_SIZE = 32

# =====================================================
# LOAD MODEL
# =====================================================
print("Loading trained model...")
model = tf.keras.models.load_model(MODEL_PATH)
print("\nModel loaded successfully.\n")

# =====================================================
# VALIDATION DATA GENERATOR
# =====================================================
datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    validation_split=0.2
)

val_generator = datagen.flow_from_directory(
    DATASET_PATH,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="validation",
    shuffle=False
)

print("Class mapping validation:", val_generator.class_indices)

# =====================================================
# PREDICTION
# =====================================================
print("\nEvaluating model...\n")

pred_probs = model.predict(val_generator, verbose=1)
y_pred = np.argmax(pred_probs, axis=1)
y_true = val_generator.classes

# =====================================================
# METRICS CALCULATION
# =====================================================
accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred, average="macro")
recall = recall_score(y_true, y_pred, average="macro")
f1 = f1_score(y_true, y_pred, average="macro")

print("===== HASIL EVALUASI =====")
print(f"Accuracy        : {accuracy*100:.2f}%")
print(f"Macro Precision : {precision*100:.2f}%")
print(f"Macro Recall    : {recall*100:.2f}%")
print(f"Macro F1-score  : {f1*100:.2f}%")

# =====================================================
# CONFUSION MATRIX
# =====================================================
cm = confusion_matrix(y_true, y_pred)

print("\nConfusion Matrix:")
print(cm)

# =====================================================
# CLASSIFICATION REPORT (DICT VERSION UNTUK DB)
# =====================================================
class_names = list(val_generator.class_indices.keys())

classification_report_dict = classification_report(
    y_true,
    y_pred,
    target_names=class_names,
    output_dict=True  # 🔥 penting agar jadi dictionary
)

print("\nClassification Report:")
print(classification_report(y_true, y_pred, target_names=class_names))

# =====================================================
# SAVE TO DATABASE
# =====================================================
print("\nMenyimpan hasil ke database...")

conn = sqlite3.connect("database.db")
cur = conn.cursor()

cur.execute("""
INSERT INTO model_metrics 
(accuracy, precision, recall, f1_score, created_at, confusion_matrix, classification_report)
VALUES (?, ?, ?, ?, ?, ?, ?)
""", (
    accuracy,
    precision,
    recall,
    f1,
    datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    json.dumps(cm.tolist()),  
    json.dumps(classification_report_dict)  
))

conn.commit()
conn.close()

print("Metrics berhasil disimpan ke database.")
print("\nEvaluasi selesai.\n")
