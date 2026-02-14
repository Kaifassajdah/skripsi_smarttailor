import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing import image
from sklearn.preprocessing import normalize

# ================= CONFIG =================
MODEL_PATH = "model/resnet_model.h5"
FEATURE_DIR = "features"
IMG_SIZE = 224
TOP_K = 5

# ================= LOAD MODEL =================
print("Loading model...")
model = load_model(MODEL_PATH)

embedding_model = Model(
    inputs=model.input,
    outputs=model.get_layer("embedding").output
)

print("Model loaded.")

# ================= LOAD PRECOMPUTED FEATURES =================
print("Loading feature matrix...")

features = np.load(os.path.join(FEATURE_DIR, "features.npy"))
image_paths = np.load(os.path.join(FEATURE_DIR, "image_paths.npy"))
categories = np.load(os.path.join(FEATURE_DIR, "categories.npy"))
tubuh_list = np.load(os.path.join(FEATURE_DIR, "tubuh.npy"))

print("Feature matrix shape:", features.shape)

# ================= CLASS MAPPING =================
CLASS_LABELS = ['dress', 'pakaian_atasan', 'pakaian_bawahan']


# ==============================================================
# ======================== RECOMMEND ===========================
# ==============================================================

def recommend(img_path, tubuh_user):

    # ===== 1️⃣ LOAD & PREPROCESS IMAGE =====
    img = image.load_img(img_path, target_size=(IMG_SIZE, IMG_SIZE))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    # ===== 2️⃣ PREDICT CATEGORY =====
    preds = model.predict(img_array, verbose=0)[0]
    predicted_class_index = np.argmax(preds)
    category = CLASS_LABELS[predicted_class_index]
    confidence = float(preds[predicted_class_index])

    # ===== 3️⃣ EXTRACT EMBEDDING =====
    query_embedding = embedding_model.predict(img_array, verbose=0)[0]

    # 🔥 L2 NORMALIZATION
    query_embedding = normalize([query_embedding])[0]

    # ===== 4️⃣ FILTER DATASET =====
    mask = (categories == category) & (tubuh_list == tubuh_user)

    filtered_features = features[mask]
    filtered_paths = image_paths[mask]

    if len(filtered_features) == 0:
        return category, confidence, []

    # ===== 5️⃣ COSINE SIMILARITY (FAST MATRIX) =====
    similarities = np.dot(filtered_features, query_embedding)

    # ===== 6️⃣ TOP K =====
    top_indices = np.argsort(similarities)[::-1][:TOP_K]

    results = []
    for idx in top_indices:
        results.append({
            "image": "/" + filtered_paths[idx].replace("\\", "/"),
            "score": float(similarities[idx])
        })

    return category, confidence, results


# ==============================================================
# ======================== TEST MODE ===========================
# ==============================================================

if __name__ == "__main__":
    print("Similarity system ready.")
