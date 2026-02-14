import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing import image
from sklearn.preprocessing import normalize
from tqdm import tqdm

# ================= CONFIG =================
DATASET_PATH = "dataset"
MODEL_PATH = "model/resnet_model.h5"
FEATURE_SAVE_DIR = "features"
IMG_SIZE = 224

os.makedirs(FEATURE_SAVE_DIR, exist_ok=True)

# ================= LOAD MODEL =================
print("Loading model...")
model = load_model(MODEL_PATH)

# Ambil layer embedding (Dense 256)
embedding_model = Model(
    inputs=model.input,
    outputs=model.get_layer("embedding").output
)

print("Embedding model ready.")

# ================= STORAGE =================
features = []
image_paths = []
categories = []
tubuh_list = []

# ================= EXTRACT LOOP =================
print("Extracting features...")

for category in os.listdir(DATASET_PATH):
    category_path = os.path.join(DATASET_PATH, category)

    if not os.path.isdir(category_path):
        continue

    for tubuh in os.listdir(category_path):
        tubuh_path = os.path.join(category_path, tubuh)

        if not os.path.isdir(tubuh_path):
            continue

        for file in tqdm(os.listdir(tubuh_path)):
            img_path = os.path.join(tubuh_path, file)

            try:
                img = image.load_img(img_path, target_size=(IMG_SIZE, IMG_SIZE))
                img_array = image.img_to_array(img)
                img_array = np.expand_dims(img_array, axis=0)
                img_array = preprocess_input(img_array)

                embedding = embedding_model.predict(img_array, verbose=0)[0]

                # 🔥 L2 NORMALIZATION (WAJIB ILMIAH)
                embedding = normalize([embedding])[0]

                features.append(embedding)
                image_paths.append(img_path.replace("\\", "/"))
                categories.append(category)
                tubuh_list.append(tubuh)

            except Exception as e:
                print("Error:", img_path, e)

# ================= SAVE =================
features = np.array(features)

np.save(os.path.join(FEATURE_SAVE_DIR, "features.npy"), features)
np.save(os.path.join(FEATURE_SAVE_DIR, "image_paths.npy"), np.array(image_paths))
np.save(os.path.join(FEATURE_SAVE_DIR, "categories.npy"), np.array(categories))
np.save(os.path.join(FEATURE_SAVE_DIR, "tubuh.npy"), np.array(tubuh_list))

print("\nFeature extraction selesai!")
print("Total images processed:", len(features))
print("Features shape:", features.shape)
