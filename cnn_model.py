import json
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam

# ===============================
# KONFIGURASI
# ===============================
DATASET_DIR = r"E:\dataset"
IMG_SIZE = (224, 224)
BATCH_SIZE = 16
EPOCHS = 10
MODEL_NAME = "cnn_smarttailor.h5"

# ===============================
# DATA GENERATOR
# ===============================
datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2
)

train_data = datagen.flow_from_directory(
    DATASET_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="training"
)

val_data = datagen.flow_from_directory(
    DATASET_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="validation"
)

# ===============================
# SIMPAN CLASS INDICES (KUNCI)
# ===============================
with open("class_indices.json", "w") as f:
    json.dump(train_data.class_indices, f, indent=4)

print("Class indices:", train_data.class_indices)

# ===============================
# MODEL CNN
# ===============================
model = Sequential([
    Conv2D(32, (3,3), activation="relu", input_shape=(224,224,3)),
    MaxPooling2D(2,2),

    Conv2D(64, (3,3), activation="relu"),
    MaxPooling2D(2,2),

    Conv2D(128, (3,3), activation="relu"),
    MaxPooling2D(2,2),

    Flatten(),
    Dense(256, activation="relu"),
    Dropout(0.5),
    Dense(train_data.num_classes, activation="softmax")
])

model.compile(
    optimizer=Adam(learning_rate=0.0001),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

# ===============================
# TRAINING
# ===============================
model.fit(
    train_data,
    validation_data=val_data,
    epochs=EPOCHS
)

# ===============================
# SIMPAN MODEL
# ===============================
model.save(MODEL_NAME)
print("Model disimpan:", MODEL_NAME)
