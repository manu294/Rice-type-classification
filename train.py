import os
import pathlib
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.models import load_model
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import pandas as pd

# 📁 Dataset path
data_dir = pathlib.Path("C:/Users/Hello/Desktop/Rice_Type_Detection/Data/Training/Rice_Image_Dataset")

# 🏷️ Labels
labels_map = {
    'Arborio': 0,
    'Basmati': 1,
    'Ipsala': 2,
    'Jasmine': 3,
    'Karacadag': 4
}

# 🖼️ Load up to 600 images per class
print("🔁 Loading up to 600 images per category...")
X, y = [], []
for label, value in labels_map.items():
    folder = data_dir / label
    images = list(folder.glob("*"))[:600]  # limit to 600
    print(f"📂 {label}: {len(images)} images")
    for img_path in images:
        img_data = cv2.imread(str(img_path))
        if img_data is not None:
            resized = cv2.resize(img_data, (224, 224))
            X.append(resized)
            y.append(value)

X = np.array(X, dtype='float32') / 255.0
y = np.array(y)

print("✅ Finished loading all images.")

# 📊 Split into training and validation
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=42)

# 🧠 Load base model with pretrained weights
print("📦 Loading MobileNetV2 with imagenet weights...")
base_model = MobileNetV2(input_shape=(224, 224, 3), include_top=False, weights='imagenet')
base_model.trainable = True

model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(5, activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# ⏱️ Train with EarlyStopping
print("🚀 Training model...")
early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

history = model.fit(X_train, y_train, epochs=15, validation_data=(X_val, y_val), callbacks=[early_stop])

# 💾 Save model
model.save("Training/rice.h5")
print("✅ Model saved as rice.h5")

# 📈 Plot accuracy/loss
acc = pd.DataFrame({
    'Train Accuracy': history.history['accuracy'],
    'Validation Accuracy': history.history['val_accuracy']
})
acc.plot(title='Accuracy per Epoch')
plt.show()

loss = pd.DataFrame({
    'Train Loss': history.history['loss'],
    'Validation Loss': history.history['val_loss']
})
loss.plot(title='Loss per Epoch')
plt.show()

# 📊 Final classification report
print("📊 Evaluating model...")
y_pred = model.predict(X_val)
y_pred_labels = np.argmax(y_pred, axis=1)

print("\n📋 Classification Report:")
print(classification_report(y_val, y_pred_labels, zero_division=0))
