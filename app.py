from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing import image
import numpy as np
import os
import cv2

app = Flask(__name__)
model = load_model("Training/rice.h5")

labels_map = {
    0: "Arborio",
    1: "Basmati",
    2: "Ipsala",
    3: "Jasmine",
    4: "Karacadag"
}

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/details")
def details():
    return render_template("details.html")

@app.route("/predict", methods=["POST"])
def predict():
    if 'file' not in request.files:
        return "No file part"
    file = request.files['file']
    if file.filename == '':
        return "No selected file"
    if file:
        img_path = os.path.join("static", file.filename)
        file.save(img_path)

        img = cv2.imread(img_path)
        img = cv2.resize(img, (224, 224))
        img = img.astype(np.float32) / 255.0
        img = np.expand_dims(img, axis=0)

        pred = model.predict(img)
        class_id = np.argmax(pred)
        label = labels_map[class_id]

        return render_template("results.html", label=label, image_file=file.filename)

if __name__ == "__main__":
    app.run(debug=True)
