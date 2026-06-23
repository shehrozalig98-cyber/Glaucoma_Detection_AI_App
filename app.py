import streamlit as st
import numpy as np
import cv2
import tensorflow as tf
import json

IMG_SIZE = 224

# Load model
model = tf.keras.models.load_model("model/glaucoma_model.h5")

# Load config
with open("model/config.json", "r") as f:
    config = json.load(f)

threshold = config["threshold"]

# Preprocessing (same as yours)
def preprocess_image(img):
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)

    lab = cv2.merge((l, a, b))
    img = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

    red = img[:, :, 0]
    red = np.stack([red, red, red], axis=-1)

    red = red.astype("float32") / 255.0
    return red

# Prediction
def predict(image):
    img = preprocess_image(image)
    img = np.expand_dims(img, axis=0)

    pred = model.predict(img)[0][0]

    label = "Glaucoma" if pred >= threshold else "Non-Glaucoma"

    return pred, label


# UI
st.title("Glaucoma Detection AI System")

uploaded_file = st.file_uploader("Upload Retina Image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    st.image(image, caption="Uploaded Image")

    pred, label = predict(image)

    st.write("Prediction:", label)
    st.write("Confidence Score:", float(pred))