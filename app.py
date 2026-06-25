import gradio as gr
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


def preprocess_image(img):
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )

    l = clahe.apply(l)

    lab = cv2.merge((l, a, b))
    img = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

    red = img[:, :, 0]
    red = np.stack([red, red, red], axis=-1)

    red = red.astype("float32") / 255.0

    return red


def predict(image):
    img = preprocess_image(image)
    img = np.expand_dims(img, axis=0)

    pred = model.predict(img, verbose=0)[0][0]

    label = (
        "Glaucoma"
        if pred >= threshold
        else "Non-Glaucoma"
    )

    return label, float(pred)


demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="numpy"),
    outputs=[
        gr.Textbox(label="Prediction"),
        gr.Number(label="Confidence Score")
    ],
    title="Glaucoma Detection AI System",
    description="Upload a retinal fundus image to detect glaucoma."
)

demo.launch()
