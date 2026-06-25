import gradio as gr
import tensorflow as tf
import numpy as np
import cv2
import os

IMG_SIZE = 224

MODEL_PATH = os.path.join("model", "glaucoma_model.h5")
model = tf.keras.models.load_model(MODEL_PATH)

def preprocess(img):
    img = np.array(img)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    return img

def predict(image):
    processed = preprocess(image)
    pred = model.predict(processed)[0][0]

    label = "Glaucoma Detected" if pred > 0.5 else "No Glaucoma"
    return label, float(pred)

interface = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="numpy"),
    outputs=["text", "number"],
    title="Glaucoma Detection AI",
    description="Upload an eye image for glaucoma prediction"
)

interface.launch()
