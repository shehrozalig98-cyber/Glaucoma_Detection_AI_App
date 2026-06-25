import gradio as gr
import tensorflow as tf
import numpy as np
import cv2

IMG_SIZE = 224

# Load model
MODEL_PATH = "glaucoma_model.h5"
model = tf.keras.models.load_model(MODEL_PATH)


def preprocess(img):
    img = np.array(img)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img.astype("float32") / 255.0
    img = np.expand_dims(img, axis=0)
    return img


def predict(image):
    processed = preprocess(image)

    pred = model.predict(processed, verbose=0)[0][0]

    label = "Glaucoma Detected" if pred > 0.5 else "No Glaucoma"

    confidence = round(float(pred) * 100, 2)

    return label, confidence


interface = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="numpy", label="Upload Retina Image"),
    outputs=[
        gr.Textbox(label="Diagnosis"),
        gr.Number(label="Confidence (%)")
    ],
    title="Glaucoma Detection AI",
    description="Upload a retinal fundus image for glaucoma prediction."
)

interface.launch()
