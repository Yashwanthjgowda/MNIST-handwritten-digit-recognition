import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import cv2

# ---------------- PAGE CONFIG ---------------- #
st.set_page_config(page_title="MNIST CNN Digit Recognizer", layout="centered")

# ---------------- LOAD CNN MODEL ---------------- #
@st.cache_resource
def load_cnn_model():
    return tf.keras.models.load_model("model/mnist_cnn_model.keras", compile=False)

model = load_cnn_model()

# ---------------- IMAGE PREPROCESSING ---------------- #
def preprocess_image(img):
    """
    Convert input image to MNIST-like format for CNN
    """
    # Convert to grayscale if 3 channels
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Invert colors (white digit on black)
    img = 255 - img

    # Resize to 28x28
    img = cv2.resize(img, (28, 28))

    # Normalize to [0,1]
    img = img / 255.0

    # Reshape for CNN: (1, 28, 28, 1)
    img = img.reshape(1, 28, 28, 1)
    return img

# ---------------- UI ---------------- #
st.title("📷 MNIST Digit Recognition (CNN)")
st.write("Upload an image of a digit (0–9) to predict.")

uploaded = st.file_uploader("Upload digit image (PNG/JPG)", type=["png", "jpg", "jpeg"])
if uploaded:
    image = Image.open(uploaded).convert("L")  # convert to grayscale
    img_array = np.array(image)
    processed = preprocess_image(img_array)

    st.image(processed.reshape(28, 28), caption="Model Input", width=120)

    if st.button("🔍 Predict Digit"):
        prediction = model.predict(processed, verbose=0)
        st.success(f"Predicted Digit: **{np.argmax(prediction)}**")
