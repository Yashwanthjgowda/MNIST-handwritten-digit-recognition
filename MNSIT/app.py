import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image, ImageOps
import cv2
from streamlit_drawable_canvas import st_canvas

# ---------------- PAGE CONFIG ---------------- #
st.set_page_config(page_title="MNIST CNN Digit Recognizer", layout="centered")

# ---------------- LOAD CNN MODEL ---------------- #
@st.cache_resource
def load_cnn_model():
    try:
        model = tf.keras.models.load_model("./model/mnist_cnn_model.keras", compile=False)
        return model
    except OSError:
        st.error("❌ Model file not found! Make sure 'mnist_cnn_model.keras' is inside the 'model' folder.")
        return None

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
st.write("Upload an image of a digit (0–9) or draw one below to predict.")

# --- Upload Image ---
uploaded = st.file_uploader("Upload digit image (PNG/JPG)", type=["png", "jpg", "jpeg"])
if uploaded:
    image = Image.open(uploaded).convert("L")  # convert to grayscale
    img_array = np.array(image)
    processed = preprocess_image(img_array)

    st.image(processed.reshape(28, 28), caption="Model Input", width=120)

    if st.button("🔍 Predict Digit from Image"):
        if model:
            prediction = model.predict(processed, verbose=0)
            st.success(f"Predicted Digit: **{np.argmax(prediction)}**")
        else:
            st.warning("Model is not loaded. Cannot predict.")

# --- Draw Digit ---
st.write("Or draw a digit below:")
canvas_result = st_canvas(
    fill_color="#000000",
    stroke_width=15,
    stroke_color="#FFFFFF",
    background_color="#000000",
    width=280,
    height=280,
    drawing_mode="freedraw",
    key="canvas",
)

if st.button("🔍 Predict Digit from Drawing"):
    if canvas_result.image_data is not None:
        drawn_image = cv2.cvtColor(np.array(canvas_result.image_data), cv2.COLOR_RGBA2BGR)
        processed = preprocess_image(drawn_image)
        st.image(processed.reshape(28, 28), caption="Processed Digit", width=120)

        if model:
            prediction = model.predict(processed, verbose=0)
            st.success(f"Predicted Digit: **{np.argmax(prediction)}**")
        else:
            st.warning("Model is not loaded. Cannot predict.")
