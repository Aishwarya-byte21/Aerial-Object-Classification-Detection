import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from ultralytics import YOLO
import tempfile
import os

# Load classification model
clf_model = tf.keras.models.load_model("transfer_model.h5")

# Load YOLO model
yolo_model = YOLO("runs/detect/train/weights/best.pt")

IMG_SIZE = 224

st.title("🛰️ Aerial Object Classification & Detection")
st.write("Upload an image to classify and detect aerial objects.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # ----- Classification -----
    img_resized = image.resize((IMG_SIZE, IMG_SIZE))
    img_array = np.array(img_resized) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = clf_model.predict(img_array)[0][0]
    confidence = prediction if prediction > 0.5 else 1 - prediction

    label = "Drone 🚁" if prediction > 0.5 else "Bird 🐦"

    st.subheader("📌 Classification Result")
    st.write(f"Prediction: **{label}**")
    st.write(f"Confidence: **{confidence:.2f}**")

    # ----- YOLO Detection -----
    st.subheader("🎯 YOLO Detection Result")

    # Save temp image
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
        image.save(tmp.name)
        results = yolo_model(tmp.name)

    # Show detection image
    for r in results:
        annotated_image = r.plot()
        st.image(annotated_image, caption="Detected Image", use_column_width=True)