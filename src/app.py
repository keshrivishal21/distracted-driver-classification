import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image

# -------------------------------
# CONFIG
# -------------------------------
st.set_page_config(page_title="Driver Detection", layout="centered")

# -------------------------------
# TITLE
# -------------------------------
st.title("🚗 AI-Based Distracted Driver Detection")
st.markdown("Upload an image to classify driver behavior")

# -------------------------------
# LOAD MODEL (cached)
# -------------------------------
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("C:\\Users\\visha\\Downloads\\ML project\\distracted_driver_classification\\models\\final_model.keras")
    return model

model = load_model()

# -------------------------------
# CLASS NAMES (IMPORTANT)
# -------------------------------
class_names = [
    "Safe Driving",
    "Texting Right",
    "Talking Right",
    "Texting Left",
    "Talking Left",
    "Operating Radio",
    "Drinking",
    "Reaching Behind",
    "Hair & Makeup",
    "Talking to Passenger"
]

# -------------------------------
# IMAGE PREPROCESSING
# -------------------------------
def preprocess_image(image):
    image = image.resize((224, 224))
    image = np.array(image)
    
    if image.shape[-1] == 4:  # RGBA to RGB
        image = image[:, :, :3]
        
    image = image / 255.0
    image = np.expand_dims(image, axis=0)
    
    return image

# -------------------------------
# FILE UPLOAD
# -------------------------------
uploaded_file = st.file_uploader("📤 Upload Driver Image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file)
    
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Predict
    with st.spinner("🔍 Analyzing driver behavior..."):
        processed = preprocess_image(image)
        preds = model.predict(processed)[0]

    # Results
    predicted_class = class_names[np.argmax(preds)]
    confidence = np.max(preds)

    st.success(f"✅ Prediction: {predicted_class}")
    st.write(f"🎯 Confidence: {confidence:.2f}")

    # Bar chart
    st.subheader("📊 Prediction Probabilities")
    st.bar_chart(preds)