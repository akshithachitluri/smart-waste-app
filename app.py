import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import tempfile
from ultralytics import YOLO
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
from datetime import date, timedelta

# ---------------- ## New Feature ##: Initialize Session State ----------------
# This ensures that points and streak data persists across user interactions.
if 'points' not in st.session_state:
    st.session_state.points = 0
if 'streak' not in st.session_state:
    st.session_state.streak = 0
if 'last_classification_date' not in st.session_state:
    st.session_state.last_classification_date = None

# ---------------- ## New Feature ##: Gamification Logic ----------------
def update_gamification():
    """
    Awards points and updates the daily streak.
    """
    st.session_state.points += 10 # Award 10 points for each classification
    today = date.today()

    # Check and update the streak
    last_date = st.session_state.last_classification_date
    if last_date is None: # First classification
        st.session_state.streak = 1
    elif today == last_date: # Already classified today, do nothing to streak
        pass
    elif today == last_date + timedelta(days=1): # Consecutive day
        st.session_state.streak += 1
        st.balloons() # Celebrate the new streak!
    else: # Missed a day or more, reset streak
        st.session_state.streak = 1

    # Update the last classification date
    st.session_state.last_classification_date = today

# ---------------- Model Loading ----------------

# Load TensorFlow SavedModel (for single waste)
# NOTE: Ensure the 'waste_model' directory is in the same folder as your app.py
try:
    model = tf.keras.layers.TFSMLayer("waste_model", call_endpoint="serving_default")
except Exception as e:
    st.error(f"Error loading TensorFlow model: {e}")
    st.stop() # Stop the app if model loading fails

# Load YOLOv8 model (for multi-waste)
yolo_model = YOLO("yolov8n.pt")

# Waste classes
class_names = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']

# Categorize for YOLO
dry = {"bottle", "paper", "can", "cup", "cardboard", "plastic"}
wet = {"banana", "food", "apple", "vegetable", "orange"}
hazardous = {"battery", "syringe", "broken glass"}

# Disposal tips for TF model
tips = {
    "plastic": "Recycle in dry plastic bin.",
    "glass": "Recycle at a glass collection center.",
    "metal": "Put in a metal recycling bin.",
    "paper": "Recycle with dry paper waste.",
    "cardboard": "Flatten and recycle it.",
    "trash": "Dispose in general waste bin."
}

# ---------------- UI Setup ----------------

st.title("♻️ Smart Waste Classifier")

# ---------------- ## New Feature ##: Display Points and Streak ----------------
col1, col2 = st.columns(2)
col1.metric("🏆 Points", st.session_state.points)
col2.metric("🔥 Streak", f"{st.session_state.streak} Days")
st.markdown("---") # Visual separator

mode = st.radio("Choose Detection Mode:", ["Single Waste (Simple)", "Multi-Waste (Advanced)"])
st.write("Upload a waste image and get classification with disposal tip!")

# ---------------- Single Waste Detection ----------------

if mode == "Single Waste (Simple)":
    uploaded_file = st.file_uploader("Choose a waste image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        def preprocess_image(image):
            image = image.resize((150, 150))
            img_array = np.array(image) / 255.0
            # Ensure 3 channels for RGB images
            if img_array.shape[-1] == 4: # Handle RGBA
                img_array = img_array[..., :3]
            img_array = np.expand_dims(img_array, axis=0)
            return img_array

        def predict(image):
            img_array = preprocess_image(image)
            prediction_dict = model(img_array)
            # The output key might be different, check your model signature
            prediction_key = list(prediction_dict.keys())[0]
            prediction = prediction_dict[prediction_key].numpy()
            predicted_class = class_names[np.argmax(prediction)]
            return predicted_class


        with st.spinner("Predicting..."):
            try:
                predicted_class = predict(image)
                st.success(f"🧠 Predicted: {predicted_class.upper()}")
                st.info(f"💡 Tip: {tips[predicted_class]}")
                update_gamification() ## New Feature ##: Update score and streak
            except Exception as e:
                st.error("Prediction failed. Please try another image.")
                st.exception(e)

# ---------------- Multi Waste Detection ----------------

elif mode == "Multi-Waste (Advanced)":
    st.subheader("Choose Image Input Method")
    tab1, tab2 = st.tabs(["📁 Upload Image", "📷 Use Camera"])

    # Tab 1: Upload image
    with tab1:
        uploaded_file = st.file_uploader("Upload landfill image...", type=["jpg", "jpeg", "png"])
        if uploaded_file is not None:
            st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)

            with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
                tmp.write(uploaded_file.read())
                temp_img_path = tmp.name

            results = yolo_model(temp_img_path)
            detected = results[0].names
            boxes = results[0].boxes
            class_ids = boxes.cls.cpu().numpy().astype(int)
            detected_objects = list(set([detected[i] for i in class_ids]))

            if detected_objects:
                update_gamification() ## New Feature ##: Update score and streak
                st.markdown("### ♻️ Detected Waste Items:")
                for obj in detected_objects:
                    if obj in dry:
                        st.write(f"🔵 {obj} → Dry Waste")
                    elif obj in wet:
                        st.write(f"🟢 {obj} → Wet Waste")
                    elif obj in hazardous:
                        st.write(f"🔴 {obj} → Hazardous Waste")
                    else:
                        st.write(f"⚪ {obj} → Unknown")
            else:
                st.warning("No waste items detected in the image.")


    # Tab 2: Live webcam
    with tab2:
        st.write("📷 Use your webcam to capture and detect waste live")
        st.info("Note: Live detection does not award points or update streaks.")

        class WasteDetector(VideoTransformerBase):
            def transform(self, frame):
                img = frame.to_ndarray(format="bgr24")
                results = yolo_model.predict(source=img, save=False, conf=0.3, verbose=False)
                return results[0].plot()

        webrtc_streamer(
            key="waste-live-cam",
            video_transformer_factory=WasteDetector,
            rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
        )

# ---------------- Location Finder ----------------

st.markdown("---")
st.subheader("📍 Find Nearest Waste Disposal Center")

user_input = st.text_input("Enter your area name or pincode:")

# A simple dictionary for location mapping.
# In a real-world app, you might use a database or a Geocoding API.
waste_centers = {
    "500032": "GHMC Dry Waste Center, CyberTowers, Hyderabad",
    "500081": "Kondapur Compost Point, Hyderabad",
    "500072": "Kukatpally Plastic Drop Zone, Hyderabad",
    "500018": "Miyapur Wet Waste Station, Hyderabad",
    "Gachibowli": "Bio Waste Pit, Gachibowli Cross, Hyderabad",
    "KPHB": "Paper Sorting Hub, KPHB Phase 3, Hyderabad",
    "Miyapur": "Wet Waste Station near BHEL, Hyderabad",
    "Kondapur": "Plastic Shredding Point near Botanical Garden, Hyderabad",
    "Narsapur": "T.m traders, Narsapur",
}

if user_input:
    # Use .strip() to remove leading/trailing whitespace
    found_location = waste_centers.get(user_input.strip().title()) # .title() can help match "gachibowli" to "Gachibowli"
    if found_location:
        st.success(f"✅ Nearest Disposal Location: {found_location}")
        # Properly format for Google Maps URL
        maps_url = f"https://www.google.com/maps/search/?api=1&query={found_location.replace(' ', '+')}"
        st.markdown(f"[🗺️ Open in Google Maps]({maps_url})")
    else:
        st.warning("🚫 No matching location found. Try another area or pincode.")