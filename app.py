import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image

# Load trained model
model = load_model("face_mask_model.h5")

st.title("😷 Real-Time Face Mask Detection")
st.write("Upload an image or use webcam to detect mask status.")

# Upload image section
uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "png", "jpeg"])


def predict(img):
    img = img.resize((128, 128))
    img = np.array(img) / 255.0
    img = img.reshape(1, 128, 128, 3)

    pred = model.predict(img)[0][0]
    
    return "MASK 😷" if pred < 0.5 else "NO MASK ❌"


# ----------- IMAGE PREDICTION -----------
if uploaded_file:
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Image", width=300)

    result = predict(img)
    st.markdown(f"### Prediction: **{result}**")


# ----------- REAL-TIME WEBCAM-------------
st.write("---")
st.subheader("📷 Live Webcam Detection")

run_camera = st.checkbox("Start Webcam")

face_detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

if run_camera:
    camera = cv2.VideoCapture(0)

    while run_camera:
        ret, frame = camera.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = face_detector.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            face = frame[y:y+h, x:x+w]
            face = cv2.resize(face, (128,128)) / 255.0
            face = face.reshape(1,128,128,3)

            pred = model.predict(face)[0][0]

            label = "MASK 😷" if pred < 0.5 else "NO MASK ❌"
            color = (0,255,0) if pred < 0.5 else (0,0,255)

            cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            cv2.rectangle(frame, (x,y), (x+w,y+h), color, 2)

        st.image(frame, channels="BGR")
    camera.release()
