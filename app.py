

import streamlit as st
import numpy as np
import cv2
from streamlit_drawable_canvas import st_canvas
from tensorflow.keras.models import load_model

# ---------------- Page Config ----------------
st.set_page_config(
    page_title="Digit Recognition",
    page_icon="✍️",
    layout="centered"
)

# ---------------- CSS ----------------
st.markdown("""
<style>

.stApp{
    background: linear-gradient(to right,#eef2ff,#f8fbff);
}

h1{
    color:#1E3A8A;
    text-align:center;
}

.info{
    background:white;
    padding:15px;
    border-radius:12px;
    box-shadow:0px 0px 10px rgba(0,0,0,0.15);
    margin-bottom:20px;
}

.result{
    background:#d1fae5;
    padding:20px;
    border-radius:15px;
    text-align:center;
    font-size:28px;
    font-weight:bold;
    color:#065f46;
}

.stButton>button{
    width:100%;
    background:#2563EB;
    color:white;
    border-radius:10px;
    font-size:18px;
    font-weight:bold;
    height:50px;
}

.stButton>button:hover{
    background:#1d4ed8;
    color:white;
}

</style>
""", unsafe_allow_html=True)

# ---------------- Load Model ----------------
model = load_model("digit_recognition_model.keras")

# ---------------- Title ----------------
st.title("✍️ Handwritten Digit Recognition")

st.markdown("""
<div class="info">

### 📝 Instructions

✔ Draw one digit (0-9) inside the box.

✔ Click **Predict Digit**.

✔ The AI model will identify your handwritten digit.

</div>
""", unsafe_allow_html=True)

# ---------------- Canvas ----------------
canvas_result = st_canvas(
    fill_color="#00000000",
    stroke_width=10,
    stroke_color="#FFFFFF",
    background_color="#000000",
    width=280,
    height=280,
    drawing_mode="freedraw",
    key="canvas",
)

# ---------------- Prediction ----------------
if st.button("🔍 Predict Digit"):

    if canvas_result.image_data is not None:

        img = canvas_result.image_data.astype(np.uint8)

        grey_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        grey_img = cv2.resize(grey_img, (28,28))

        grey_img = grey_img / 255.0

        grey_img = grey_img.reshape(-1,784)

        prediction = model.predict(grey_img)

        digit = np.argmax(prediction)

        confidence = np.max(prediction) * 100

        st.markdown(f"""
        <div class="result">

        🎯 Predicted Digit

        <h1>{digit}</h1>

        Confidence : <b>{confidence:.2f}%</b>

        </div>
        """, unsafe_allow_html=True)

        st.progress(float(confidence)/100)

        st.success("Prediction Completed Successfully ✅")