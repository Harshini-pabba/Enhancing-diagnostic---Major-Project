import streamlit as st
from PIL import Image
import os
from tensorflow import keras

# --- Custom Modules ---
from predict_module import predict_new_image
from lime_module import generate_lime_explanation
from gradcam_module import generate_gradcam

# ==============================
# 1. Model Loading
# ==============================
MODEL_PATH = r"D:\MajorProjectChestXRay\MobileNetV2_chestxray_model.keras"
model = keras.models.load_model(MODEL_PATH, compile=False)

# ==============================
# 2. Streamlit Page Setup
# ==============================
st.set_page_config(
    page_title="Chest X-Ray Diagnosis",
    page_icon="🧬",
    layout="wide"
)

# ==============================
# 3. Custom CSS for Styling
# ==============================
st.markdown("""
<style>
h1 {
    text-align: center;
    color: #2E8B57;
}
p, h3 {
    text-align: center;
}
img {
    display: block;
    margin-left: auto;
    margin-right: auto;
    max-width: 700px !important;
    height: auto;
    border-radius: 10px;
    box-shadow: 0 0 12px rgba(0,0,0,0.15);
}
.result-card {
    background-color: #f8f9fa;
    border-radius: 10px;
    padding: 15px;
    margin-top: 20px;
    box-shadow: 0 0 8px rgba(0,0,0,0.1);
    text-align: center;
}
footer {
    text-align: center;
    color: gray;
    margin-top: 30px;
}
</style>
""", unsafe_allow_html=True)

# ==============================
# 4. App Title & Description
# ==============================
st.markdown("<h1>Chest X-Ray Diagnosis</h1>", unsafe_allow_html=True)
st.markdown("""
<p>Upload a chest X-ray image to diagnose it 
and visualize explanations.</p>
""", unsafe_allow_html=True)
st.write("---")



# ==============================
# 5. File Upload Section
# ==============================
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

col1, col2 = st.columns([1, 2])
with col1:
    uploaded_file = st.file_uploader(" Upload Chest X-Ray Image", type=["jpg", "jpeg", "png"])
with col2:
    st.markdown("""
        <div style="background-color:#eef5f9; padding:15px; border-radius:10px; box-shadow:0 0 5px rgba(0,0,0,0.1);">
        <h4 style="text-align:center; color:#0f5c70;">Diagnostic Capabilities</h4>
        <p style="text-align:center;">This AI-powered model can identify the following chest conditions:</p>
        <table style="margin:auto; text-align:center;">
        <tr><td> <b>Normal</b></td><td> <b>Covid-19</b></td></tr>
        <tr><td> <b>Pneumonia</b></td><td> <b>Tuberculosis</b></td></tr>
        </table>
        </div>
        """, unsafe_allow_html=True)

# ==============================
# 6. Main Logic
# ==============================
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption=" Uploaded X-Ray", use_container_width=True)

    # Save the uploaded image
    save_path = os.path.join(UPLOAD_FOLDER, uploaded_file.name)
    with open(save_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Buttons for actions
    colA, colB, colC = st.columns(3)
    with colA:
        classify = st.button(" Classify Image")
    with colB:
        lime_btn = st.button(" Generate LIME")
    with colC:
        gradcam_btn = st.button(" Generate Grad-CAM")

    # --- Prediction ---
    if classify:
        with st.spinner("Analyzing the X-ray... Please wait ⏳"):
            pred_class, confidence = predict_new_image(save_path, model)

        st.markdown(f"""
        <div class='result-card'>
            <h3> Prediction Result</h3>
            <p><b>Disease Class:</b> {pred_class}</p>
            <p><b>Confidence:</b> {confidence:.2f}%</p>
        </div>
        """, unsafe_allow_html=True)

    # --- LIME Visualization ---
    if lime_btn:
        with st.spinner("Generating LIME Explanation... ⏳"):
            lime_img_path = generate_lime_explanation(save_path)
        st.image(lime_img_path, caption=" LIME Explanation", use_container_width=True)

    # --- Grad-CAM Visualization ---
    if gradcam_btn:
        with st.spinner("Computing Grad-CAM Visualization... ⏳"):
            try:
                gradcam_img_path = generate_gradcam(save_path)
                st.image(gradcam_img_path, caption=" Grad-CAM Visualization", use_container_width=True)
                st.info(" Red regions show the most influential parts for prediction.")
            except ValueError as e:
                st.error(f"⚠️ Grad-CAM Error: {e}")

else:
    st.info("👆 Upload a chest X-ray image to begin.")

# ==============================
# 7. Footer
# ==============================
st.markdown("""
<footer>
<hr>
<b> AI-Powered Chest X-Ray Diagnostic System</b><br>
<small>Disclaimer: This is a clinical assistance tool and not a replacement for medical professionals.</small>
</footer>
""", unsafe_allow_html=True)