import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image

# ---------------------------
# Page config
# ---------------------------
st.set_page_config(page_title="Brain Tumor Segmentation", layout="wide")
st.title("🧠 Brain Tumor Segmentation with U-Net")

# ---------------------------
# Load model
# ---------------------------
MODEL_PATH = "brain_tumor_unet.h5"
model = load_model(MODEL_PATH, compile=False)

# ---------------------------
# Colormap for overlay
# ---------------------------
colors = [
    [0,0,0],       # Background
    [255,0,0],     # Glioma
    [0,255,0],     # Meningioma
    [0,0,255]      # Pituitary
]

# ---------------------------
# Minimalistic upload box
# ---------------------------
uploaded_image = st.file_uploader("Upload MRI Image", type=["jpg", "png", "jpeg"])

# ---------------------------
# Process uploaded image
# ---------------------------
if uploaded_image:
    img = Image.open(uploaded_image).convert("RGB")
    img_np = np.array(img)
    img_resized = cv2.resize(img_np, (128,128))
    
    img_input = img_resized / 255.0
    img_input = np.expand_dims(img_input, axis=0)
    
    # Predict mask
    pred_mask = model.predict(img_input)[0]
    pred_mask_class = np.argmax(pred_mask, axis=-1)
    
    # Colored overlay
    colored_mask = np.zeros((128,128,3), dtype=np.uint8)
    for i, color in enumerate(colors):
        colored_mask[pred_mask_class == i] = color
    overlay = cv2.addWeighted(img_resized, 0.7, colored_mask, 0.3, 0)
    
    # Grayscale mask
    grayscale_mask_display = (pred_mask_class * (255 // 3)).astype(np.uint8)
    
    # ---------------------------
    # Display in order: Original → Grayscale → Overlay
    # ---------------------------
    st.markdown("### Prediction Results")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.image(img_np, caption="Original Image", use_container_width=True)
    
    with col2:
        st.image(grayscale_mask_display, caption="Predicted Mask (Grayscale)", use_container_width=True)
    
    with col3:
        st.image(overlay, caption="Predicted Mask Overlay", use_container_width=True)
    
    st.success("✅ Prediction completed!")
