"""
author : @akash
"""


import sys
import os
import streamlit as st

# Force the working directory to be the root of the project
# This allows 'import app.utils' to work correctly
root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if root_path not in sys.path:
    sys.path.insert(0, root_path)

# Now try the imports
try:
    from app.utils import read_image
    from app.predict import predict
except ModuleNotFoundError:
    # Backup if 'app' is being treated as the root itself
    from utils import read_image
    from predict import predict

st.title("CNN Image Classifier")

# Image Uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)
    
    # Predict button
    if st.button("Classify"):
        with st.spinner("Classifying..."):
            try:
                # 1. Read the image using your existing utils
                image_bytes = uploaded_file.getvalue()
                image = read_image(image_bytes)

                # 2. Run prediction directly (No API call needed!)
                predicted_class, confidence = predict(image)

                # 3. Show results
                st.success("Classification Complete!")
                st.write(f"**Result:** {predicted_class}")
                st.write(f"**Confidence:** {confidence:.2f}")
                
            except Exception as e:
                st.error(f"Prediction failed: {e}")
