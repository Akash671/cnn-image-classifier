#API_URL = "http://127.0.0.1:8001/predict"
import streamlit as st
import requests

# FastAPI endpoint
#API_URL = "http://127.0.0.1"

API_URL = "http://127.0.0.1:8001/predict"


st.title("CNN Image Classifier")

# Image Uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)
    
    # Predict button
    if st.button("Classify"):
        with st.spinner("Classifying..."):
            # Prepare the file for the requests post
            files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
            
            try:
                response = requests.post(API_URL, files=files)
                if response.status_code == 200:
                    st.success("Classification Complete!")
                    # Pretty-print the JSON output
                    st.json(response.json())
                else:
                    st.error(f"Error: {response.status_code}")
            except Exception as e:
                st.error(f"Connection failed: {e}")
