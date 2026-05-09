import gradio as gr
import requests

API_URL = "http://127.0.0.1:8001/predict"

def classify_image(image_path):

    with open(image_path, "rb") as f:

        response = requests.post(
            API_URL,
            files={"file": f}
        )

    return response.json()

interface = gr.Interface(
    fn=classify_image,
    inputs=gr.Image(type="filepath"),
    outputs="json",
    title="CNN Image Classifier"
)

interface.launch()