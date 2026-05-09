from fastapi import FastAPI, UploadFile, File
from app.utils import read_image
from app.predict import predict



app = FastAPI()


@app.get("/") # get method -> retrieve data from server using api
def home():
    return {"message": "Welcome to the Image Classification API!"}


@app.post('/predict') # post method -> send data to server using api..
async def predict_api(file: UploadFile = File(...)):
    image = read_image(await file.read())

    predicted_class, confidence = predict(image)

    return {
        "class" : predicted_class,
        "confidence" : confidence
    }


