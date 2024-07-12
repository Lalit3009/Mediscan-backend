from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import cv2
import os
from tensorflow.keras.models import load_model
import requests  # Import requests for image download

app = FastAPI()

# Set up CORS middleware
origins = [
    "http://localhost:5173",  # The origin of the frontend server
    "http://localhost:3000",  # The origin of the frontend server
    "https://yourproductionwebsite.com"  # If you have a production frontend
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the pre-trained Keras model
model = load_model('model.h5')

CLASSES = ['COVID19', 'NORMAL', 'PNEUMONIA', 'TUBERCULOSIS']

def transform_image(image):
    image_size = (224, 224)
    new_image = cv2.resize(image, image_size)
    new_image = new_image.astype(np.float32) / 255.0
    new_image = np.expand_dims(new_image, axis=0)
    return new_image

@app.get("/")
async def home():
    return {"Message": "SERVER IS RUNNING FINE"}

@app.post("/predict")
async def predict_image(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        image_path = "uploaded_image.jpg"
        with open(image_path, "wb") as f:
            f.write(contents)

        transformed_image = transform_image(cv2.imread(image_path))

        prediction = model.predict(transformed_image)
        class_index = np.argmax(prediction)
        class_name = CLASSES[class_index]

        probabilities = [{"name": name, "probability": float(val)} for name, val in zip(CLASSES, prediction[0])]

        # Remove the temporary image file
        os.remove(image_path)

        return {"prediction": class_name, "probabilities": probabilities}
    except Exception as e:
        return {"error": "An error occurred during prediction"}

@app.get("/test")
async def test():
    return {"message" : "TEST IS RUNNING FINE"}
