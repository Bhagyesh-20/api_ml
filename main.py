from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from PIL import Image
import numpy as np
from tensorflow.keras.models import load_model
import logging
import io
import boto3
import os
from dotenv import load_dotenv

# Initialize FastAPI app
app = FastAPI()

# Configure logging
logger = logging.getLogger("predict_logger")
logging.basicConfig(level=logging.DEBUG)

# Load environment variables
load_dotenv()

# AWS S3 configuration
S3_BUCKET_NAME = os.getenv('S3_BUCKET_NAME')
MODEL_FILE_NAME = 'crop_disease_model.h5'

# Download model from S3
def download_model_from_s3():
    s3 = boto3.client(
        's3',
        aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
        aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
        region_name=os.getenv('AWS_REGION')
    )
    s3.download_file(S3_BUCKET_NAME, MODEL_FILE_NAME, MODEL_FILE_NAME)

# Ensure model file is present
if not os.path.exists(MODEL_FILE_NAME):
    logger.info("Downloading model from S3...")
    download_model_from_s3()

# Load the model
model = load_model(MODEL_FILE_NAME)

# Class names for the predictions
class_names = [
    'Anthracnose', 'Apple Scab', 'Black Spot', 
    'Blight', 'Blossom End Rot', 'Botrytis', 'Brown Rot',
    'Canker', 'Cedar Apple Rust', 'Clubroot', 'Crown Gall',
    'Downy Mildew', 'Fire Blight', 'Fusarium', 'Gray Mold',
    'Leaf Spots', 'Mosaic Virus', 'Nematodes', 'Powdery Mildew',
    'Verticilium'
]

@app.get("/")
async def root():
    return {"message": "Welcome to the Crop Disease Prediction API!"}

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    logger.debug("Received file: %s", file.filename)
    
    try:
        # Read file content
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))

        # Preprocess the image
        image = image.resize((224, 224))
        image = np.array(image) / 255.0
        image = np.expand_dims(image, axis=0)

        # Make prediction
        prediction = model.predict(image)
        predicted_class = np.argmax(prediction)
        predicted_class_name = class_names[predicted_class]

        return JSONResponse(content={"prediction": predicted_class_name})

    except Exception as e:
        logger.error("Error processing file: %s", str(e))
        raise HTTPException(status_code=400, detail="Could not process the file")

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
