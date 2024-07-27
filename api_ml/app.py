from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from PIL import Image
import numpy as np
from tensorflow.keras.models import load_model
import logging
import io

app = FastAPI()

logger = logging.getLogger("predict_logger")
logging.basicConfig(level=logging.DEBUG)

model_path = "crop_disease_model.h5"  
model = load_model(model_path)

class_names = [
    'Anthracnose', 'Apple Scab', 'Black Spot', 
    'Blight', 'Blossom End Rot', 'Botrytis', 'Brown Rot',
    'Canker', 'Cedar Apple Rust', 'Clubroot', 'Crown Gall',
    'Downy Mildew', 'Fire Blight', 'Fusarium', 'Gray Mold',
    'Leaf Spots', 'Mosaic Virus', 'Nematodes', 'Powdery Mildew',
    'Verticilium'
]


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

