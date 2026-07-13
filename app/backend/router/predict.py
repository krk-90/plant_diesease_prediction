import os 
import sys
import io
import torch
from PIL import Image,UnidentifiedImageError
from fastapi import FastAPI,File, UploadFile, HTTPException
from contextlib import asynccontextmanager

PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from preprocess import processed
from model import main

model = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global model
    try:
        model = main()
    except Exception as e:
        print(f"Error loading model: {e}")
        raise
    yield

app = FastAPI(title="plant_diesease_prediction",version="1.0.0",lifespan=lifespan)

class_names = ['Canker', 'Greening', 'Gummosis', 'Healthy', 'Leaf-miner', 'Lemon-butterfly']
@app.post("/predict/")
async def predict_image(file:UploadFile=File(...)):
    if model is None:
        raise HTTPException(status_code=503,detail="model not found")
    
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400,detail="file must be an image")
    
    content = await file.read()
    try:
        img = Image.open(io.BytesIO(content)).convert("RGB")
    except UnidentifiedImageError:
        raise HTTPException(status_code=400,detail="could not read image file")
    
    Img_tensor = processed(img)

    with torch.no_grad():
        output_class = model(Img_tensor)
        probs = torch.softmax(output_class,1)
        confidence,predicted_class = torch.max(probs,1)

    return {
        "filename":file.filename,
        "prediction":class_names[predicted_class.item()],
        "confidence":round(confidence.item(),4)
    }    

@app.get("/health")
def health_check():
    return {"status": "ok", "model_loaded": model is not None}