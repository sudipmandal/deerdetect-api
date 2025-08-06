from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import requests
from PIL import Image
from io import BytesIO
import torch
import os
from transformers import ViTForImageClassification, ViTFeatureExtractor
app = FastAPI()

class ImageRequest(BaseModel):
    url: str

# Load model and feature extractor from environment variable, if not set use the default samipfjo/deer-detection model
model_name = os.getenv("HG-VIT-MODEL", "samipfjo/deer-detection")

print(f"Using model: {model_name}")
model = ViTForImageClassification.from_pretrained(model_name)

model.eval()
feature_extractor = ViTFeatureExtractor.from_pretrained(model_name)

@app.post("/detect_deer")
async def detect_deer(request: ImageRequest):
    try:
        response = requests.get(request.url)
        image = Image.open(BytesIO(response.content)).convert("RGB")
        inputs = feature_extractor(images=image, return_tensors="pt")

        with torch.no_grad():
            outputs = model(**inputs)
            predicted_class = outputs.logits.argmax(-1).item()

        return {"deer_present": predicted_class == 1}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
