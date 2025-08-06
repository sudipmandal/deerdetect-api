
# 🦌 Deer Detection API

This FastAPI application uses a Vision Transformer (ViT) model from Hugging Face to detect whether a deer is present in an image. The model is fine-tuned and hosted at [`samipfjo/deer-detection`](https://huggingface.co/samipfjo/deer-detection).

## 🚀 How It Works
- Accepts a POST request with an image URL.
- Downloads and preprocesses the image using `ViTImageProcessor`.
- Runs inference using the ViT model.
- Returns a boolean indicating if a deer is present.


## 🐳 Docker Setup
### 1. Build the Docker image
```bash
docker build -t deer-detector .
```

### 2. Run the container
```bash
docker run -p 8000:8000 deer-detector
```

## 📦 API Usage
Send a POST request to `/detect_deer` with a JSON body:
```json
{
  "url": "https://example.com/image.jpg"
}
```

### Example using curl:
```bash
curl -X POST http://localhost:8000/detect_deer      -H "Content-Type: application/json"      -d '{"url": "https://example.com/image.jpg"}'
```

## 📄 Response
```json
{
  "deer_present": true
}
```

## 📁 Files
- `main.py`: FastAPI application.
- `Dockerfile`: Container configuration.
- `requirements.txt`: Python dependencies.
- `README.md`: Project documentation.


