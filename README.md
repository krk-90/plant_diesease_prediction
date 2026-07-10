## 📌 Overview
This project uses deep learning to classify plant leaf images into six categories:
Canker, Greening, Gummosis, Healthy, Leaf‑miner, and Lemon‑butterfly.

By leveraging ResNet18, a powerful convolutional neural network, the model helps farmers and researchers detect plant diseases early. Early detection means healthier crops, improved yields, and reduced losses.

## Plant Disease Prediction API

A FastAPI service that classifies plant leaf images into one of six categories using a fine-tuned ResNet18 model (PyTorch).

## Classes

The model predicts one of the following:

- Canker
- Greening
- Gummosis
- Healthy
- Leaf-miner
- Lemon-butterfly

## Project Structure

```
plant_disease_prediction/
├── app/
│   ├── backend/
│   │   ├── fast_api.py       # FastAPI app + /predict endpoint
│   │   ├── model.py          # Model definition + loading
│   │   └── preprocess.py     # Image preprocessing pipeline
│   └── Resnet/
│       └── plant_disease_model.pth   # Trained model weights
├── requirements.txt
└── README.md
```

## Requirements

- Python 3.10+
- See `requirements.txt` for Python package dependencies

## Setup

1. **Clone the repository**
   ```bash
   git clone <your-repo-url>
   cd plant_disease_prediction
   ```

2. **Create and activate a virtual environment**
   ```bash
   python -m venv venv
   # Windows
   venv\Scripts\activate
   # macOS/Linux
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Add the model weights**

   Ensure `plant_disease_model.pth` is placed at `app/Resnet/plant_disease_model.pth`.

## Running the Server

From `app/backend/`:

```bash
cd app/backend
uvicorn fast_api:app --reload --host 0.0.0.0 --port 8000
```

The server will be available locally at:

```
http://127.0.0.1:8000
```

## API Endpoints

### `GET /health`

Health check — confirms the server is running and the model loaded successfully.

**Response**
```json
{
  "status": "ok",
  "model_loaded": true
}
```

### `POST /predict/`

Upload an image and receive a disease classification.

**Request**: `multipart/form-data` with a `file` field containing an image (JPEG/PNG).

**Response**
```json
{
  "filename": "leaf.jpg",
  "prediction": "Healthy",
  "confidence": 0.9231
}
```

### Interactive Docs

Once the server is running, visit:

```
http://127.0.0.1:8000/docs
```

for a Swagger UI where you can test the `/predict/` endpoint directly from the browser.

## Model Details

- **Architecture**: ResNet18 (from `torchvision.models`), final fully-connected layer replaced to output 6 classes
- **Input size**: 224×224 RGB images
- **Normalization**: ImageNet mean/std (`[0.485, 0.456, 0.406]` / `[0.229, 0.224, 0.225]`)

## License
This project is open‑source. Feel free to use, modify, and share with proper attribution.