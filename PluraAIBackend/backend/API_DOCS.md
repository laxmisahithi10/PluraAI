# PluraAI Backend API Documentation

## Overview
Backend API for chest X-ray pneumonia detection using Vision Transformer (ViT) + CALSL model.

## Base URL
```
http://localhost:8000/api/
```

## Endpoints

### POST /api/predict/
Predicts pneumonia from chest X-ray image.

**Request:**
- Method: POST
- Content-Type: multipart/form-data
- Body: 
  - `image`: Image file (JPG/PNG, max 10MB)

**Response:**
```json
{
    "prediction": "Normal" | "Pneumonia",
    "confidence": 0.85,
    "inference_time": 0.123,
    "status": "success"
}
```

**Error Responses:**
- 400: Invalid image format/size
- 500: Internal server error

## Testing

### Using curl:
```bash
curl -X POST http://localhost:8000/api/predict/ \
  -F "image=@chest_xray.jpg"
```

### Using Python requests:
```python
import requests

url = "http://localhost:8000/api/predict/"
files = {"image": open("chest_xray.jpg", "rb")}
response = requests.post(url, files=files)
print(response.json())
```

## Setup Instructions

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run migrations:
```bash
python manage.py migrate
```

3. Start server:
```bash
python manage.py runserver
```

## Model Integration
Uses the trained ViT+CALSL model (`vitcal_best (1).pth`). Model is loaded once at first request and cached in memory.

Preprocessing pipeline:
1. Load image with OpenCV
2. Split into full / left-lung / right-lung views
3. Extract 196 patches of 16×16 pixels → (196, 768) tensors
4. Run through CALSLModel encoder
5. Softmax over 2 classes (Normal / Pneumonia)