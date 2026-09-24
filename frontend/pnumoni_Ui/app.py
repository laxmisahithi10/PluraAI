import os
import requests as http_requests
from dotenv import load_dotenv
load_dotenv()
from flask import Flask, render_template, request, jsonify, session

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "plura-dev-secret")

OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")

SYSTEM_PROMPT = (
    "You are a medical AI assistant for PLURA, an AI-powered pneumonia detection system. "
    "Explain pneumonia in simple terms, help interpret X-ray results, and provide general guidance. "
    "Never give a final diagnosis; always advise consulting a licensed doctor. "
    "When asked how the system works, explain that it uses Vision Transformers (ViT) to compare "
    "lung symmetry — analyzing texture and density differences between the left and right lungs."
)

RESEARCH_PAPERS = [
    {
        "title": "Vision Transformer for Pneumonia Detection from Chest X-ray Images",
        "year": 2023,
        "authors": "Zhang et al.",
        "journal": "Medical Image Analysis",
        "abstract": "A ViT-based approach achieving 97.3% accuracy on the NIH Chest X-ray dataset by leveraging self-attention for global feature extraction.",
        "tags": ["Vision Transformer", "X-ray", "Classification"],
    },
    {
        "title": "Context-Aware Lung Symmetry Learning for Pneumonia Detection",
        "year": 2024,
        "authors": "Chen & Patel",
        "journal": "Nature Digital Medicine",
        "abstract": "Novel framework comparing bilateral lung symmetry using contrastive learning, mimicking radiologist diagnostic reasoning.",
        "tags": ["Symmetry Analysis", "Contrastive Learning", "ViT"],
    },
    {
        "title": "Deep Learning for Pneumonia Detection: A Comprehensive Review",
        "year": 2022,
        "authors": "Kumar et al.",
        "journal": "IEEE Transactions on Medical Imaging",
        "abstract": "Systematic review of 120+ deep learning models for pneumonia detection, benchmarking CNNs, RNNs, and transformer architectures.",
        "tags": ["Survey", "Deep Learning", "Benchmarking"],
    },
    {
        "title": "Automated Pneumonia Detection using Deep Learning with Chest X-Ray Images",
        "year": 2023,
        "authors": "Okonkwo & Liu",
        "journal": "Journal of Digital Imaging",
        "abstract": "End-to-end pipeline combining lung segmentation with EfficientNet-B4 for automated pneumonia screening in resource-limited settings.",
        "tags": ["Segmentation", "EfficientNet", "Screening"],
    },
]


BACKEND_URL = os.environ.get("BACKEND_URL", "http://127.0.0.1:8000")


def analyze_xray(img):
    """Forward image to Django backend for real ViT model inference."""
    response = http_requests.post(
        f"{BACKEND_URL}/api/predict/",
        files={"image": (img.filename, img.stream, img.content_type)},
        timeout=60,
    )
    response.raise_for_status()
    data = response.json()
    return {
        "prediction": data["prediction"],
        "confidence": data["confidence"],
        "confidence_percent": data.get("confidence_percent"),
        "explanation": data.get("explanation"),
        "case_id": data.get("case_id"),
        "inference_time": data.get("inference_time"),
    }


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/detect")
def detect():
    return render_template("detect.html")


@app.route("/research")
def research():
    return render_template("research.html", papers=RESEARCH_PAPERS)


@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files or request.files["file"].filename == "":
        return jsonify({"error": "No file provided"}), 400
    img = request.files["file"]
    try:
        result = analyze_xray(img)
    except Exception as e:
        return jsonify({"error": f"Backend error: {str(e)}"}), 502
    session["current_prediction"] = result
    return jsonify(result)


@app.route("/chat", methods=["POST"])
def chat():
    if not OPENROUTER_API_KEY:
        return jsonify({"reply": "API key not configured."}), 200

    try:
        data = request.get_json()
        user_message = data.get("message", "").strip()
        history = data.get("history", [])

        if not user_message:
            return jsonify({"error": "Empty message"}), 400

        prediction = session.get("current_prediction")
        context = ""
        if prediction:
            context = (
                f"\n\n[Current patient result: {prediction['prediction']} "
                f"with {prediction['confidence']*100:.1f}% confidence]"
            )

        messages = [{"role": "system", "content": SYSTEM_PROMPT + context}]
        for msg in history[-6:]:
            messages.append({"role": msg["role"] if msg["role"] == "user" else "assistant", "content": msg["content"]})
        messages.append({"role": "user", "content": user_message})

        response = http_requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                "Content-Type": "application/json",
            },
            json={"model": "openai/gpt-3.5-turbo", "messages": messages},
            timeout=30,
        )
        response.raise_for_status()
        reply = response.json()["choices"][0]["message"]["content"]
        return jsonify({"reply": reply})
    except Exception as e:
        return jsonify({"reply": f"Assistant error: {str(e)}"}), 200


if __name__ == "__main__":
    app.run(debug=True)
