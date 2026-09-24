from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from utils.preprocess import preprocess_image
from utils import real_model
from utils.mongo import save_prediction, get_all_predictions, get_latest_predictions
import os
import time
import logging

logger = logging.getLogger(__name__)


def generate_explanation(prediction, confidence):
    """Generate a human-readable explanation based on prediction and confidence."""
    percent = round(confidence * 100, 2)
    if prediction == "Pneumonia":
        return (
            f"The model detected signs of Pneumonia with {percent}% confidence. "
            "Abnormal patterns were identified in the chest X-ray that are consistent "
            "with pneumonia. Please consult a medical professional for confirmation."
        )
    return (
        f"The model found no signs of Pneumonia with {percent}% confidence. "
        "The chest X-ray appears normal. Regular check-ups are still recommended."
    )


class HealthCheckAPIView(APIView):
    def get(self, request):
        return Response({
            "status": "healthy",
            "message": "PluraAI Backend API is running",
            "timestamp": time.time()
        })


class PredictAPIView(APIView):
    def post(self, request):
        try:
            if "image" not in request.FILES:
                return Response({"error": "No image uploaded"}, status=status.HTTP_400_BAD_REQUEST)

            image = request.FILES["image"]

            if not image.name.lower().endswith((".png", ".jpg", ".jpeg")):
                return Response({"error": "Only JPG and PNG images are allowed"}, status=status.HTTP_400_BAD_REQUEST)

            if image.size > 10 * 1024 * 1024:
                return Response({"error": "Image size must be less than 10MB"}, status=status.HTTP_400_BAD_REQUEST)

            start_time = time.time()

            uploads_dir = "uploads"
            os.makedirs(uploads_dir, exist_ok=True)
            image_path = os.path.join(uploads_dir, f"{int(time.time())}_{image.name}")

            with open(image_path, "wb+") as f:
                for chunk in image.chunks():
                    f.write(chunk)

            try:
                full_t, left_t, right_t = preprocess_image(image_path)
            except ValueError as e:
                return Response({"error": f"Image preprocessing failed: {str(e)}"}, status=status.HTTP_400_BAD_REQUEST)
            finally:
                if os.path.exists(image_path):
                    os.remove(image_path)

            prediction, confidence = real_model.predict(full_t, left_t, right_t)
            explanation = generate_explanation(prediction, confidence)

            # Save to MongoDB — safe fallback: API still responds even if DB fails
            case_id = save_prediction(prediction, confidence, explanation)

            return Response({
                "prediction": prediction,
                "confidence": confidence,
                "confidence_percent": round(confidence * 100, 1),
                "explanation": explanation,
                "case_id": case_id,
                "inference_time": round(time.time() - start_time, 3),
                "status": "success"
            })

        except Exception as e:
            logger.error(f"Prediction API error: {str(e)}")
            return Response({"error": "Internal server error occurred"}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


class PredictionHistoryAPIView(APIView):
    """GET /api/history/ — all predictions, newest first."""
    def get(self, request):
        data = get_all_predictions()
        return Response({"count": len(data), "predictions": data})


class LatestPredictionsAPIView(APIView):
    """GET /api/history/latest/ — latest 10 predictions."""
    def get(self, request):
        data = get_latest_predictions(limit=10)
        return Response({"count": len(data), "predictions": data})
