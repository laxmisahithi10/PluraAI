from django.urls import path
from .views import PredictAPIView, HealthCheckAPIView, PredictionHistoryAPIView, LatestPredictionsAPIView

urlpatterns = [
    path('health/', HealthCheckAPIView.as_view(), name='health'),
    path('predict/', PredictAPIView.as_view(), name='predict'),
    path('history/', PredictionHistoryAPIView.as_view(), name='history'),
    path('history/latest/', LatestPredictionsAPIView.as_view(), name='latest'),
]
