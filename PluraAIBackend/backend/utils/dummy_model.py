import numpy as np
import random

class DummyModel:
    """Dummy model for testing inference pipeline before ViT+CALSL is ready"""
    
    def __init__(self):
        self.classes = ["Normal", "Pneumonia"]
    
    def predict(self, image_tensor):
        """
        Simulate model prediction
        Returns: (prediction_class, confidence_score)
        """
        # Simulate random prediction for now
        prediction = random.choice(self.classes)
        confidence = round(random.uniform(0.6, 0.95), 3)
        
        return prediction, confidence