import torch
import torch.nn as nn
import torch.nn.functional as F
from django.conf import settings


class TransformerEncoder(nn.Module):
    def __init__(self, embed_dim=768, num_heads=12, depth=6, mlp_dim=2048, dropout=0.1):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=mlp_dim,
            dropout=dropout,
            norm_first=True,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=depth)

    def forward(self, x):
        return self.encoder(x)


class CALSLModel(nn.Module):
    def __init__(self, num_classes=2, embed_dim=768, num_patches=196):
        super().__init__()
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.encoder = TransformerEncoder(embed_dim=embed_dim)
        self.classifier = nn.Linear(embed_dim, num_classes)

    def forward(self, full, left, right):
        B = full.size(0)
        combined = torch.cat([full, left, right], dim=0)
        cls_tokens = self.cls_token.expand(combined.size(0), -1, -1)
        combined = torch.cat([cls_tokens, combined], dim=1)
        combined = combined + self.pos_embed
        combined = self.encoder(combined)
        cls_out = combined[:, 0]
        f_full, f_left, f_right = cls_out.split(B, dim=0)
        logits = self.classifier(f_full)
        cal_loss = 1 - F.cosine_similarity(f_left, f_right).mean()
        return logits, cal_loss


_model_instance = None


def get_model():
    global _model_instance
    if _model_instance is None:
        model = CALSLModel(num_classes=2)
        checkpoint = torch.load(settings.MODEL_PATH, map_location="cpu", weights_only=True)
        state_dict = checkpoint.get("model_state_dict", checkpoint)
        model.load_state_dict(state_dict)
        model.eval()
        _model_instance = model
    return _model_instance


CLASSES = ["Normal", "Pneumonia"]
TEMPERATURE = 3.5


def predict(full_tensor, left_tensor, right_tensor):
    """
    Accepts three (1, 196, 768) float32 tensors.
    Returns (prediction_label, confidence_float).
    """
    model = get_model()
    with torch.no_grad():
        logits, _ = model(full_tensor, left_tensor, right_tensor)
        print(f"[DEBUG] Raw logits: Normal={logits[0][0].item():.4f}, Pneumonia={logits[0][1].item():.4f}")
        scaled_logits = logits / TEMPERATURE
        probs = torch.softmax(scaled_logits, dim=1)
        print(f"[DEBUG] Probs: Normal={probs[0][0].item():.4f}, Pneumonia={probs[0][1].item():.4f}")
        pneumonia_prob = probs[0][1].item()
        normal_prob = probs[0][0].item()
        # Lower threshold for Pneumonia to fix bias toward Normal
        if pneumonia_prob >= 0.35:
            class_idx = 1
            confidence_val = min(round(pneumonia_prob, 4), 0.97)
        else:
            class_idx = 0
            confidence_val = min(round(normal_prob, 4), 0.97)
        print(f"[DEBUG] Predicted: {CLASSES[class_idx]} | Confidence: {confidence_val}")
    return CLASSES[class_idx], confidence_val
