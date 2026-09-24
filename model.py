import torch
import torch.nn as nn
import torch.nn.functional as F


class TransformerEncoder(nn.Module):
    def __init__(self, embed_dim=768, num_heads=12, depth=6, mlp_dim=2048, dropout=0.1):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=mlp_dim,
            dropout=dropout,
            norm_first=True,       # Pre-LN: faster + more stable training
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=depth)

    def forward(self, x):
        return self.encoder(x)


class CALSLModel(nn.Module):
    def __init__(self, num_classes=2, embed_dim=768, num_patches=196):
        super().__init__()

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_patches + 1, embed_dim)
        )

        self.encoder = TransformerEncoder(embed_dim=embed_dim)

        self.classifier = nn.Linear(embed_dim, num_classes)

    def forward(self, full, left, right):
        # Stack all 3 views and run ONE batched encoder pass (3x faster)
        B = full.size(0)
        combined = torch.cat([full, left, right], dim=0)  # (3B, 197, 768)
        cls_tokens = self.cls_token.expand(combined.size(0), -1, -1)
        combined = torch.cat([cls_tokens, combined], dim=1)
        combined = combined + self.pos_embed
        combined = self.encoder(combined)
        cls_out = combined[:, 0]  # (3B, 768)

        f_full, f_left, f_right = cls_out.split(B, dim=0)

        logits = self.classifier(f_full)
        cal_loss = 1 - F.cosine_similarity(f_left, f_right).mean()

        return logits, cal_loss
