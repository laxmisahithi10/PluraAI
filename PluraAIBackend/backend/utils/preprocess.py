import cv2
import numpy as np
import torch

IMG_SIZE = 224
PATCH_SIZE = 16


def _generate_patches(img):
    """Returns (196, 768) float32 numpy array."""
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE)).astype(np.float32) / 255.0
    patches = (
        img.reshape(IMG_SIZE // PATCH_SIZE, PATCH_SIZE, IMG_SIZE // PATCH_SIZE, PATCH_SIZE, 3)
        .transpose(0, 2, 1, 3, 4)
        .reshape(-1, PATCH_SIZE * PATCH_SIZE * 3)
    )
    return patches


def preprocess_image(image_path):
    """
    Load image and return three (1, 196, 768) tensors:
    full image, left lung, right lung.
    """
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError("Could not load image")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    h, w = img.shape[:2]
    left_img = img[:, : w // 2]
    right_img = img[:, w // 2 :]

    full_t = torch.tensor(_generate_patches(img), dtype=torch.float32).unsqueeze(0)
    left_t = torch.tensor(_generate_patches(left_img), dtype=torch.float32).unsqueeze(0)
    right_t = torch.tensor(_generate_patches(right_img), dtype=torch.float32).unsqueeze(0)

    return full_t, left_t, right_t
