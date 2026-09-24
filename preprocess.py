import cv2
import numpy as np

IMG_SIZE = 224
PATCH_SIZE = 16
EMBED_DIM = PATCH_SIZE * PATCH_SIZE * 3  # 768


def preprocess_image(image_path):
    """
    Load and preprocess image from path
    """
    img = cv2.imread(image_path)
    if img is None:
        return None
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def split_lungs(img):
    """
    Split image into left and right lung regions
    Simple implementation: split vertically in half
    """
    h, w = img.shape[:2]
    left_img = img[:, :w//2]
    right_img = img[:, w//2:]
    return left_img, right_img


def generate_patches(img):
    """
    img: H×W×3 image
    returns: (196, 768)
    """
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img.astype(np.float32) / 255.0
    # vectorized patch extraction via reshape — no Python loop
    patches = (img
               .reshape(IMG_SIZE // PATCH_SIZE, PATCH_SIZE,
                        IMG_SIZE // PATCH_SIZE, PATCH_SIZE, 3)
               .transpose(0, 2, 1, 3, 4)
               .reshape(-1, PATCH_SIZE * PATCH_SIZE * 3))  # (196, 768)
    return patches
