import django
import os
import sys
import traceback

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'backend.settings')
sys.path.insert(0, '.')
django.setup()

from PIL import Image
import numpy as np

try:
    from utils.preprocess import preprocess_image
    print("preprocess imported OK")

    from utils import real_model
    print("real_model imported OK")

    img = Image.fromarray(np.random.randint(30, 220, (512, 512, 3), dtype=np.uint8))
    img.save("test_xray.jpg")
    print("test image saved")

    full_t, left_t, right_t = preprocess_image("test_xray.jpg")
    print("Shapes:", full_t.shape, left_t.shape, right_t.shape)

    pred, conf = real_model.predict(full_t, left_t, right_t)
    print("Prediction:", pred, "| Confidence:", conf)

except Exception:
    traceback.print_exc()
finally:
    if os.path.exists("test_xray.jpg"):
        os.remove("test_xray.jpg")
