import requests
import json
import numpy as np
from PIL import Image
import os

BASE_URL = "http://localhost:8000/api"

# --- Health Check ---
print("=" * 45)
print("TEST 1: Health Check")
r = requests.get(f"{BASE_URL}/health/")
print(f"  Status : {r.status_code}")
print(f"  Body   : {json.dumps(r.json(), indent=4)}")

# --- Create dummy X-ray image ---
img = Image.fromarray(np.random.randint(30, 220, (512, 512, 3), dtype=np.uint8))
img.save("test_xray.jpg")

# --- Predict ---
print("\nTEST 2: Predict (real model inference)")
with open("test_xray.jpg", "rb") as f:
    r = requests.post(f"{BASE_URL}/predict/", files={"image": ("test_xray.jpg", f, "image/jpeg")}, timeout=120)
print(f"  Status : {r.status_code}")
print(f"  Body   : {json.dumps(r.json(), indent=4)}")

# --- Error: no image ---
print("\nTEST 3: Error - no image uploaded")
r = requests.post(f"{BASE_URL}/predict/")
print(f"  Status : {r.status_code} (expected 400)")
print(f"  Body   : {r.json()}")

# --- Error: wrong file type ---
print("\nTEST 4: Error - invalid file type")
with open("test_xray.jpg", "rb") as f:
    r = requests.post(f"{BASE_URL}/predict/", files={"image": ("file.txt", f, "text/plain")})
print(f"  Status : {r.status_code} (expected 400)")
print(f"  Body   : {r.json()}")

os.remove("test_xray.jpg")
print("\n" + "=" * 45)
print("All tests done.")
