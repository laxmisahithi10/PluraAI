"""
Full end-to-end test:
  1. Health check
  2. Predict — real model inference + MongoDB save
  3. Verify saved record in MongoDB
  4. History endpoint
  5. Latest endpoint
  6. Error: no image
  7. Error: wrong file type
"""

import requests
import json
import os
import sys
import django
import numpy as np
from PIL import Image

# Setup Django for direct MongoDB verification
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'backend.settings')
sys.path.insert(0, '.')
django.setup()

from utils.mongo import get_all_predictions

BASE = "http://localhost:8000/api"
PASS = "PASS"
FAIL = "FAIL"

def banner(title):
    print(f"\n{'='*55}")
    print(f"  {title}")
    print(f"{'='*55}")

def result(label, ok, detail=""):
    icon = "[PASS]" if ok else "[FAIL]"
    print(f"  {icon} {label}")
    if detail:
        print(f"     {detail}")

# ── 1. Health Check ────────────────────────────────────────
banner("TEST 1: Health Check  GET /api/health/")
r = requests.get(f"{BASE}/health/")
ok = r.status_code == 200 and r.json().get("status") == "healthy"
result(f"Status {r.status_code}", ok)
result("Response body", ok, json.dumps(r.json()))

# ── 2. Predict (real model + MongoDB) ─────────────────────
banner("TEST 2: Predict  POST /api/predict/")
img = Image.fromarray(np.random.randint(30, 220, (512, 512, 3), dtype=np.uint8))
img.save("test_xray.jpg")

with open("test_xray.jpg", "rb") as f:
    r = requests.post(f"{BASE}/predict/",
                      files={"image": ("test_xray.jpg", f, "image/jpeg")},
                      timeout=120)

os.remove("test_xray.jpg")

ok = r.status_code == 200
result(f"Status {r.status_code}", ok)

if ok:
    body = r.json()
    result("prediction field",    body.get("prediction") in ["Normal", "Pneumonia"],
           f"prediction = {body.get('prediction')}")
    result("confidence field",    0 <= body.get("confidence", -1) <= 1,
           f"confidence = {body.get('confidence')}")
    result("confidence_percent",  body.get("confidence_percent") is not None,
           f"confidence_percent = {body.get('confidence_percent')}%")
    result("explanation field",   bool(body.get("explanation")),
           f"explanation = {body.get('explanation')[:80]}...")
    result("case_id field",       bool(body.get("case_id")),
           f"case_id = {body.get('case_id')}")
    result("inference_time field",body.get("inference_time") is not None,
           f"inference_time = {body.get('inference_time')}s")
    saved_case_id = body.get("case_id")
else:
    print(f"     Response: {r.text}")
    saved_case_id = None

# ── 3. Verify record saved in MongoDB ─────────────────────
banner("TEST 3: MongoDB Verification")
all_preds = get_all_predictions()
if saved_case_id:
    match = next((p for p in all_preds if p.get("case_id") == saved_case_id), None)
    result("Record saved in MongoDB", match is not None,
           f"case_id = {saved_case_id}")
    if match:
        result("model_version stored", match.get("model_version") == "ViT_CALSL_v1",
               f"model_version = {match.get('model_version')}")
        result("timestamp stored",     bool(match.get("timestamp")),
               f"timestamp = {match.get('timestamp')}")
        result("confidence_percent",   match.get("confidence_percent") is not None,
               f"confidence_percent = {match.get('confidence_percent')}%")
else:
    result("Skipped — no case_id from predict", False)

# ── 4. History endpoint ────────────────────────────────────
banner("TEST 4: All Predictions  GET /api/history/")
r = requests.get(f"{BASE}/history/")
ok = r.status_code == 200
result(f"Status {r.status_code}", ok)
if ok:
    body = r.json()
    result("count field present", "count" in body, f"count = {body.get('count')}")
    result("predictions list",    isinstance(body.get("predictions"), list),
           f"total records = {len(body.get('predictions', []))}")

# ── 5. Latest endpoint ─────────────────────────────────────
banner("TEST 5: Latest 10  GET /api/history/latest/")
r = requests.get(f"{BASE}/history/latest/")
ok = r.status_code == 200
result(f"Status {r.status_code}", ok)
if ok:
    body = r.json()
    result("max 10 records returned", len(body.get("predictions", [])) <= 10,
           f"returned = {len(body.get('predictions', []))} records")

# ── 6. Error: no image ─────────────────────────────────────
banner("TEST 6: Error Handling — No Image")
r = requests.post(f"{BASE}/predict/")
result(f"Status {r.status_code} (expected 400)", r.status_code == 400,
       f"error = {r.json().get('error')}")

# ── 7. Error: wrong file type ──────────────────────────────
banner("TEST 7: Error Handling — Wrong File Type")
with open("test_xray.jpg", "w") as f:
    f.write("not an image")
with open("test_xray.jpg", "rb") as f:
    r = requests.post(f"{BASE}/predict/",
                      files={"image": ("file.txt", f, "text/plain")})
os.remove("test_xray.jpg")
result(f"Status {r.status_code} (expected 400)", r.status_code == 400,
       f"error = {r.json().get('error')}")

# ── Summary ────────────────────────────────────────────────
banner("ALL TESTS COMPLETE")
