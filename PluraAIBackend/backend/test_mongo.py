import django
import os
import sys

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'backend.settings')
sys.path.insert(0, '.')
django.setup()

from utils.mongo import save_prediction, get_all_predictions, get_latest_predictions

print("=" * 50)
print("TEST: MongoDB Atlas Integration")
print("=" * 50)

# Insert a test record
print("\n1. Saving test prediction to MongoDB Atlas...")
case_id = save_prediction(
    prediction="Pneumonia",
    confidence=0.97,
    explanation="Test: Abnormal patterns detected in chest X-ray."
)
if case_id:
    print(f"   SUCCESS — case_id: {case_id}")
else:
    print("   FAILED — check MongoDB URI or network")

# Fetch latest
print("\n2. Fetching latest 3 predictions...")
latest = get_latest_predictions(limit=3)
for i, doc in enumerate(latest, 1):
    print(f"   [{i}] case_id={doc['case_id']} | {doc['prediction']} | {doc['confidence_percent']}% | {doc['timestamp']}")

# Fetch all count
print("\n3. Total predictions in collection...")
all_preds = get_all_predictions()
print(f"   Total: {len(all_preds)} records")

print("\n" + "=" * 50)
print("MongoDB integration test complete.")
