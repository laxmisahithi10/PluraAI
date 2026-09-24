"""
utils/mongo.py
--------------
MongoDB Atlas connection using PyMongo.

Why singleton?
- MongoClient is thread-safe and designed to be reused.
- Creating a new connection per request is expensive (TCP handshake + TLS).
- One client instance handles a connection pool internally — reusing it
  means zero reconnection overhead on every API call.
"""

import uuid
import datetime
import logging
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure, PyMongoError
from django.conf import settings

logger = logging.getLogger(__name__)

_client = None


def get_db():
    """Return plura_db, creating the MongoClient once (singleton)."""
    global _client
    if _client is None:
        _client = MongoClient(settings.MONGO_URI, serverSelectionTimeoutMS=5000)
    return _client[settings.MONGO_DB_NAME]


def save_prediction(prediction, confidence, explanation):
    """
    Insert one prediction document into the predictions collection.

    Document shape:
    {
        "case_id":            "uuid4 string",
        "prediction":         "Pneumonia" | "Normal",
        "confidence":         0.97,
        "confidence_percent": 97.0,
        "explanation":        "...",
        "timestamp":          "2024-01-01T00:00:00.000000",
        "model_version":      "ViT_CALSL_v1"
    }

    Returns case_id on success, None on failure.
    API still returns prediction even if MongoDB is down (safe fallback).
    """
    try:
        db = get_db()
        case_id = str(uuid.uuid4())
        record = {
            "case_id": case_id,
            "prediction": prediction,
            "confidence": confidence,
            "confidence_percent": round(confidence * 100, 2),
            "explanation": explanation,
            "timestamp": datetime.datetime.utcnow().isoformat(),
            "model_version": "ViT_CALSL_v1",
        }
        db.predictions.insert_one(record)
        logger.info(f"Saved prediction {case_id} to MongoDB")
        return case_id
    except (ConnectionFailure, PyMongoError) as e:
        logger.error(f"MongoDB save failed: {str(e)}")
        return None


def get_all_predictions():
    """Return all predictions, newest first, without internal _id field."""
    try:
        db = get_db()
        return list(db.predictions.find({}, {"_id": 0}).sort("timestamp", -1))
    except (ConnectionFailure, PyMongoError) as e:
        logger.error(f"MongoDB fetch failed: {str(e)}")
        return []


def get_latest_predictions(limit=10):
    """Return the latest N predictions, newest first."""
    try:
        db = get_db()
        return list(db.predictions.find({}, {"_id": 0}).sort("timestamp", -1).limit(limit))
    except (ConnectionFailure, PyMongoError) as e:
        logger.error(f"MongoDB fetch failed: {str(e)}")
        return []
