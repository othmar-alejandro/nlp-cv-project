"""Shared utilities for HomeEstimator AI."""
import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
IMAGE_DIR = os.path.join(DATA_DIR, "images")
TEXT_DIR = os.path.join(DATA_DIR, "texts")
PRICING_DIR = os.path.join(DATA_DIR, "pricing")
MODEL_DIR = os.path.join(PROJECT_ROOT, "models")

CATEGORIES = ["plumbing", "painting", "roofing", "electrical", "hvac", "general_repair"]
URGENCY_LEVELS = ["low", "medium", "high", "emergency"]
SCOPE_LEVELS = ["small", "medium", "large"]

CATEGORY_LABELS = {i: cat for i, cat in enumerate(CATEGORIES)}
URGENCY_LABELS = {i: urg for i, urg in enumerate(URGENCY_LEVELS)}
