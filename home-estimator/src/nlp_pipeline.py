"""NLP pipeline - TF-IDF + Logistic Regression for text classification + entity extraction."""
import os
import re
import joblib
import spacy

from src.utils import CATEGORIES, URGENCY_LEVELS, MODEL_DIR

nlp = spacy.load("en_core_web_sm")


def load_models():
    """Load trained text classification models."""
    category_model = joblib.load(os.path.join(MODEL_DIR, "text_category_model.joblib"))
    urgency_model = joblib.load(os.path.join(MODEL_DIR, "text_urgency_model.joblib"))
    tfidf = joblib.load(os.path.join(MODEL_DIR, "tfidf_vectorizer.joblib"))
    return category_model, urgency_model, tfidf


def predict_text(category_model, urgency_model, tfidf, text):
    """Predict category and urgency from text description.

    Returns:
        dict with 'category', 'category_confidence', 'urgency', 'urgency_confidence'
    """
    text_vec = tfidf.transform([text])

    cat_probs = category_model.predict_proba(text_vec)[0]
    cat_idx = cat_probs.argmax()
    cat_classes = category_model.classes_

    urg_probs = urgency_model.predict_proba(text_vec)[0]
    urg_idx = urg_probs.argmax()
    urg_classes = urgency_model.classes_

    return {
        "category": cat_classes[cat_idx],
        "category_confidence": cat_probs[cat_idx],
        "category_probabilities": {cat_classes[i]: cat_probs[i] for i in range(len(cat_classes))},
        "urgency": urg_classes[urg_idx],
        "urgency_confidence": urg_probs[urg_idx],
    }


def extract_entities(text):
    """Extract relevant entities from job description using spaCy + regex."""
    doc = nlp(text)
    entities = {
        "measurements": [],
        "materials": [],
        "locations": [],
        "quantities": [],
    }

    # Regex patterns for measurements
    measurement_patterns = [
        r'\d+\s*(?:sq\.?\s*ft\.?|square\s*feet|sqft)',
        r'\d+\s*x\s*\d+\s*(?:feet|foot|ft)?',
        r'\d+\s*(?:feet|foot|ft|inches|inch|in)\b',
        r'\d+\s*(?:linear\s*feet|linear\s*ft)',
    ]
    for pattern in measurement_patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        entities["measurements"].extend(matches)

    # Materials
    material_keywords = [
        "copper", "pvc", "galvanized", "cast iron", "wood", "vinyl", "asphalt",
        "metal", "rubber", "concrete", "brick", "stucco", "drywall", "shingle",
        "tile", "hardwood", "laminate", "epoxy", "led", "aluminum",
    ]
    text_lower = text.lower()
    for mat in material_keywords:
        if mat in text_lower:
            entities["materials"].append(mat)

    # Room/location mentions
    location_keywords = [
        "kitchen", "bathroom", "bedroom", "living room", "basement", "attic",
        "garage", "laundry", "dining room", "hallway", "porch", "deck",
        "crawl space", "yard", "exterior", "interior", "roof", "ceiling",
        "wall", "floor", "front", "back", "upstairs", "downstairs",
    ]
    for loc in location_keywords:
        if loc in text_lower:
            entities["locations"].append(loc)

    # Quantities
    quantity_patterns = [
        r'(\d+)\s*(?:rooms?|windows?|doors?|outlets?|fixtures?|faucets?|toilets?|vents?|lights?)',
    ]
    for pattern in quantity_patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        entities["quantities"].extend(matches)

    return entities
