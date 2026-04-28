"""Estimation engine - fuses CV and NLP predictions into actionable output."""
import os
import pandas as pd

from src.utils import PRICING_DIR


def load_pricing_table():
    """Load the pricing reference table."""
    return pd.read_csv(os.path.join(PRICING_DIR, "pricing_table.csv"))


def fuse_predictions(cv_result, nlp_result, entities):
    """Combine CV and NLP predictions into a final assessment.

    Strategy:
    - If both agree on category: high confidence, use that category
    - If they disagree: weight NLP higher (text is usually more specific)
    - Urgency comes from NLP (text carries urgency cues)
    - Scope is inferred from entities (measurements, quantities)
    """
    cv_cat = cv_result["category"]
    nlp_cat = nlp_result["category"]
    cv_conf = cv_result["confidence"]
    nlp_conf = nlp_result["category_confidence"]

    # Category fusion
    if cv_cat == nlp_cat:
        final_category = cv_cat
        agreement = True
        combined_confidence = (cv_conf + nlp_conf) / 2
    else:
        # Weight NLP 60%, CV 40% — text is typically more specific
        if nlp_conf * 0.6 >= cv_conf * 0.4:
            final_category = nlp_cat
        else:
            final_category = cv_cat
        agreement = False
        combined_confidence = max(cv_conf, nlp_conf) * 0.8  # Reduced confidence on disagreement

    # Urgency from NLP
    urgency = nlp_result["urgency"]

    # Scope inference from entities
    scope = infer_scope(entities)

    return {
        "category": final_category,
        "urgency": urgency,
        "scope": scope,
        "confidence": combined_confidence,
        "agreement": agreement,
        "cv_category": cv_cat,
        "nlp_category": nlp_cat,
    }


def infer_scope(entities):
    """Infer job scope from extracted entities."""
    score = 0

    # Measurements suggest larger scope
    if entities.get("measurements"):
        for m in entities["measurements"]:
            import re
            nums = re.findall(r'\d+', m)
            if nums:
                val = max(int(n) for n in nums)
                if val > 500:
                    score += 3
                elif val > 100:
                    score += 2
                else:
                    score += 1

    # Multiple quantities suggest larger scope
    if entities.get("quantities"):
        total = sum(int(q) for q in entities["quantities"] if q.isdigit())
        if total > 5:
            score += 2
        elif total > 2:
            score += 1

    # Multiple locations suggest larger scope
    if entities.get("locations"):
        if len(entities["locations"]) > 3:
            score += 2
        elif len(entities["locations"]) > 1:
            score += 1

    if score >= 4:
        return "large"
    elif score >= 2:
        return "medium"
    return "small"


def generate_estimate(fused_result, pricing_table):
    """Look up pricing and generate the final estimate output."""
    cat = fused_result["category"]
    urg = fused_result["urgency"]
    scope = fused_result["scope"]

    # Find matching row in pricing table
    match = pricing_table[
        (pricing_table["category"] == cat)
        & (pricing_table["urgency"] == urg)
        & (pricing_table["scope"] == scope)
    ]

    if match.empty:
        # Fallback: match category and urgency, any scope
        match = pricing_table[
            (pricing_table["category"] == cat) & (pricing_table["urgency"] == urg)
        ]
    if match.empty:
        # Fallback: match category only
        match = pricing_table[pricing_table["category"] == cat]

    if match.empty:
        return {
            "price_low": "N/A",
            "price_high": "N/A",
            "typical_tasks": "Unable to determine",
            "recommendations": "Please provide more details for an accurate estimate.",
            "next_steps": "Contact a professional for an on-site assessment.",
        }

    row = match.iloc[0]
    return {
        "price_low": int(row["price_low"]),
        "price_high": int(row["price_high"]),
        "typical_tasks": row["typical_tasks"],
        "recommendations": row["recommendations"],
        "next_steps": row["next_steps"],
        "category": fused_result["category"],
        "urgency": fused_result["urgency"],
        "scope": fused_result["scope"],
        "confidence": fused_result["confidence"],
        "agreement": fused_result["agreement"],
    }
