"""OpenAI Chat pipeline - GPT-powered estimate generation and conversation."""
import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

SYSTEM_PROMPT = """You are HomeEstimator AI, an expert assistant for home service cost estimation.
You help homeowners understand their repair/maintenance needs and provide helpful guidance.

When given analysis results from our CV and NLP models, you should:
1. Explain the detected issue in plain language
2. Provide a clear cost estimate range with context
3. Give practical recommendations
4. Suggest clear next steps
5. Mention any safety concerns if applicable

Be concise, friendly, and professional. Use bullet points for readability.
Always mention that estimates are approximate and actual costs may vary by region and contractor."""


def generate_smart_estimate(estimate_data, entities, original_text="", image_category=""):
    """Use GPT to generate a natural language estimate with recommendations.

    Args:
        estimate_data: dict from the pricing engine
        entities: dict from entity extraction
        original_text: the user's original description
        image_category: what the CV model detected

    Returns:
        str: GPT-generated response
    """
    user_message = f"""Based on the following analysis, provide a helpful estimate and recommendations:

**User's Description**: {original_text if original_text else 'No text provided'}
**Image Analysis**: Detected category: {image_category if image_category else 'No image provided'}
**Final Assessment**:
- Category: {estimate_data.get('category', 'Unknown')}
- Urgency: {estimate_data.get('urgency', 'Unknown')}
- Scope: {estimate_data.get('scope', 'Unknown')}
- Estimated Cost: ${estimate_data.get('price_low', 'N/A')} - ${estimate_data.get('price_high', 'N/A')}
- Typical Tasks: {estimate_data.get('typical_tasks', 'N/A')}

**Extracted Details**:
- Measurements: {', '.join(entities.get('measurements', [])) or 'None detected'}
- Materials: {', '.join(entities.get('materials', [])) or 'None detected'}
- Locations: {', '.join(entities.get('locations', [])) or 'None detected'}

Please provide a clear, helpful response to the homeowner."""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
        max_tokens=500,
        temperature=0.7,
    )

    return response.choices[0].message.content


def chat_followup(conversation_history, user_message):
    """Handle follow-up questions in a conversation.

    Args:
        conversation_history: list of {"role": ..., "content": ...} dicts
        user_message: the new user message

    Returns:
        str: GPT response
    """
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages.extend(conversation_history)
    messages.append({"role": "user", "content": user_message})

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        max_tokens=400,
        temperature=0.7,
    )

    return response.choices[0].message.content
