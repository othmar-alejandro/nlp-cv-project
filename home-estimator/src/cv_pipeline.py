"""Computer Vision pipeline - MobileNetV2 transfer learning for job image classification."""
import os
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

from src.utils import CATEGORIES, MODEL_DIR

# Image preprocessing (ImageNet normalization)
TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def build_model(num_classes=6):
    """Build MobileNetV2 with frozen backbone and custom classifier head."""
    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
    # Freeze all backbone layers
    for param in model.features.parameters():
        param.requires_grad = False
    # Replace classifier head
    model.classifier = nn.Sequential(
        nn.Dropout(0.2),
        nn.Linear(1280, num_classes),
    )
    return model


def load_model(model_path=None):
    """Load trained image classification model."""
    if model_path is None:
        model_path = os.path.join(MODEL_DIR, "image_classifier.pth")
    model = build_model(num_classes=len(CATEGORIES))
    model.load_state_dict(torch.load(model_path, map_location="cpu", weights_only=True))
    model.eval()
    return model


def predict_image(model, image):
    """Predict job category from an image.

    Args:
        model: Trained MobileNetV2 model
        image: PIL Image or file path

    Returns:
        dict with 'category', 'confidence', and 'probabilities'
    """
    if isinstance(image, str):
        image = Image.open(image).convert("RGB")
    elif not isinstance(image, Image.Image):
        image = Image.open(image).convert("RGB")
    else:
        image = image.convert("RGB")

    input_tensor = TRANSFORM(image).unsqueeze(0)

    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = torch.softmax(outputs, dim=1)[0]

    top_idx = probabilities.argmax().item()
    return {
        "category": CATEGORIES[top_idx],
        "confidence": probabilities[top_idx].item(),
        "probabilities": {CATEGORIES[i]: probabilities[i].item() for i in range(len(CATEGORIES))},
    }
