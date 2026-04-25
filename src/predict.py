"""
Inference utilities used by the Streamlit app.

Given a PIL image and a loaded model, returns the top prediction + confidences,
plus a Grad-CAM heatmap for the predicted class.
"""

from typing import List, Tuple
import torch
import torch.nn.functional as F
from PIL import Image

from .dataset import build_eval_transform
from .gradcam import GradCAM, overlay_heatmap_on_image


def preprocess_image(pil_image: Image.Image, device: str = "cpu") -> torch.Tensor:
    """PIL image -> normalized (1, 3, 224, 224) tensor on device."""
    transform = build_eval_transform()
    tensor = transform(pil_image.convert("RGB")).unsqueeze(0)
    return tensor.to(device)


@torch.no_grad()
def predict_topk(
    model, pil_image: Image.Image, class_names: List[str], k: int = 3, device: str = "cpu"
) -> List[Tuple[str, float]]:
    """Return top-k (class_name, probability) tuples."""
    x = preprocess_image(pil_image, device=device)
    logits = model(x)
    probs = F.softmax(logits, dim=1)[0]
    top_probs, top_idx = probs.topk(min(k, len(class_names)))
    return [(class_names[i], float(p)) for i, p in zip(top_idx.cpu().numpy(), top_probs.cpu().numpy())]


def predict_with_gradcam(
    model, pil_image: Image.Image, class_names: List[str], device: str = "cpu"
):
    """Return (top1_class, top1_prob, cam_overlay_image).

    cam_overlay is a PIL Image with the Grad-CAM heatmap blended on the input.
    """
    x = preprocess_image(pil_image, device=device)

    # Top-1 prediction
    with torch.no_grad():
        logits = model(x)
        probs = F.softmax(logits, dim=1)[0]
        top_idx = int(probs.argmax().item())
        top_prob = float(probs[top_idx].item())

    # Grad-CAM needs gradients, so we re-run without torch.no_grad().
    # For ResNet-50 our custom head is model.fc; layer4 is still the last conv stage.
    cam = GradCAM(model, target_layer=model.layer4)
    heatmap = cam.generate(x, target_class=top_idx)
    overlay = overlay_heatmap_on_image(pil_image, heatmap)

    return class_names[top_idx], top_prob, overlay
