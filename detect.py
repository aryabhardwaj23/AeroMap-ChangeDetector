"""
Change detection using pretrained ResNet18 feature diff.
No training needed — transfer learning zero-shot approach.
"""

import io
import numpy as np
import cv2
import torch
import torchvision.models as models
import torchvision.transforms as T
from PIL import Image

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

_transform = T.Compose([
    T.Resize((256, 256)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]),
])


def _load_encoder():
    net = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    encoder = torch.nn.Sequential(*list(net.children())[:-3])  # up to layer3
    encoder.eval().to(DEVICE)
    return encoder


_encoder = None


def _get_encoder():
    global _encoder
    if _encoder is None:
        _encoder = _load_encoder()
    return _encoder


def predict(img_a: Image.Image, img_b: Image.Image,
            threshold: float | None = None) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns:
        mask_bin  : (H, W) uint8  binary change mask (255 = change)
        heat      : (H, W, 3) uint8  BGR heatmap for display

    threshold=None uses Otsu auto-threshold (recommended).
    """
    # Ensure both images are the same size before diffing
    if img_a.size != img_b.size:
        img_b = img_b.resize(img_a.size, Image.BILINEAR)

    a = np.array(img_a).astype(np.float32)
    b = np.array(img_b).astype(np.float32)

    # Per-channel absolute difference → mean across channels
    diff = np.abs(a - b).mean(axis=2).astype(np.uint8)

    if threshold is None:
        # Otsu finds the natural split between changed / unchanged pixels
        _, bin_mask = cv2.threshold(diff, 0, 255,
                                    cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    else:
        bin_mask = (diff > threshold * 255).astype(np.uint8) * 255

    # Morphological cleanup — close small holes, remove tiny specks
    kernel   = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    bin_mask = cv2.morphologyEx(bin_mask, cv2.MORPH_CLOSE, kernel)
    bin_mask = cv2.morphologyEx(bin_mask, cv2.MORPH_OPEN,  kernel)

    # Heat for display: normalised diff coloured with inferno
    heat_norm = (diff.astype(np.float32) / (diff.max() + 1e-6) * 255).astype(np.uint8)
    heat_bgr  = cv2.applyColorMap(heat_norm, cv2.COLORMAP_INFERNO)

    return bin_mask, heat_bgr


def overlay(img: Image.Image, mask: np.ndarray,
            color=(255, 60, 60), alpha=0.55) -> Image.Image:
    """Red overlay of change regions on the image."""
    h, w = mask.shape[:2]
    if img.size != (w, h):
        img = img.resize((w, h), Image.BILINEAR)
    base = np.array(img).copy()
    changed = mask > 0
    blend = base.copy()
    blend[changed] = (
        np.array(color, dtype=np.float32) * alpha
        + base[changed] * (1 - alpha)
    ).astype(np.uint8)
    return Image.fromarray(blend)


def heatmap_rgb(heat_bgr: np.ndarray) -> Image.Image:
    """Convert BGR heatmap array to RGB PIL image for display."""
    return Image.fromarray(cv2.cvtColor(heat_bgr, cv2.COLOR_BGR2RGB))


def changed_area_m2(mask: np.ndarray, gsd_m: float = 0.5) -> float:
    """Estimate changed area in m² using ground sampling distance."""
    px = int((mask > 0).sum())
    return px * (gsd_m ** 2)


def load_pair_from_parquet(parquet_path: str, index: int) -> tuple:
    """Load (imageA, imageB, label) from the LEVIR-CD parquet file."""
    import pyarrow.parquet as pq
    table = pq.read_table(parquet_path)
    row   = table.slice(index, 1).to_pydict()
    img_a = Image.open(io.BytesIO(row["imageA"][0]["bytes"])).convert("RGB")
    img_b = Image.open(io.BytesIO(row["imageB"][0]["bytes"])).convert("RGB")
    label = Image.open(io.BytesIO(row["label"][0]["bytes"])).convert("L")
    # Normalise to img_a size so all three arrays are aligned
    if img_b.size != img_a.size:
        img_b = img_b.resize(img_a.size, Image.BILINEAR)
    if label.size != img_a.size:
        label = label.resize(img_a.size, Image.NEAREST)
    return img_a, img_b, label
