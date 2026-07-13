import io
import numpy as np
import cv2
import torch
import torchvision.transforms as T
from PIL import Image

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

_TRANSFORM = T.Compose([
    T.Resize((256, 256)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

_model = None


def _get_model():
    global _model
    if _model is None:
        from models.ChangeFormer import ChangeFormerV6
        from huggingface_hub import hf_hub_download
        weights_path = hf_hub_download(
            repo_id="arya2323/changeformer-levir-cd",
            filename="changeformer_levir.pt",
        )
        net = ChangeFormerV6(input_nc=3, output_nc=2, decoder_softmax=False, embed_dim=256)
        sd = torch.load(weights_path, map_location="cpu", weights_only=True)
        net.load_state_dict(sd)
        net.eval().to(DEVICE)
        _model = net
    return _model


def predict(img_a: Image.Image, img_b: Image.Image) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns:
        mask_bin : (H, W) uint8  binary change mask (255 = change)  at original img_a size
        heat     : (H, W, 3) uint8  BGR heatmap for display
    """
    orig_w, orig_h = img_a.size

    a = _TRANSFORM(img_a).unsqueeze(0).to(DEVICE)
    b = _TRANSFORM(img_b if img_b.size == img_a.size
                   else img_b.resize(img_a.size, Image.BILINEAR)).unsqueeze(0).to(DEVICE)

    model = _get_model()
    with torch.no_grad():
        out = model(a, b)

    # out[-1]: [1, 2, 256, 256] — softmax → change-class probability
    prob = torch.softmax(out[-1], dim=1)[0, 1].cpu().numpy()   # (256, 256) float32

    # Resize back to original resolution
    prob_full = np.array(Image.fromarray((prob * 255).astype(np.uint8)).resize(
        (orig_w, orig_h), Image.BILINEAR)).astype(np.float32) / 255.0

    bin_mask = (prob_full > 0.5).astype(np.uint8) * 255

    heat_norm = (prob_full * 255).astype(np.uint8)
    heat_bgr  = cv2.applyColorMap(heat_norm, cv2.COLORMAP_INFERNO)

    return bin_mask, heat_bgr


def overlay(img: Image.Image, mask: np.ndarray,
            color=(255, 60, 60), alpha=0.55) -> Image.Image:
    h, w = mask.shape[:2]
    if img.size != (w, h):
        img = img.resize((w, h), Image.BILINEAR)
    base    = np.array(img).copy()
    changed = mask > 0
    blend   = base.copy()
    blend[changed] = (
        np.array(color, dtype=np.float32) * alpha
        + base[changed] * (1 - alpha)
    ).astype(np.uint8)
    return Image.fromarray(blend)


def heatmap_rgb(heat_bgr: np.ndarray) -> Image.Image:
    return Image.fromarray(cv2.cvtColor(heat_bgr, cv2.COLOR_BGR2RGB))


def changed_area_m2(mask: np.ndarray, gsd_m: float = 0.5) -> float:
    return int((mask > 0).sum()) * (gsd_m ** 2)


def get_parquet_path() -> str:
    from huggingface_hub import hf_hub_download
    return hf_hub_download(
        repo_id="ericyu/LEVIRCD_Cropped_256",
        filename="data/test-00000-of-00001-31d7c3e3444e5b5d.parquet",
        repo_type="dataset",
    )


def load_pair_from_parquet(parquet_path: str, index: int) -> tuple:
    import pyarrow.parquet as pq
    table = pq.read_table(parquet_path)
    row   = table.slice(index, 1).to_pydict()
    img_a = Image.open(io.BytesIO(row["imageA"][0]["bytes"])).convert("RGB")
    img_b = Image.open(io.BytesIO(row["imageB"][0]["bytes"])).convert("RGB")
    label = Image.open(io.BytesIO(row["label"][0]["bytes"])).convert("L")
    if img_b.size != img_a.size:
        img_b = img_b.resize(img_a.size, Image.BILINEAR)
    if label.size != img_a.size:
        label = label.resize(img_a.size, Image.NEAREST)
    return img_a, img_b, label
