<h1 align="center">AeroMap Change Intelligence</h1>

<p align="center">
  <b>Aerial Bi-Temporal Change Detection · LEVIR-CD · ChangeFormer Transformer · IoU 0.845</b><br/>
  Detect structural change between two aerial images — season-invariant, no manual tuning.
</p>

<p align="center">
  <a href="https://aeromap-changedetector-edroxqbw6tk2w4wvtsfwce.streamlit.app/">
    <img src="https://img.shields.io/badge/🚀%20Live%20Demo-Streamlit-FF4B4B?style=for-the-badge"/>
  </a>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white"/>
  <img src="https://img.shields.io/badge/Streamlit-1.x-FF4B4B?logo=streamlit&logoColor=white"/>
  <img src="https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch&logoColor=white"/>
  <img src="https://img.shields.io/badge/Dataset-LEVIR--CD-00C853"/>
  <img src="https://img.shields.io/badge/IoU-0.845-brightgreen"/>
</p>

---

![Demo](assets/demo.png)

> *LEVIR-CD Test Pair #260 — Vacant land (2002) → Dense residential estate (2016) · 7,820 m² of detected structural change*

---

## Try it Live

**No installation needed** → [🚀 Open the live demo](https://aeromap-changedetector-edroxqbw6tk2w4wvtsfwce.streamlit.app/)

Select a test pair from the sidebar and click **▶ Detect Changes**. Pair **#260** is the most dramatic.

---

## What it does

Upload two aerial images of the same location taken at different times. The ChangeFormer model detects structural changes (new buildings, demolitions, construction) while ignoring seasonal colour shift, lighting variation, and vegetation changes.

- **Change Overlay** — red mask over changed regions on the "After" image
- **Change Heatmap** — inferno-coloured confidence map from the model
- **Metrics** — changed area in m², site coverage %, IoU vs ground truth

Season-invariant: a green field in summer vs the same field in winter triggers **0% false change**. Only structural changes register.

---

## How it works

```
Image A (Before) ──┐
                   ├──► ChangeFormer V6 (Siamese Transformer) ──► Change Probability Map
Image B (After)  ──┘         │                                            │
                    Shared MiT encoder                            Threshold @ 0.5
                    4-stage feature extraction                             │
                    Multi-scale decoder                         ┌──────────┴──────────┐
                                                                ▼                     ▼
                                                        Change Overlay         Inferno Heatmap
```

| Step | Detail |
|------|--------|
| Encoder | Mix Transformer (MiT) — 4-stage hierarchical feature extraction, shared between both images |
| Siamese diff | Feature-level absolute difference — robust to colour/lighting shift |
| Decoder | Multi-scale transformer decoder with 5 progressive predictions |
| Output | Per-pixel change probability, thresholded at 0.5 |
| Area | `changed_pixels × GSD²` — configurable ground sampling distance |

---

## Results

| Metric | Value |
|--------|-------|
| IoU (LEVIR-CD test) | **0.845** |
| F1 Score | ~0.916 |
| Seasonal false-change | **0%** |

---

## Dataset — LEVIR-CD

[LEVIR-CD](https://justchenhao.github.io/LEVIR/) is a large-scale building change detection dataset: 637 bi-temporal high-resolution Google Earth image pairs (0.5 m/pixel GSD) collected between 2002 and 2018. The HuggingFace version used here (`ericyu/LEVIRCD_Cropped_256`) contains **2048 pre-cropped 256×256 test pairs** with pixel-level ground truth masks.

---

## Run Locally

```bash
git clone https://github.com/aryabhardwaj23/AeroMap-ChangeDetector.git
cd AeroMap-ChangeDetector

python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

streamlit run app.py
```

Model weights (164 MB) and LEVIR-CD dataset (~150 MB) download automatically from HuggingFace on first run.

---

## Usage

### Mode 1 — LEVIR-CD test set
Select any of the 2048 test pairs with the slider. Pair **#260** (vacant land → full residential estate) and **#1700** are the most visually dramatic.

### Mode 2 — Upload your own pair
Upload two aerial images of the **same location** at different times (Google Earth historical imagery works well). ChangeFormer handles size normalisation and is robust to seasonal variation.

---

## Project Structure

```
AeroMap-ChangeDetector/
├── app.py              # Streamlit dashboard
├── detect.py           # Inference pipeline — ChangeFormer wrapper
├── models/             # ChangeFormer V6 architecture (wgcban/ChangeFormer)
├── assets/
│   └── demo.png        # 4-panel demo composite
└── samples/            # 20 local LEVIR-CD sample pairs (A / B / L)
```

Model weights hosted at [arya2323/changeformer-levir-cd](https://huggingface.co/arya2323/changeformer-levir-cd) on HuggingFace.

---

## Tech Stack

| Library | Role |
|---------|------|
| PyTorch + MPS | ChangeFormer inference on Apple Silicon |
| timm | Mix Transformer encoder backbone |
| einops | Tensor reshaping in transformer blocks |
| OpenCV | Colourmap, overlay compositing |
| Pillow | Image I/O |
| PyArrow | Reading LEVIR-CD parquet without pandas |
| Streamlit | Interactive dashboard |
| HuggingFace Hub | Model weights + dataset download |

---

## Related Projects

- [AeroMap-Edge](https://github.com/aryabhardwaj23/AeroMap-Edge) — Real-time drone inspection: YOLO11n detection + Depth Anything V2 metric depth + ORB visual odometry

---

<p align="center">Built for drone-company portfolios · Tested on Apple M3 · CPU/MPS inference</p>
