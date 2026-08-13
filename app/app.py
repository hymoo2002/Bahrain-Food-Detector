"""
Bahraini Food Detector — Streamlit app
======================================
Upload a food photo (or use your camera) and the fine-tuned YOLO26 model draws
boxes around the Bahraini dishes it recognizes.

Run from the repo root:

    streamlit run app/app.py

It loads weights from (first that exists):
    models/model.pt  ->  models/best_v2.pt  ->  models/best_v1.pt
...or pick any .pt in the sidebar. Every prediction is logged to
results/prediction_log.csv (a simple "monitoring" trail for the next data round).
"""
import os
import sys
import csv
import time
from pathlib import Path

import numpy as np
from PIL import Image
import streamlit as st

# import our src/utils.py explicitly (never clashes with another 'utils')
ROOT = Path(__file__).resolve().parent.parent
import importlib.util as _ilu
_uspec = _ilu.spec_from_file_location('food_utils', str(ROOT / 'src' / 'utils.py'))
U = _ilu.module_from_spec(_uspec); _uspec.loader.exec_module(U)  # noqa: E402

MODELS_DIR = ROOT / "models"
RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)
LOG_PATH = RESULTS_DIR / "prediction_log.csv"

st.set_page_config(page_title="Bahraini Food Detector", page_icon="🍽️", layout="wide")


# ---------------------------------------------------------------------------
# Model loading (cached so it only loads once per weights file)
# ---------------------------------------------------------------------------
def default_weights() -> str | None:
    for name in ("model.pt", "best_v2.pt", "best_v1.pt"):
        p = MODELS_DIR / name
        if p.exists():
            return str(p)
    found = sorted(MODELS_DIR.glob("*.pt"))
    return str(found[0]) if found else None


@st.cache_resource(show_spinner="Loading model…")
def load_model(weights_path: str, _mtime: float):
    from ultralytics import YOLO
    return YOLO(weights_path)


def log_predictions(filename: str, result) -> None:
    new_file = not LOG_PATH.exists()
    with open(LOG_PATH, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new_file:
            w.writerow(["timestamp", "image", "class", "confidence"])
        ts = time.strftime("%Y-%m-%d %H:%M:%S")
        if result.boxes is not None:
            for c, conf in zip(result.boxes.cls.cpu().numpy().astype(int),
                               result.boxes.conf.cpu().numpy()):
                w.writerow([ts, filename, result.names[int(c)], round(float(conf), 4)])


# ---------------------------------------------------------------------------
# Sidebar — settings
# ---------------------------------------------------------------------------
st.sidebar.title("🍽️ Settings")

weights = default_weights()
custom = st.sidebar.text_input("Weights path (.pt)", value=weights or "")
uploaded_w = st.sidebar.file_uploader("…or upload weights (.pt)", type=["pt"])
if uploaded_w is not None:
    tmp = MODELS_DIR / uploaded_w.name
    tmp.write_bytes(uploaded_w.getbuffer())
    custom = str(tmp)

conf = st.sidebar.slider("Confidence threshold", 0.05, 0.95, 0.35, 0.05)
iou = st.sidebar.slider("IoU (NMS) threshold", 0.1, 0.9, 0.5, 0.05)
do_log = st.sidebar.checkbox("Log predictions (monitoring)", value=True)

st.sidebar.markdown("---")
st.sidebar.caption("Fine-tune the model in `02_training.ipynb`. "
                   "Weights are read from `models/`.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
st.title("Bahraini Food Detector 🇧🇭")
st.write("Teach an AI to *see* Bahraini food. Upload a photo and the detector will "
         "find and label the dishes it recognizes.")

if not custom or not os.path.exists(custom):
    st.warning("No trained weights found. Train a model first with "
               "`notebooks/02_training.ipynb`, then reload this page.")
    st.stop()

try:
    model = load_model(custom, os.path.getmtime(custom))
except ModuleNotFoundError:
    st.error("`ultralytics` is not installed. Run:  `pip install -r requirements.txt`")
    st.stop()

st.caption(f"Model: `{custom}`  •  classes: {', '.join(model.names.values())}")

source = st.radio("Input", ["Upload image(s)", "Camera", "Sample images"], horizontal=True)

images = []  # list of (name, PIL.Image)
if source == "Upload image(s)":
    files = st.file_uploader("Choose one or more food photos",
                             type=["jpg", "jpeg", "png", "webp"], accept_multiple_files=True)
    for f in files or []:
        images.append((f.name, Image.open(f).convert("RGB")))
elif source == "Camera":
    shot = st.camera_input("Take a photo")
    if shot is not None:
        images.append(("camera.jpg", Image.open(shot).convert("RGB")))
else:
    sample_dir = ROOT / "images-mixed"
    samples = U.list_images(sample_dir)
    picked = st.multiselect("Pick sample image(s)", [os.path.basename(s) for s in samples],
                            default=[os.path.basename(s) for s in samples[:3]])
    for s in samples:
        if os.path.basename(s) in picked:
            images.append((os.path.basename(s), Image.open(s).convert("RGB")))

if not images:
    st.info("Add an image above to run detection.")
    st.stop()

# ---- run detection ----
for name, img in images:
    result = model.predict(np.array(img), conf=conf, iou=iou, verbose=False)[0]
    annotated = result.plot()[:, :, ::-1]  # BGR -> RGB

    c1, c2 = st.columns([3, 2])
    with c1:
        st.image(annotated, caption=name, use_container_width=True)
    with c2:
        if result.boxes is not None and len(result.boxes):
            rows = [{"dish": result.names[int(c)], "confidence": round(float(cf), 3)}
                    for c, cf in zip(result.boxes.cls.cpu().numpy().astype(int),
                                     result.boxes.conf.cpu().numpy())]
            st.subheader(f"Found {len(rows)} item(s)")
            st.dataframe(rows, use_container_width=True, hide_index=True)
            st.success("Dishes: " + ", ".join(sorted({r["dish"] for r in rows})))
        else:
            st.subheader("No dishes detected")
            st.caption("Try lowering the confidence threshold in the sidebar.")

    if do_log:
        log_predictions(name, result)

if do_log:
    st.sidebar.caption(f"Logged to `results/prediction_log.csv`")
