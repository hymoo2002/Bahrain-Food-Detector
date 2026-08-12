# Bahraini Food Detector 🇧🇭🍽️

Teach an AI to **see** Bahraini food. This is an object-detection project: give it a
photo of a meal and it draws boxes around the local dishes it recognizes (samboosa,
machboos, luqaimat, dates…). Built with **YOLO11 + Python**, dataset labeled in
**Roboflow**, demo served with **Streamlit**.

> Workflow: **Collect → Annotate → Fine-tune → Evaluate → Break → Improve → Deploy → Monitor**

---

## 📂 What's in this repo

```
Bahrain Food Detector/
├── README.md
├── requirements.txt
├── data.template.yaml            # reference for the YOLO data.yaml + the 10 classes
├── notebooks/
│   ├── 01_dataset_analysis.ipynb        # dataset quality check + charts
│   ├── 02_training.ipynb                # fine-tune v1 & v2, evaluate, compare
│   └── 03_testing_and_deployment.ipynb  # test-set eval, error analysis, demo, export
├── app/
│   └── app.py                    # Streamlit app (upload photo → detections)
├── src/
│   └── utils.py                  # shared helpers (stats, drawing, error analysis)
├── datasets/                     # <- your Roboflow export lands here (data.yaml + images)
├── models/                       # <- trained weights: best_v1.pt, best_v2.pt, model.pt
├── results/                      # <- charts, metrics, failure cases, prediction log
└── images-mixed/                 # a few sample photos for quick demos
```

---

## ⚡ Quickstart

```bash
# 1) (recommended) create a virtual environment
python -m venv .venv
.venv\Scripts\activate            # Windows PowerShell:  .venv\Scripts\Activate.ps1

# 2) install dependencies
pip install -r requirements.txt

# 3) launch Jupyter and open the notebooks/ in order
jupyter lab
```

> **GPU strongly recommended for training.** No GPU? Open `02_training.ipynb` in
> **Google Colab** (free GPU: *Runtime → Change runtime type → GPU*). Everything else
> (analysis, the app) runs fine on CPU.

---

## 🧭 The full workflow

### Step 0 — Collect images *(you)*
- Pick **10 Bahraini food classes** (e.g. samboosa, kubbah, mutabbaq, balaleet,
  machboos, harees, luqaimat, dates, khubz, gahwa).
- Collect **≥ 300 original images**, aiming for **30–40 per class**.
- **Variety is the whole game:** different plates, distances (close/medium/far),
  angles (top-down/front/angled), lighting, backgrounds, single-dish **and**
  multi-dish photos, some partial occlusion. Avoid near-duplicates.

### Step 1 — Annotate in Roboflow *(you)*
1. Create a project in [Roboflow](https://roboflow.com) → type **Object Detection**.
2. Upload your images and draw a **tight bounding box** around every visible instance
   of your 10 classes. Use the **exact same class names** everywhere (see
   `data.template.yaml`). Don't box things outside your classes.
3. **Generate a version.** Set the split to **70% train / 20% valid / 10% test**.
   - Preprocessing: *Auto-Orient* + *Resize 640×640* is plenty.
   - Augmentations: you can **leave these off** — YOLO augments the training set itself.
     If you do add Roboflow augmentations, they apply to **train only** (correct).
4. **Export** → format **YOLOv11** (YOLOv8 works too — identical labels) → either:
   - **Download zip** and unzip into `datasets/`, **or**
   - copy the **`pip`/Python snippet** (workspace, project, version, API key) — paste
     those into the config cell of `02_training.ipynb` to download automatically.

### Step 2 — Analyze the dataset → `notebooks/01_dataset_analysis.ipynb`
Reports images per split, objects per class, images per class, avg objects/image,
class imbalance, plus the **required objects-per-class chart** and good/questionable
annotation examples. Saves charts to `results/`.

### Step 3 — Fine-tune v1 + evaluate → `notebooks/02_training.ipynb`
Shows the **baseline** (pretrained COCO model can't name Bahraini food), fine-tunes
**YOLO11**, then evaluates on the **test** set (precision, recall, mAP50, mAP50-95,
overall + per class). Saves `models/best_v1.pt`.

### Step 4 — Error analysis → `notebooks/03_testing_and_deployment.ipynb`
Final test-set evaluation + confusion matrix, then automatically finds and renders
**≥ 10 failure cases** (false positive / false negative / wrong class / poor
localization) into `results/failures/` and `results/failure_cases.csv`.

### Step 5 — Targeted collection + retrain v2 *(you + `02_training.ipynb`)*
Based on the failures, collect **30–50 targeted images** (e.g. more low-light shots,
more crowded plates), label them in Roboflow as a **new version (v2)**, then run the
**v2** section of `02_training.ipynb`. It trains with the **same settings** and prints
the **v1 vs v2 comparison** (`results/comparison_v1_v2.csv` + chart).

### Step 6 — Deploy → `app/app.py`
```bash
streamlit run app/app.py
```
Upload a photo → boxes + labels + confidence. Every prediction is appended to
`results/prediction_log.csv` (your **monitoring** trail).

---

## 🔁 How to retrain / fine-tune (what to change)

Everything lives in the **config cells** — no need to touch the logic.

**Point at a different / newer dataset** (`02_training.ipynb`, cell 1):
```python
ROBOFLOW_API_KEY = "xxxx"     # your key
WORKSPACE        = "my-workspace"
PROJECT          = "bahraini-food"
VERSION          = 3          # <- bump to your newest Roboflow version and re-run
```
…or set `LOCAL_DATASET_DIR` to a folder you unzipped into `datasets/` and leave the
API key empty.

**Change the training recipe** (`02_training.ipynb`, cell 3):
```python
BASE_MODEL = "yolo11s.pt"   # yolo11n.pt = faster/smaller, yolo11m.pt = more accurate
EPOCHS     = 80             # raise if loss/mAP still improving; lower to iterate faster
IMGSZ      = 640            # 640 is standard; 960 can help tiny items (needs more VRAM)
BATCH      = 16             # lower to 8/4 if you get a GPU out-of-memory error; -1 = auto
PATIENCE   = 20            # early-stop patience on val mAP
```
Then just re-run the training + evaluation cells. Each run is saved separately under
`runs/finetune_*/`. To **resume** an interrupted run:
```python
from ultralytics import YOLO
YOLO("runs/finetune_v1/weights/last.pt").train(resume=True)
```

**Rules of thumb**
- Low **recall** (missing objects) → collect more data / more variety of the missed class.
- Low **precision** (false alarms) → add "hard negative" backgrounds; raise `conf`.
- Two classes confused → collect clearer, balanced examples of both.
- Tiny objects missed → shoot closer, or train at `IMGSZ = 960`.
- More data usually helps, but **diverse** data helps most — that's the lab's core lesson.

---

## ✅ Deliverables checklist (from the lab)

| # | Deliverable | Where |
|---|---|---|
| 1 | README | this file |
| 2 | ≥ 300 collected images | `datasets/` (from Roboflow) |
| 3 | Annotated dataset (YOLO format) | `datasets/.../{train,valid,test}` |
| 4 | Dataset analysis notebook | `notebooks/01_dataset_analysis.ipynb` |
| 5 | Training notebook + Python app | `notebooks/02_training.ipynb`, `app/app.py` |
| 6 | Trained model weights | `models/best_v1.pt`, `models/best_v2.pt` |
| 7 | Evaluation results | `results/` (metrics CSVs, curves, confusion matrix) |
| 8 | ≥ 10 documented failure cases | `results/failures/`, `results/failure_cases.csv` |
| 9 | Second-round targeted dataset | Roboflow v2 → `datasets/` |
| 10 | Baseline vs v1 vs v2 comparison | `results/comparison_v1_v2.csv` + notebook §2/§7 |
| 11 | Short video demo | record the Streamlit app on unseen photos |
| 12 | 5–8 min presentation | follow the lab's presentation structure |

---

## 🛠️ Troubleshooting

- **`ultralytics`/torch install is slow or fails** — that's PyTorch. For an NVIDIA GPU,
  install the matching torch build from <https://pytorch.org/get-started/locally/>
  first, then `pip install -r requirements.txt`.
- **`YOLO can't find images` / `data.yaml` path errors** — the training notebook rewrites
  the split paths to absolute automatically (`absolutize()`); just re-run that cell.
- **CUDA out of memory** — lower `BATCH` (8 → 4) or `IMGSZ` (640 → 512).
- **`model.val(split="test")` errors** — your Roboflow version must include a **test**
  split. Regenerate the version with the 70/20/10 split.
- **App says "no trained weights"** — run `02_training.ipynb` first, then reload; or paste
  a `.pt` path / upload weights in the sidebar.

---

## 📝 Notes
- Uses **YOLO11** (Ultralytics). Swap `BASE_MODEL` for any `yolo11{n,s,m,l,x}.pt`.
- Roboflow can also train a model for you ("Roboflow Train"). It's a fine *extra*
  baseline, but this lab asks you to run the fine-tuning yourself so you get the
  baseline → v1 → v2 comparison, error analysis, and experiment tracking.
- Class names must match **exactly** across Roboflow, the notebooks, and the app.
