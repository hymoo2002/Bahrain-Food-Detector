# Bahraini Food Detector 🇧🇭🍽️

An object detector that recognizes **10 Bahraini dishes** in a photo — the model behind a
tourist "what am I eating?" app. Built with **YOLO26 + Python**, labelled in **Roboflow**,
served with **Streamlit**.

## 🔗 Try the live model → **https://bahrain-food-detector-explosive.streamlit.app/**

Upload a food photo and it draws boxes around the dishes it recognizes.

**Detects:** Balaleet · Fried Fish · Halwa · Harees · Kebab · Luqaimat · Machboos · Mattai · Nakhi · Samboosa

---

## 📊 Results (unseen test set)

| Metric | Score |
|---|---|
| **mAP@50** | **0.93** |
| mAP@50-95 | 0.69 |
| Recall | 0.91 |
| Precision | 0.74 |

Trained on **527 labelled images** (70 / 20 / 10 split), fine-tuned with Ultralytics **YOLO26**.
Averaged across all 10 classes. *(mAP = mean Average Precision — the standard detection score, 0–1.)*

---

## 🚀 Run it locally

```bash
pip install -r requirements.txt
streamlit run app/app.py
```

Or work through the notebooks in order:

| Notebook | Does |
|---|---|
| `01_dataset_analysis.ipynb` | dataset stats + charts |
| `02_training.ipynb` | fine-tune the model |
| `03_testing_and_deployment.ipynb` | evaluate + error analysis |
| `04_live_camera.ipynb` | live webcam detection |

> No GPU? Open `02_training.ipynb` in **Google Colab** (*Runtime → GPU*).

## 🔁 Retrain on your own data

Label images in Roboflow (YOLO format), then in the config cell of `02_training.ipynb` paste
your Roboflow key/version (or drop an export into `datasets/`) and **Run all**. Tune with
`EPOCHS` and `BASE_MODEL` in the same notebook.

## 📂 Repo layout

```
app/app.py     Streamlit web app
notebooks/     analysis · training · evaluation · live camera
src/utils.py   shared helpers (stats, drawing, error analysis)
datasets/      labelled images (YOLO format, from Roboflow)
models/        trained weights (.pt)
results/       metrics, charts, failure cases
```

Built end-to-end: **collect → label → fine-tune → evaluate → find failures → improve → deploy.**
