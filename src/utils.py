"""
utils.py — shared helpers for the Bahraini Food Detector lab.

Kept deliberately dependency-light (numpy / pandas / matplotlib / PIL / pyyaml)
so the dataset-analysis notebook runs even before you install PyTorch/YOLO.

Used by:
  - notebooks/01_dataset_analysis.ipynb   (dataset stats + charts)
  - notebooks/03_testing_and_deployment.ipynb (error analysis)
  - app/app.py                            (drawing boxes, class colors)
"""
from __future__ import annotations

import os
import glob
import colorsys
from pathlib import Path

import yaml
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff")


# ---------------------------------------------------------------------------
# Dataset config (data.yaml) loading
# ---------------------------------------------------------------------------
def find_data_yaml(search_dir: str | os.PathLike) -> str | None:
    """Return the path to the first `data.yaml` found under `search_dir`."""
    search_dir = str(search_dir)
    hits = sorted(glob.glob(os.path.join(search_dir, "**", "data.yaml"), recursive=True))
    return hits[0] if hits else None


def _resolve_split(yaml_dir: str, value, split: str) -> str:
    """Resolve a split entry from data.yaml to an absolute image directory."""
    candidates = []
    if value:
        value = str(value)
        candidates.append(os.path.normpath(os.path.join(yaml_dir, value)))
        candidates.append(os.path.normpath(os.path.join(yaml_dir, value.lstrip("./"))))
    candidates.append(os.path.join(yaml_dir, split, "images"))
    candidates.append(os.path.join(yaml_dir, split))
    for c in candidates:
        if os.path.isdir(c):
            return c
    return candidates[0]  # best guess (may not exist yet)


def load_data_yaml(data_yaml: str | os.PathLike) -> dict:
    """
    Load a YOLO data.yaml and return a normalized dict:
        {names: [...], nc: int, splits: {train: <img_dir>, val: ..., test: ...},
         yaml_path, root}
    Handles Roboflow's relative paths and an optional top-level `path:` root.
    """
    data_yaml = str(data_yaml)
    with open(data_yaml, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    yaml_dir = os.path.dirname(os.path.abspath(data_yaml))
    if cfg.get("path"):
        root = os.path.normpath(os.path.join(yaml_dir, str(cfg["path"])))
        if os.path.isdir(root):
            yaml_dir = root

    names = cfg.get("names")
    if isinstance(names, dict):  # {0: 'a', 1: 'b'} -> ['a', 'b']
        names = [names[k] for k in sorted(names, key=int)]

    splits = {s: _resolve_split(yaml_dir, cfg.get(s), s if s != "val" else "valid")
              for s in ("train", "val", "test")}

    return {
        "names": names or [],
        "nc": int(cfg.get("nc", len(names or []))),
        "splits": splits,
        "yaml_path": os.path.abspath(data_yaml),
        "root": yaml_dir,
    }


# ---------------------------------------------------------------------------
# Image / label file helpers
# ---------------------------------------------------------------------------
def list_images(images_dir: str | os.PathLike) -> list[str]:
    if not images_dir or not os.path.isdir(images_dir):
        return []
    files = [p for p in glob.glob(os.path.join(images_dir, "*"))
             if p.lower().endswith(IMAGE_EXTS)]
    return sorted(files)


def label_path_for_image(image_path: str | os.PathLike) -> str:
    """Map .../images/foo.jpg -> .../labels/foo.txt (YOLO convention)."""
    p = Path(image_path)
    label_dir = Path(str(p.parent).replace(os.sep + "images", os.sep + "labels"))
    if label_dir == p.parent:  # no 'images' segment; sit labels next to image
        label_dir = p.parent
    return str(label_dir / (p.stem + ".txt"))


def read_yolo_label(label_path: str | os.PathLike) -> np.ndarray:
    """Read a YOLO label file -> array of shape (N, 5): [class, xc, yc, w, h] (normalized)."""
    if not os.path.isfile(label_path):
        return np.zeros((0, 5), dtype=float)
    rows = []
    with open(label_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.split()
            if len(parts) >= 5:
                rows.append([float(x) for x in parts[:5]])
    return np.array(rows, dtype=float) if rows else np.zeros((0, 5), dtype=float)


# ---------------------------------------------------------------------------
# Dataset statistics  (Lab Section 6 — Dataset Quality Check)
# ---------------------------------------------------------------------------
def dataset_stats(data_cfg: dict, splits=("train", "val", "test")) -> dict:
    """
    Compute the numbers the lab asks you to report:
      - images per split
      - objects per class (+ per split)
      - images containing each class
      - average objects per image
    Returns a dict of pandas DataFrames + a few scalars.
    """
    names = data_cfg["names"]
    n_classes = len(names)

    per_class_objs = {s: np.zeros(n_classes, dtype=int) for s in splits}
    per_class_imgs = {s: np.zeros(n_classes, dtype=int) for s in splits}
    images_per_split = {}
    objs_per_image = {s: [] for s in splits}

    for s in splits:
        imgs = list_images(data_cfg["splits"].get(s))
        images_per_split[s] = len(imgs)
        for img in imgs:
            lab = read_yolo_label(label_path_for_image(img))
            objs_per_image[s].append(len(lab))
            present = set()
            for cls_id in lab[:, 0].astype(int) if len(lab) else []:
                if 0 <= cls_id < n_classes:
                    per_class_objs[s][cls_id] += 1
                    present.add(cls_id)
            for cls_id in present:
                per_class_imgs[s][cls_id] += 1

    # ---- assemble tidy DataFrames ----
    obj_df = pd.DataFrame({s: per_class_objs[s] for s in splits}, index=names)
    obj_df["total"] = obj_df.sum(axis=1)
    img_df = pd.DataFrame({s: per_class_imgs[s] for s in splits}, index=names)
    img_df["total"] = img_df.sum(axis=1)

    split_df = pd.DataFrame({
        "images": [images_per_split[s] for s in splits],
        "objects": [int(np.sum(objs_per_image[s])) for s in splits],
        "avg_objects_per_image": [
            round(float(np.mean(objs_per_image[s])), 2) if objs_per_image[s] else 0.0
            for s in splits
        ],
    }, index=list(splits))

    total_imgs = int(sum(images_per_split.values()))
    total_objs = int(obj_df["total"].sum())

    # class imbalance: ratio of most-common to least-common class (by total objects)
    totals = obj_df["total"].values
    nonzero = totals[totals > 0]
    imbalance_ratio = float(nonzero.max() / nonzero.min()) if len(nonzero) else float("nan")

    return {
        "objects_per_class": obj_df,
        "images_per_class": img_df,
        "per_split": split_df,
        "total_images": total_imgs,
        "total_objects": total_objs,
        "avg_objects_per_image": round(total_objs / total_imgs, 2) if total_imgs else 0.0,
        "imbalance_ratio": imbalance_ratio,
    }


# ---------------------------------------------------------------------------
# Geometry + drawing
# ---------------------------------------------------------------------------
def xywhn_to_xyxy(box_n, w: int, h: int):
    """Normalized [xc, yc, bw, bh] -> pixel [x1, y1, x2, y2]."""
    xc, yc, bw, bh = box_n
    x1 = (xc - bw / 2) * w
    y1 = (yc - bh / 2) * h
    x2 = (xc + bw / 2) * w
    y2 = (yc + bh / 2) * h
    return [x1, y1, x2, y2]


def iou_xyxy(a, b) -> float:
    """IoU of two [x1, y1, x2, y2] boxes."""
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def class_color(idx: int) -> tuple[int, int, int]:
    """Deterministic distinct RGB color per class index."""
    hue = (idx * 0.61803398875) % 1.0  # golden-ratio spacing
    r, g, b = colorsys.hsv_to_rgb(hue, 0.65, 1.0)
    return int(r * 255), int(g * 255), int(b * 255)


def draw_boxes(img, boxes_xyxy, labels, scores=None, class_ids=None, width=3):
    """
    Draw boxes on a PIL image (or path / numpy array). Returns a new PIL.Image.
      boxes_xyxy : list of [x1,y1,x2,y2] in pixels
      labels     : list of text labels (strings)
      scores     : optional list of confidences (floats)
      class_ids  : optional list of ints (for consistent colors); else uses order
    """
    if isinstance(img, (str, os.PathLike)):
        img = Image.open(img)
    elif isinstance(img, np.ndarray):
        img = Image.fromarray(img)
    img = img.convert("RGB").copy()
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("arial.ttf", max(14, img.width // 60))
    except Exception:
        font = ImageFont.load_default()

    for i, (box, label) in enumerate(zip(boxes_xyxy, labels)):
        cid = class_ids[i] if class_ids is not None else i
        color = class_color(int(cid))
        x1, y1, x2, y2 = [float(v) for v in box]
        draw.rectangle([x1, y1, x2, y2], outline=color, width=width)
        text = f"{label} {scores[i]:.2f}" if scores is not None else str(label)
        tb = draw.textbbox((0, 0), text, font=font)
        tw, th = tb[2] - tb[0], tb[3] - tb[1]
        draw.rectangle([x1, max(0, y1 - th - 4), x1 + tw + 6, y1], fill=color)
        draw.text((x1 + 3, max(0, y1 - th - 3)), text, fill=(0, 0, 0), font=font)
    return img


# ---------------------------------------------------------------------------
# Error analysis  (Lab Phase 6)
# ---------------------------------------------------------------------------
def match_detections(pred_boxes, pred_cls, pred_conf,
                     gt_boxes, gt_cls, iou_thr=0.5, loc_thr=0.1) -> dict:
    """
    Classify predictions vs ground truth for ONE image into the failure
    categories the lab lists (Phase 6). Greedy match, highest-confidence first.

    Returns dict with index records:
      tp                -> [(pred_i, gt_j), ...]   correct (right class, IoU>=thr)
      wrong_class       -> [(pred_i, gt_j), ...]   overlaps a GT but wrong label
      poor_localization -> [(pred_i, gt_j), ...]   right class, loc_thr<=IoU<thr
      false_positive    -> [pred_i, ...]           predicted where nothing matches
      false_negative    -> [gt_j, ...]             real object the model missed
    """
    P, G = len(pred_cls), len(gt_cls)
    rec = {"tp": [], "wrong_class": [], "poor_localization": [],
           "false_positive": [], "false_negative": []}

    iou = np.zeros((P, G))
    for i in range(P):
        for j in range(G):
            iou[i, j] = iou_xyxy(pred_boxes[i], gt_boxes[j])

    gt_matched = [False] * G
    order = np.argsort(-np.asarray(pred_conf)) if P else []
    for i in order:
        if G == 0:
            rec["false_positive"].append(int(i))
            continue
        j = int(np.argmax(iou[i]))
        best = iou[i, j]
        same = int(pred_cls[i]) == int(gt_cls[j])
        if best >= iou_thr and same and not gt_matched[j]:
            gt_matched[j] = True
            rec["tp"].append((int(i), j))
        elif best >= iou_thr and not same:
            rec["wrong_class"].append((int(i), j))
        elif loc_thr <= best < iou_thr and same:
            rec["poor_localization"].append((int(i), j))
        else:
            rec["false_positive"].append(int(i))

    for j in range(G):
        if not gt_matched[j]:
            rec["false_negative"].append(j)
    return rec


# ---------------------------------------------------------------------------
# YOLO (ultralytics) metrics -> tidy tables
# ---------------------------------------------------------------------------
def summarize_metrics(metrics) -> tuple[dict, "pd.DataFrame"]:
    """
    Turn an ultralytics DetMetrics object (returned by `model.val(...)`) into
    (overall_dict, per_class_dataframe) with precision/recall/mAP50/mAP50-95.
    """
    box = metrics.box
    overall = {
        "precision": float(box.mp),
        "recall": float(box.mr),
        "mAP50": float(box.map50),
        "mAP50-95": float(box.map),
    }
    names = getattr(metrics, "names", {}) or {}
    rows = []
    for k, c in enumerate(list(box.ap_class_index)):
        rows.append({
            "class": names.get(int(c), str(c)),
            "precision": float(box.p[k]),
            "recall": float(box.r[k]),
            "mAP50": float(box.ap50[k]),
            "mAP50-95": float(box.ap[k]),
        })
    per_class = pd.DataFrame(rows).sort_values("mAP50-95").reset_index(drop=True)
    return overall, per_class
