"""Analyze SAR visual characteristics using YOLO prediction boxes.

This script reads an existing predictions.csv file and test images. It does not
run training or prediction.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw


CLASS_NAMES = {
    0: "Fishing",
    1: "Cargo",
    2: "Passenger",
}

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

FEATURE_DESCRIPTIONS = {
    "run_name": "Prediction run folder name.",
    "image_name": "Image file name from predictions.csv.",
    "pred_class_id": "Predicted YOLO class id.",
    "pred_class_name": "Predicted YOLO class name.",
    "confidence": "YOLO prediction confidence.",
    "mean_intensity_vv": "Mean intensity of channel R, interpreted as VV, over the full patch.",
    "mean_intensity_vh": "Mean intensity of channel G, interpreted as VH, over the full patch.",
    "mean_intensity_rgb_mean": "Mean intensity of channel B, interpreted as mean(VV, VH), over the full patch.",
    "max_intensity_vv": "Maximum intensity of channel R, interpreted as VV, over the full patch.",
    "max_intensity_vh": "Maximum intensity of channel G, interpreted as VH, over the full patch.",
    "std_intensity_vv": "Standard deviation of channel R, interpreted as VV, over the full patch.",
    "std_intensity_vh": "Standard deviation of channel G, interpreted as VH, over the full patch.",
    "vv_vh_difference": "Difference between mean VV and mean VH intensity over the full patch.",
    "bright_area_ratio": "Ratio of grayscale pixels above mean plus one standard deviation.",
    "entropy": "Shannon entropy of grayscale intensity histogram.",
    "pred_x1": "Predicted bbox left coordinate in pixels.",
    "pred_y1": "Predicted bbox top coordinate in pixels.",
    "pred_x2": "Predicted bbox right coordinate in pixels.",
    "pred_y2": "Predicted bbox bottom coordinate in pixels.",
    "pred_bbox_x_center": "Predicted bbox normalized x center.",
    "pred_bbox_y_center": "Predicted bbox normalized y center.",
    "pred_bbox_width": "Predicted bbox normalized width.",
    "pred_bbox_height": "Predicted bbox normalized height.",
    "pred_bbox_area_ratio": "Predicted bbox area divided by image area.",
    "pred_bbox_aspect_ratio": "Predicted bbox width divided by height.",
    "pred_object_mean_intensity": "Mean grayscale intensity inside predicted bbox.",
    "pred_background_mean_intensity": "Mean grayscale intensity outside predicted bbox.",
    "pred_object_background_contrast": "Predicted object mean intensity minus background mean intensity.",
    "gt_class_id": "Ground-truth class id with the highest IoU for the same image, if available.",
    "gt_class_name": "Ground-truth class name with the highest IoU for the same image, if available.",
    "best_gt_iou": "Highest IoU between predicted bbox and ground-truth boxes in the same image.",
    "is_class_match": "Whether predicted class id matches the best-overlap ground-truth class id.",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze SAR characteristics from existing YOLO prediction boxes."
    )
    parser.add_argument(
        "--predictions-csv",
        default="predictions/YOLOV12N_128_E100_B16_3class_VV_VH_RGB_scene_seed42/predictions.csv",
        help="Existing YOLO predictions.csv to analyze.",
    )
    parser.add_argument(
        "--dataset-dir",
        default="yolo_new_gen/dataset_yolo_det_128_3class_vv_vh_rgb_scene",
        help="Final YOLO dataset directory.",
    )
    parser.add_argument(
        "--image-split",
        default="test",
        choices=["train", "val", "test", "auto"],
        help="Image split used by the predictions. Use auto to search all splits.",
    )
    parser.add_argument(
        "--output-dir",
        default="analysis_outputs/yolo_prediction_characteristics",
        help="Output directory for CSV, Excel, plots, and visualizations.",
    )
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.0,
        help="Optional minimum prediction confidence to include.",
    )
    parser.add_argument(
        "--samples-per-class",
        type=int,
        default=10,
        help="Maximum annotated prediction samples to save per predicted class.",
    )
    return parser.parse_args()


def load_rgb_array(image_path: Path) -> np.ndarray:
    with Image.open(image_path) as image:
        return np.asarray(image.convert("RGB"), dtype=np.float32)


def compute_entropy(grayscale: np.ndarray) -> float:
    if grayscale.size == 0:
        return math.nan
    hist, _ = np.histogram(grayscale, bins=256, range=(0, 255))
    total = hist.sum()
    if total == 0:
        return math.nan
    probabilities = hist.astype(np.float64) / float(total)
    probabilities = probabilities[probabilities > 0]
    return float(-(probabilities * np.log2(probabilities)).sum())


def image_level_features(rgb: np.ndarray) -> Dict[str, float]:
    vv = rgb[:, :, 0]
    vh = rgb[:, :, 1]
    rgb_mean_channel = rgb[:, :, 2]
    grayscale = rgb.mean(axis=2)
    bright_threshold = float(grayscale.mean() + grayscale.std())
    return {
        "mean_intensity_vv": float(vv.mean()),
        "mean_intensity_vh": float(vh.mean()),
        "mean_intensity_rgb_mean": float(rgb_mean_channel.mean()),
        "max_intensity_vv": float(vv.max()),
        "max_intensity_vh": float(vh.max()),
        "std_intensity_vv": float(vv.std()),
        "std_intensity_vh": float(vh.std()),
        "vv_vh_difference": float(vv.mean() - vh.mean()),
        "bright_area_ratio": float((grayscale > bright_threshold).mean()),
        "entropy": compute_entropy(grayscale),
    }


def clamp_bbox(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    image_width: int,
    image_height: int,
) -> Tuple[int, int, int, int]:
    left = max(0, min(image_width - 1, int(math.floor(min(x1, x2)))))
    top = max(0, min(image_height - 1, int(math.floor(min(y1, y2)))))
    right = max(0, min(image_width, int(math.ceil(max(x1, x2)))))
    bottom = max(0, min(image_height, int(math.ceil(max(y1, y2)))))
    if right <= left:
        right = min(image_width, left + 1)
    if bottom <= top:
        bottom = min(image_height, top + 1)
    return left, top, right, bottom


def prediction_bbox_features(
    grayscale: np.ndarray,
    x1: float,
    y1: float,
    x2: float,
    y2: float,
) -> Dict[str, float]:
    image_height, image_width = grayscale.shape
    left, top, right, bottom = clamp_bbox(x1, y1, x2, y2, image_width, image_height)
    bbox_width_px = right - left
    bbox_height_px = bottom - top
    object_pixels = grayscale[top:bottom, left:right]

    background_mask = np.ones(grayscale.shape, dtype=bool)
    background_mask[top:bottom, left:right] = False
    background_pixels = grayscale[background_mask]

    object_mean = float(object_pixels.mean()) if object_pixels.size else math.nan
    background_mean = float(background_pixels.mean()) if background_pixels.size else math.nan
    contrast = object_mean - background_mean if not math.isnan(background_mean) else math.nan

    return {
        "pred_x1": float(left),
        "pred_y1": float(top),
        "pred_x2": float(right),
        "pred_y2": float(bottom),
        "pred_bbox_x_center": float(((left + right) / 2.0) / image_width),
        "pred_bbox_y_center": float(((top + bottom) / 2.0) / image_height),
        "pred_bbox_width": float(bbox_width_px / image_width),
        "pred_bbox_height": float(bbox_height_px / image_height),
        "pred_bbox_area_ratio": float((bbox_width_px * bbox_height_px) / (image_width * image_height)),
        "pred_bbox_aspect_ratio": float(bbox_width_px / bbox_height_px) if bbox_height_px else math.nan,
        "pred_object_mean_intensity": object_mean,
        "pred_background_mean_intensity": background_mean,
        "pred_object_background_contrast": contrast,
    }


def find_image_path(dataset_dir: Path, image_split: str, image_name: str) -> Optional[Path]:
    splits = ["train", "val", "test"] if image_split == "auto" else [image_split]
    for split in splits:
        candidate = dataset_dir / split / "images" / image_name
        if candidate.exists():
            return candidate
    stem = Path(image_name).stem
    for split in splits:
        images_dir = dataset_dir / split / "images"
        if not images_dir.exists():
            continue
        for suffix in IMAGE_EXTENSIONS:
            candidate = images_dir / f"{stem}{suffix}"
            if candidate.exists():
                return candidate
    return None


def read_predictions(predictions_csv: Path, confidence_threshold: float) -> pd.DataFrame:
    if not predictions_csv.exists():
        raise FileNotFoundError(f"predictions.csv not found: {predictions_csv}")
    data = pd.read_csv(predictions_csv)
    required = {"image_name", "pred_class_id", "confidence", "x1", "y1", "x2", "y2"}
    missing = sorted(required - set(data.columns))
    if missing:
        raise ValueError(f"Missing required columns in {predictions_csv}: {missing}")
    data["confidence"] = pd.to_numeric(data["confidence"], errors="coerce")
    data = data[data["confidence"].fillna(-math.inf) >= confidence_threshold].copy()
    return data


def yolo_to_pixels(
    x_center: float,
    y_center: float,
    width: float,
    height: float,
    image_width: int,
    image_height: int,
) -> Tuple[float, float, float, float]:
    x1 = (x_center - width / 2.0) * image_width
    y1 = (y_center - height / 2.0) * image_height
    x2 = (x_center + width / 2.0) * image_width
    y2 = (y_center + height / 2.0) * image_height
    return x1, y1, x2, y2


def read_ground_truth_boxes(label_path: Path, image_width: int, image_height: int) -> List[Dict[str, float]]:
    boxes: List[Dict[str, float]] = []
    if not label_path.exists():
        return boxes
    for line in label_path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        parts = stripped.split()
        if len(parts) < 5:
            continue
        try:
            class_id = int(float(parts[0]))
            x_center, y_center, width, height = (float(value) for value in parts[1:5])
        except ValueError:
            continue
        x1, y1, x2, y2 = yolo_to_pixels(x_center, y_center, width, height, image_width, image_height)
        boxes.append({"class_id": class_id, "x1": x1, "y1": y1, "x2": x2, "y2": y2})
    return boxes


def bbox_iou(a: Dict[str, float], b: Dict[str, float]) -> float:
    left = max(float(a["x1"]), float(b["x1"]))
    top = max(float(a["y1"]), float(b["y1"]))
    right = min(float(a["x2"]), float(b["x2"]))
    bottom = min(float(a["y2"]), float(b["y2"]))
    inter_width = max(0.0, right - left)
    inter_height = max(0.0, bottom - top)
    intersection = inter_width * inter_height
    area_a = max(0.0, float(a["x2"]) - float(a["x1"])) * max(0.0, float(a["y2"]) - float(a["y1"]))
    area_b = max(0.0, float(b["x2"]) - float(b["x1"])) * max(0.0, float(b["y2"]) - float(b["y1"]))
    union = area_a + area_b - intersection
    return float(intersection / union) if union > 0 else 0.0


def match_ground_truth(
    dataset_dir: Path,
    image_split: str,
    image_name: str,
    image_width: int,
    image_height: int,
    pred_box: Dict[str, float],
) -> Dict[str, object]:
    splits = ["train", "val", "test"] if image_split == "auto" else [image_split]
    label_path = None
    stem = Path(image_name).stem
    for split in splits:
        candidate = dataset_dir / split / "labels" / f"{stem}.txt"
        if candidate.exists():
            label_path = candidate
            break
    if label_path is None:
        return {"gt_class_id": math.nan, "gt_class_name": None, "best_gt_iou": math.nan, "is_class_match": False}

    gt_boxes = read_ground_truth_boxes(label_path, image_width, image_height)
    if not gt_boxes:
        return {"gt_class_id": math.nan, "gt_class_name": None, "best_gt_iou": 0.0, "is_class_match": False}

    best = max(gt_boxes, key=lambda box: bbox_iou(pred_box, box))
    best_iou = bbox_iou(pred_box, best)
    gt_class_id = int(best["class_id"])
    return {
        "gt_class_id": gt_class_id,
        "gt_class_name": CLASS_NAMES.get(gt_class_id, f"Unknown_{gt_class_id}"),
        "best_gt_iou": best_iou,
        "is_class_match": int(pred_box["class_id"]) == gt_class_id,
    }


def analyze_predictions(
    predictions_csv: Path,
    dataset_dir: Path,
    image_split: str,
    confidence_threshold: float,
) -> pd.DataFrame:
    predictions = read_predictions(predictions_csv, confidence_threshold)
    run_name = predictions_csv.parent.name
    rows: List[Dict[str, object]] = []

    for _, pred in predictions.iterrows():
        image_name = str(pred["image_name"])
        image_path = find_image_path(dataset_dir, image_split, image_name)
        if image_path is None:
            print(f"Warning: image not found for prediction: {image_name}")
            continue

        rgb = load_rgb_array(image_path)
        grayscale = rgb.mean(axis=2)
        image_height, image_width = grayscale.shape

        pred_class_id = int(float(pred["pred_class_id"]))
        pred_box = {
            "class_id": pred_class_id,
            "x1": float(pred["x1"]),
            "y1": float(pred["y1"]),
            "x2": float(pred["x2"]),
            "y2": float(pred["y2"]),
        }

        row: Dict[str, object] = {
            "run_name": run_name,
            "image_name": image_name,
            "pred_class_id": pred_class_id,
            "pred_class_name": str(pred.get("pred_class_name", CLASS_NAMES.get(pred_class_id, pred_class_id))),
            "confidence": float(pred["confidence"]),
            "image_width": image_width,
            "image_height": image_height,
            "image_path": str(image_path),
        }
        row.update(image_level_features(rgb))
        row.update(prediction_bbox_features(grayscale, pred_box["x1"], pred_box["y1"], pred_box["x2"], pred_box["y2"]))
        row.update(match_ground_truth(dataset_dir, image_split, image_name, image_width, image_height, pred_box))
        rows.append(row)

    return pd.DataFrame(rows)


def summary_by_pred_class(features: pd.DataFrame) -> pd.DataFrame:
    if features.empty:
        return pd.DataFrame()
    numeric_cols = [
        "confidence",
        "pred_bbox_area_ratio",
        "pred_bbox_aspect_ratio",
        "pred_object_mean_intensity",
        "pred_background_mean_intensity",
        "pred_object_background_contrast",
        "best_gt_iou",
    ]
    available = [col for col in numeric_cols if col in features.columns]
    return features.groupby("pred_class_name")[available].agg(["count", "mean", "median", "std", "min", "max"])


def feature_description_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [{"feature": feature, "description": description} for feature, description in FEATURE_DESCRIPTIONS.items()]
    )


def write_outputs(features: pd.DataFrame, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_csv = output_dir / "yolo_prediction_characteristics_raw.csv"
    summary_excel = output_dir / "yolo_prediction_characteristics_summary.xlsx"
    features.to_csv(raw_csv, index=False)
    with pd.ExcelWriter(summary_excel, engine="openpyxl") as writer:
        features.to_excel(writer, index=False, sheet_name="Raw_Prediction_Features")
        summary_by_pred_class(features).to_excel(writer, sheet_name="Summary_By_Pred_Class")
        feature_description_frame().to_excel(writer, index=False, sheet_name="Feature_Description")
    print(f"Saved raw prediction features to {raw_csv}")
    print(f"Saved prediction feature workbook to {summary_excel}")


def make_boxplots(features: pd.DataFrame, output_dir: Path) -> None:
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    plot_features = [
        "confidence",
        "pred_bbox_area_ratio",
        "pred_object_mean_intensity",
        "pred_background_mean_intensity",
        "pred_object_background_contrast",
        "best_gt_iou",
    ]
    for feature_name in plot_features:
        if feature_name not in features.columns:
            continue
        fig, ax = plt.subplots(figsize=(8, 5))
        features.boxplot(column=feature_name, by="pred_class_name", grid=False, ax=ax)
        ax.set_title(f"{feature_name} by predicted class")
        ax.set_xlabel("Predicted class")
        ax.set_ylabel(feature_name)
        fig.suptitle("")
        fig.tight_layout()
        fig.savefig(plots_dir / f"boxplot_{feature_name}.png", dpi=150, bbox_inches="tight")
        plt.close(fig)


def draw_prediction_sample(row: pd.Series, output_path: Path) -> None:
    image_path = Path(str(row["image_path"]))
    with Image.open(image_path) as image:
        annotated = image.convert("RGB")
    draw = ImageDraw.Draw(annotated)
    x1 = float(row["pred_x1"])
    y1 = float(row["pred_y1"])
    x2 = float(row["pred_x2"])
    y2 = float(row["pred_y2"])
    draw.rectangle([x1, y1, x2, y2], outline=(255, 255, 0), width=2)
    label = f"{row['pred_class_name']} conf={float(row['confidence']):.2f}"
    draw.text((max(0, int(x1)), max(0, int(y1) - 12)), label, fill=(255, 255, 0))
    annotated.save(output_path)


def annotate_prediction_patch(row: pd.Series, size: Tuple[int, int] = (360, 360)) -> Image.Image:
    image_path = Path(str(row["image_path"]))
    with Image.open(image_path) as image:
        original = image.convert("RGB")

    original_width, original_height = original.size
    annotated = original.resize(size, Image.Resampling.NEAREST)
    scale_x = size[0] / original_width
    scale_y = size[1] / original_height

    x1 = float(row["pred_x1"]) * scale_x
    y1 = float(row["pred_y1"]) * scale_y
    x2 = float(row["pred_x2"]) * scale_x
    y2 = float(row["pred_y2"]) * scale_y

    draw = ImageDraw.Draw(annotated)
    draw.rectangle([x1, y1, x2, y2], outline=(255, 255, 0), width=3)

    gt_text = "GT=?"
    if "gt_class_name" in row and pd.notna(row["gt_class_name"]):
        gt_text = f"GT={row['gt_class_name']}"
    text_lines = [
        f"Pred={row['pred_class_name']} conf={float(row['confidence']):.2f}",
        f"{gt_text} IoU={float(row['best_gt_iou']):.2f}" if pd.notna(row.get("best_gt_iou", np.nan)) else gt_text,
        f"Obj={float(row['pred_object_mean_intensity']):.1f}",
        f"Bg={float(row['pred_background_mean_intensity']):.1f}",
        f"Contrast={float(row['pred_object_background_contrast']):.1f}",
    ]

    text_x = 6
    text_y = 6
    line_height = 15
    box_width = max(230, int(size[0] * 0.68))
    box_height = line_height * len(text_lines) + 8
    draw.rectangle([0, 0, box_width, box_height], fill=(0, 0, 0))
    for line in text_lines:
        draw.text((text_x, text_y), line, fill=(255, 255, 255))
        text_y += line_height
    return annotated


def select_representative_predictions(group: pd.DataFrame, count: int) -> pd.DataFrame:
    if len(group) <= count:
        return group.sort_values("confidence", ascending=False)

    sorted_group = group.sort_values("confidence", ascending=True).reset_index(drop=True)
    positions = np.linspace(0.1, 0.9, count)
    selected_indices = sorted({int(round(position * (len(sorted_group) - 1))) for position in positions})
    selected = sorted_group.iloc[selected_indices].copy()

    while len(selected) < count:
        remaining = sorted_group.drop(selected.index, errors="ignore")
        if remaining.empty:
            break
        selected = pd.concat([selected, remaining.tail(1)], ignore_index=True)
    return selected.sort_values("confidence", ascending=False).head(count)


def make_representative_montage(features: pd.DataFrame, output_dir: Path, samples_per_class: int) -> None:
    montage_dir = output_dir / "representative_visualizations"
    montage_dir.mkdir(parents=True, exist_ok=True)
    if features.empty or samples_per_class <= 0:
        return

    class_names = [name for name in ["Fishing", "Cargo", "Passenger"] if name in set(features["pred_class_name"])]
    if not class_names:
        class_names = sorted(features["pred_class_name"].dropna().unique())

    cols = min(samples_per_class, 5)
    rows = len(class_names)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 4.4), squeeze=False)

    for row_idx, class_name in enumerate(class_names):
        group = features[features["pred_class_name"] == class_name]
        selected = select_representative_predictions(group, cols).reset_index(drop=True)
        for col_idx in range(cols):
            ax = axes[row_idx, col_idx]
            ax.axis("off")
            if col_idx >= len(selected):
                continue
            patch = annotate_prediction_patch(selected.loc[col_idx], size=(360, 360))
            ax.imshow(patch)
            if col_idx == 0:
                ax.set_ylabel(f"Pred {class_name}", fontsize=12, weight="bold")

    fig.suptitle(
        "Representative SAR patches using YOLO prediction boxes "
        "(bbox prediksi, confidence, IoU, dan fitur intensitas)",
        fontsize=15,
    )
    fig.tight_layout()
    output_path = montage_dir / "prediction_representative_montage_by_pred_class.png"
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def save_samples(features: pd.DataFrame, output_dir: Path, samples_per_class: int) -> None:
    samples_dir = output_dir / "prediction_samples"
    samples_dir.mkdir(parents=True, exist_ok=True)
    if features.empty or samples_per_class <= 0:
        return
    sorted_features = features.sort_values("confidence", ascending=False)
    for class_name, group in sorted_features.groupby("pred_class_name"):
        class_dir = samples_dir / str(class_name)
        class_dir.mkdir(parents=True, exist_ok=True)
        for idx, (_, row) in enumerate(group.head(samples_per_class).iterrows(), start=1):
            safe_name = Path(str(row["image_name"])).stem
            output_path = class_dir / f"{idx:02d}_{safe_name}_conf_{float(row['confidence']):.2f}.png"
            draw_prediction_sample(row, output_path)


def main() -> None:
    args = parse_args()
    predictions_csv = Path(args.predictions_csv)
    dataset_dir = Path(args.dataset_dir)
    output_dir = Path(args.output_dir)

    features = analyze_predictions(
        predictions_csv=predictions_csv,
        dataset_dir=dataset_dir,
        image_split=args.image_split,
        confidence_threshold=args.confidence_threshold,
    )
    write_outputs(features, output_dir)
    make_boxplots(features, output_dir)
    make_representative_montage(features, output_dir, args.samples_per_class)
    save_samples(features, output_dir, args.samples_per_class)
    print(f"Saved plots and annotated samples to {output_dir}")


if __name__ == "__main__":
    main()
