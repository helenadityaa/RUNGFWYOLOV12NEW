"""Collect summaries from existing YOLO predict outputs without running prediction."""

from __future__ import annotations

import argparse
import math
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd


CLASS_NAMES = {
    0: "Fishing",
    1: "Cargo",
    2: "Passenger",
}

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

SUMMARY_COLUMNS = [
    "run_name",
    "total_images",
    "total_predictions",
    "total_no_detection",
    "avg_confidence",
    "predict_dir",
]

BY_CLASS_COLUMNS = [
    "run_name",
    "class_id",
    "class_name",
    "prediction_count",
    "avg_confidence",
]

NO_DETECTION_COLUMNS = ["run_name", "image_name"]

NOTES = [
    "Rekap ini membaca hasil predict yang sudah ada.",
    "Script tidak menjalankan predict ulang.",
    "Class mapping: 0 = Fishing, 1 = Cargo, 2 = Passenger.",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize existing YOLO predict outputs from runs/predict."
    )
    parser.add_argument("--predict-dir", default="runs/predict", help="Directory containing predict runs.")
    parser.add_argument(
        "--dataset-dir",
        default="yolo_new_gen/dataset_yolo_det_128_3class_vv_vh_rgb_scene",
        help="Final YOLO dataset directory, used to compare against test images.",
    )
    parser.add_argument(
        "--output-csv",
        default="summary_existing_predictions.csv",
        help="Output CSV summary path.",
    )
    parser.add_argument(
        "--output-excel",
        default="summary_existing_predictions.xlsx",
        help="Output Excel summary path.",
    )
    return parser.parse_args()


def normalize_column_name(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", name.strip().lower())


def find_column(columns: Iterable[str], candidates: Iterable[str]) -> Optional[str]:
    normalized_columns = {normalize_column_name(column): column for column in columns}
    for candidate in candidates:
        normalized = normalize_column_name(candidate)
        if normalized in normalized_columns:
            return normalized_columns[normalized]
    return None


def dataset_test_images(dataset_dir: Path) -> Dict[str, str]:
    images_dir = dataset_dir / "test" / "images"
    if not images_dir.exists():
        return {}
    return {
        path.stem: path.name
        for path in sorted(images_dir.iterdir())
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    }


def infer_run_images(run_dir: Path) -> Dict[str, str]:
    images: Dict[str, str] = {}
    for path in run_dir.rglob("*"):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            images.setdefault(path.stem, path.name)
    return images


def class_value_to_id(value: object) -> Optional[int]:
    if pd.isna(value):
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError):
        value_text = str(value).strip().lower()
        for class_id, class_name in CLASS_NAMES.items():
            if value_text == class_name.lower():
                return class_id
    return None


def image_value_to_name(value: object) -> Optional[str]:
    if pd.isna(value):
        return None
    value_text = str(value).strip()
    if not value_text:
        return None
    return Path(value_text).name


def read_predictions_csv(predictions_csv: Path) -> pd.DataFrame:
    data = pd.read_csv(predictions_csv)
    data.columns = [column.strip() for column in data.columns]
    image_col = find_column(data.columns, ["image_name", "image", "filename", "file", "path", "source"])
    class_col = find_column(
        data.columns,
        ["class_id", "pred_class_id", "class", "cls", "category_id", "name", "pred_class_name"],
    )
    confidence_col = find_column(
        data.columns,
        ["confidence", "conf", "score", "probability", "pred_confidence"],
    )

    rows: List[Dict[str, object]] = []
    for _, row in data.iterrows():
        class_id = class_value_to_id(row.get(class_col)) if class_col else None
        image_name = image_value_to_name(row.get(image_col)) if image_col else None
        confidence = pd.to_numeric(row.get(confidence_col), errors="coerce") if confidence_col else math.nan
        if class_id is None:
            continue
        rows.append(
            {
                "image_name": image_name,
                "image_stem": Path(image_name).stem if image_name else None,
                "class_id": class_id,
                "confidence": float(confidence) if not pd.isna(confidence) else math.nan,
            }
        )
    return pd.DataFrame(rows, columns=["image_name", "image_stem", "class_id", "confidence"])


def read_prediction_labels(labels_dir: Path, test_images: Dict[str, str]) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    if not labels_dir.exists():
        return pd.DataFrame(columns=["image_name", "image_stem", "class_id", "confidence"])
    for label_path in sorted(labels_dir.glob("*.txt")):
        image_stem = label_path.stem
        image_name = test_images.get(image_stem, label_path.name.replace(".txt", ""))
        for line_number, line in enumerate(label_path.read_text(encoding="utf-8").splitlines(), start=1):
            stripped = line.strip()
            if not stripped:
                continue
            parts = stripped.split()
            if len(parts) < 5:
                print(f"Warning: skipping malformed prediction line {label_path}:{line_number}")
                continue
            try:
                class_id = int(float(parts[0]))
                confidence = float(parts[5]) if len(parts) >= 6 else math.nan
            except ValueError:
                print(f"Warning: skipping non-numeric prediction line {label_path}:{line_number}")
                continue
            rows.append(
                {
                    "image_name": image_name,
                    "image_stem": image_stem,
                    "class_id": class_id,
                    "confidence": confidence,
                }
            )
    return pd.DataFrame(rows, columns=["image_name", "image_stem", "class_id", "confidence"])


def read_run_predictions(run_dir: Path, test_images: Dict[str, str]) -> pd.DataFrame:
    predictions_csv = run_dir / "predictions.csv"
    if predictions_csv.exists():
        return read_predictions_csv(predictions_csv)
    return read_prediction_labels(run_dir / "labels", test_images)


def prediction_class_name(class_id: int) -> str:
    return CLASS_NAMES.get(class_id, f"Unknown_{class_id}")


def summarize_by_class(run_name: str, predictions: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    known_class_ids = set(CLASS_NAMES)
    if not predictions.empty:
        known_class_ids.update(int(value) for value in predictions["class_id"].dropna().unique())

    for class_id in sorted(known_class_ids):
        class_predictions = predictions[predictions["class_id"] == class_id] if not predictions.empty else predictions
        confidences = pd.to_numeric(class_predictions["confidence"], errors="coerce") if not class_predictions.empty else pd.Series(dtype=float)
        rows.append(
            {
                "run_name": run_name,
                "class_id": class_id,
                "class_name": prediction_class_name(class_id),
                "prediction_count": int(len(class_predictions)),
                "avg_confidence": float(confidences.mean()) if not confidences.dropna().empty else math.nan,
            }
        )
    return pd.DataFrame(rows, columns=BY_CLASS_COLUMNS)


def summarize_run(run_dir: Path, test_images: Dict[str, str]) -> Tuple[Dict[str, object], pd.DataFrame, pd.DataFrame]:
    predictions = read_run_predictions(run_dir, test_images)
    run_images = infer_run_images(run_dir)
    expected_images = test_images or run_images
    if predictions.empty:
        detected_stems: Set[str] = set()
    else:
        detected_stems = set(str(stem) for stem in predictions["image_stem"].dropna())

    no_detection_stems = sorted(set(expected_images) - detected_stems) if expected_images else []
    no_detection = pd.DataFrame(
        [{"run_name": run_dir.name, "image_name": expected_images[stem]} for stem in no_detection_stems],
        columns=NO_DETECTION_COLUMNS,
    )

    confidences = pd.to_numeric(predictions["confidence"], errors="coerce") if not predictions.empty else pd.Series(dtype=float)
    summary = {
        "run_name": run_dir.name,
        "total_images": len(expected_images) if expected_images else len(detected_stems),
        "total_predictions": int(len(predictions)),
        "total_no_detection": int(len(no_detection)),
        "avg_confidence": float(confidences.mean()) if not confidences.dropna().empty else math.nan,
        "predict_dir": str(run_dir),
    }
    by_class = summarize_by_class(run_dir.name, predictions)
    return summary, by_class, no_detection


def collect_predictions(predict_dir: Path, dataset_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    test_images = dataset_test_images(dataset_dir)
    if not predict_dir.exists():
        print(f"Warning: predict directory not found: {predict_dir}")
        return (
            pd.DataFrame(columns=SUMMARY_COLUMNS),
            pd.DataFrame(columns=BY_CLASS_COLUMNS),
            pd.DataFrame(columns=NO_DETECTION_COLUMNS),
        )

    summary_rows: List[Dict[str, object]] = []
    by_class_frames: List[pd.DataFrame] = []
    no_detection_frames: List[pd.DataFrame] = []
    for run_dir in sorted(path for path in predict_dir.iterdir() if path.is_dir()):
        summary, by_class, no_detection = summarize_run(run_dir, test_images)
        summary_rows.append(summary)
        by_class_frames.append(by_class)
        no_detection_frames.append(no_detection)

    summary_df = pd.DataFrame(summary_rows, columns=SUMMARY_COLUMNS)
    by_class_df = (
        pd.concat(by_class_frames, ignore_index=True)
        if by_class_frames
        else pd.DataFrame(columns=BY_CLASS_COLUMNS)
    )
    no_detection_df = (
        pd.concat(no_detection_frames, ignore_index=True)
        if no_detection_frames
        else pd.DataFrame(columns=NO_DETECTION_COLUMNS)
    )
    return summary_df, by_class_df, no_detection_df


def notes_frame() -> pd.DataFrame:
    return pd.DataFrame({"notes": NOTES})


def ensure_parent(path: Path) -> None:
    if path.parent != Path("."):
        path.parent.mkdir(parents=True, exist_ok=True)


def main() -> None:
    args = parse_args()
    predict_dir = Path(args.predict_dir)
    dataset_dir = Path(args.dataset_dir)
    output_csv = Path(args.output_csv)
    output_excel = Path(args.output_excel)

    summary, by_class, no_detection = collect_predictions(predict_dir, dataset_dir)

    ensure_parent(output_csv)
    ensure_parent(output_excel)
    summary.to_csv(output_csv, index=False)
    with pd.ExcelWriter(output_excel, engine="openpyxl") as writer:
        summary.to_excel(writer, index=False, sheet_name="Prediction_Summary")
        by_class.to_excel(writer, index=False, sheet_name="Prediction_By_Class")
        no_detection.to_excel(writer, index=False, sheet_name="No_Detection_Images")
        notes_frame().to_excel(writer, index=False, sheet_name="Notes")

    print(f"Saved prediction summary CSV to {output_csv}")
    print(f"Saved prediction summary workbook to {output_excel}")


if __name__ == "__main__":
    main()
