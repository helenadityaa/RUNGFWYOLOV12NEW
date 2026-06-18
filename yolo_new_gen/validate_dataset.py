import argparse
import re
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_DATASET_DIR = SCRIPT_DIR / "dataset_yolo_det_128_3class_vv_vh_rgb_scene"
EXPECTED_NAMES = {0: "Fishing", 1: "Cargo", 2: "Passenger"}
VALID_CLASS_IDS = set(EXPECTED_NAMES)
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg"}
REQUIRED_MANIFEST_COLUMNS = [
    "split",
    "split_group",
    "scene",
    "category",
    "image_mode",
    "patch_name",
    "source_vv_patch_name",
    "source_vh_patch_name",
    "label",
    "yolo_bbox",
]


def parse_data_yaml(data_yaml):
    parsed = {"nc": None, "names": {}}
    in_names = False

    for raw_line in Path(data_yaml).read_text(encoding="utf-8").splitlines():
        line = raw_line.split("#", 1)[0].rstrip()
        stripped = line.strip()
        if not stripped:
            continue

        if stripped.startswith("nc:"):
            try:
                parsed["nc"] = int(stripped.split(":", 1)[1].strip())
            except ValueError:
                parsed["nc"] = stripped.split(":", 1)[1].strip()
            in_names = False
            continue

        if stripped == "names:":
            in_names = True
            continue

        if in_names:
            match = re.match(r"^\s*(\d+):\s*(.+?)\s*$", line)
            if match:
                parsed["names"][int(match.group(1))] = match.group(2).strip().strip("'\"")
                continue
            if not line.startswith((" ", "\t")):
                in_names = False

    return parsed


def image_files(images_dir):
    if not images_dir.exists():
        return []
    return sorted(path for path in images_dir.iterdir() if path.suffix.lower() in IMAGE_EXTENSIONS)


def label_files(labels_dir):
    if not labels_dir.exists():
        return []
    return sorted(labels_dir.glob("*.txt"))


def validate_label_line(line, label_path, line_no, errors):
    parts = line.split()
    if len(parts) != 5:
        errors.append(f"{label_path}:{line_no} must have 5 values: class x_center y_center width height")
        return

    try:
        class_id = int(float(parts[0]))
    except ValueError:
        errors.append(f"{label_path}:{line_no} has non-numeric class id: {parts[0]}")
        return

    if class_id not in VALID_CLASS_IDS:
        errors.append(f"{label_path}:{line_no} has label outside 0,1,2: {class_id}")

    try:
        x_center, y_center, width, height = [float(value) for value in parts[1:]]
    except ValueError:
        errors.append(f"{label_path}:{line_no} has non-numeric bbox values: {' '.join(parts[1:])}")
        return

    coords = [x_center, y_center, width, height]
    if any(value < 0.0 or value > 1.0 for value in coords):
        errors.append(f"{label_path}:{line_no} bbox values are outside [0, 1]: {coords}")
    if width <= 0.0 or height <= 0.0:
        errors.append(f"{label_path}:{line_no} bbox width/height must be > 0: {coords}")


def validate_split_files(dataset_dir, errors):
    for split in ["train", "val", "test"]:
        images_dir = dataset_dir / split / "images"
        labels_dir = dataset_dir / split / "labels"
        if not images_dir.exists():
            errors.append(f"Missing images directory: {images_dir}")
            continue
        if not labels_dir.exists():
            errors.append(f"Missing labels directory: {labels_dir}")
            continue

        images = image_files(images_dir)
        labels = label_files(labels_dir)
        image_stems = {path.stem for path in images}
        label_stems = {path.stem for path in labels}

        missing_labels = sorted(image_stems - label_stems)
        extra_labels = sorted(label_stems - image_stems)
        if missing_labels:
            errors.append(f"{split}: images without label files: {', '.join(missing_labels[:10])}")
        if extra_labels:
            errors.append(f"{split}: label files without images: {', '.join(extra_labels[:10])}")
        if len(images) != len(labels):
            errors.append(f"{split}: image/label count mismatch: images={len(images)}, labels={len(labels)}")

        for label_path in labels:
            content = label_path.read_text(encoding="utf-8").strip()
            if not content:
                errors.append(f"{label_path} is empty")
                continue
            for line_no, line in enumerate(content.splitlines(), start=1):
                validate_label_line(line.strip(), label_path, line_no, errors)


def validate_manifest(dataset_dir, require_image_mode, errors):
    manifest_path = dataset_dir / "split_manifest.csv"
    if not manifest_path.exists():
        errors.append(f"Missing split manifest: {manifest_path}")
        return

    manifest_df = pd.read_csv(manifest_path)
    missing_columns = [column for column in REQUIRED_MANIFEST_COLUMNS if column not in manifest_df.columns]
    if missing_columns:
        errors.append("split_manifest.csv missing columns: " + ", ".join(missing_columns))
        return

    invalid_labels = sorted(set(manifest_df["label"].dropna().astype(int)) - VALID_CLASS_IDS)
    if invalid_labels:
        errors.append("split_manifest.csv has labels outside 0,1,2: " + ", ".join(map(str, invalid_labels)))

    if manifest_df["split_group"].isna().any():
        errors.append("split_manifest.csv has blank split_group values")
    else:
        split_counts = manifest_df.groupby("split_group")["split"].nunique()
        leaked = split_counts[split_counts > 1]
        if not leaked.empty:
            errors.append("split_group leakage across splits: " + ", ".join(map(str, leaked.index[:10])))

    if require_image_mode is not None:
        image_modes = set(manifest_df["image_mode"].dropna().astype(str))
        if image_modes != {require_image_mode}:
            errors.append(
                f"image_mode must be all {require_image_mode}; found: "
                + ", ".join(sorted(image_modes))
            )

    for idx, row in manifest_df.iterrows():
        label = int(row["label"])
        yolo_bbox = str(row["yolo_bbox"])
        if not yolo_bbox.startswith(f"{label} "):
            errors.append(f"split_manifest.csv row {idx + 2}: yolo_bbox label does not match label column")
        validate_label_line(yolo_bbox, manifest_path, idx + 2, errors)


def validate_dataset(dataset_dir=DEFAULT_DATASET_DIR, data_yaml=None, require_image_mode="vv_vh_rgb", verbose=True):
    dataset_dir = Path(dataset_dir).resolve()
    data_yaml = Path(data_yaml).resolve() if data_yaml else dataset_dir / "data.yaml"
    errors = []

    if not data_yaml.exists():
        errors.append(f"Missing data.yaml: {data_yaml}")
    else:
        data_config = parse_data_yaml(data_yaml)
        if data_config["nc"] != 3:
            errors.append(f"data.yaml nc must be 3; found {data_config['nc']}")
        if data_config["names"] != EXPECTED_NAMES:
            errors.append(f"data.yaml names must be {EXPECTED_NAMES}; found {data_config['names']}")

    validate_manifest(dataset_dir, require_image_mode, errors)
    validate_split_files(dataset_dir, errors)

    if errors:
        if verbose:
            print("FAIL: Dataset validation found problems.")
            for error in errors:
                print(f"- {error}")
        return False

    if verbose:
        print("PASS: Dataset valid.")
    return True


def main():
    parser = argparse.ArgumentParser(description="Validate the final 3-class VV/VH RGB YOLO detection dataset.")
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--data", type=Path, default=None, help="Optional explicit path to data.yaml")
    args = parser.parse_args()

    ok = validate_dataset(args.dataset_dir, args.data, require_image_mode="vv_vh_rgb", verbose=True)
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
