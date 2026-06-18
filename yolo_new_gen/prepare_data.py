import argparse
import sys
from collections import Counter
from pathlib import Path
import re
import shutil

import pandas as pd
import tifffile
from PIL import Image
from sklearn.model_selection import train_test_split


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from labels import CLASS_NAMES, map_ship_class
from yolo_exp.bbox_utils import (
    axis_aligned_bbox_local,
    bbox_to_yolo,
    dual_pol_to_rgb_uint8,
    sar_to_rgb_uint8,
)


YOLO_IMAGE_SIZE = 128
DEFAULT_METADATA_PATH = PROJECT_ROOT / "new" / "metadata" / "metadata_with_vv_vh_gfw_ais_identity.csv"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "dataset_yolo_det_128_3class_vv_vh_rgb_scene"
DEFAULT_IMAGE_COLUMN = "patch"
DEFAULT_VV_COLUMN = "patch_vv_actual_file"
DEFAULT_VH_COLUMN = "patch_vh_actual_file"
DEFAULT_SPLIT_GROUP = "scene"
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15
REQUIRED_YOLO_COLUMNS = [
    "category",
    "Elaborated_type",
    DEFAULT_IMAGE_COLUMN,
    "Center_x",
    "Center_y",
    "UpperLeft_x",
    "UpperLeft_y",
    "LowerRight_x",
    "LowerRight_y",
]


def parse_image_columns(image_columns):
    columns = [column.strip() for column in str(image_columns).split(",") if column.strip()]
    if not columns:
        raise ValueError("At least one image column must be provided.")
    return columns


def base_patch_id(patch_name):
    stem = Path(str(patch_name)).stem.strip()
    stem = re.sub(r"_(vv|vh)$", "", stem, flags=re.IGNORECASE)
    return stem.lower()


def combined_rgb_patch_name(vv_patch_name):
    stem = Path(str(vv_patch_name)).stem.strip()
    stem = re.sub(r"_(vv|vh)$", "", stem, flags=re.IGNORECASE)
    return f"{stem}_vv_vh_rgb.png"


def build_split_group(row, patch_name, split_group):
    split_group = str(split_group).strip().lower()
    scene = str(row.get("scene", "")).strip()
    patch_id = base_patch_id(patch_name)

    if split_group == "none":
        return f"sample:{Path(str(patch_name)).name.lower()}"
    if split_group == "scene":
        return f"scene:{scene}" if scene else f"patch:{patch_id}"
    if split_group == "patch":
        return f"patch:{patch_id}"
    if split_group == "scene_patch":
        return f"scene:{scene}|patch:{patch_id}" if scene else f"patch:{patch_id}"

    raise ValueError(f"Unsupported split group: {split_group}")


def dominant_label(labels):
    counts = Counter(int(label) for label in labels)
    return sorted(counts.items(), key=lambda item: (-item[1], item[0]))[0][0]


def find_image_source_dir(project_root, explicit_dir=None):
    candidates = []
    if explicit_dir:
        candidates.append(Path(explicit_dir).expanduser().resolve())
    candidates.extend(
        [
            project_root / "Patch",
            project_root / "new" / "Patch",
            project_root / "PATCH",
            project_root / "new" / "PATCH",
            project_root / "dataset" / "Patch",
        ]
    )
    for path in candidates:
        if path.exists() and path.is_dir():
            return path

    searched = "\n".join(f"  - {path}" for path in candidates)
    raise FileNotFoundError(f"SAR TIFF source folder not found. Searched:\n{searched}")


def print_distribution(title, labels):
    counter = Counter(int(label) for label in labels)
    print(title)
    for class_id, class_name in enumerate(CLASS_NAMES):
        print(f"  {class_id} {class_name}: {counter[class_id]}")


def save_yolo_yaml(det_dir):
    dataset_root = Path(det_dir).resolve().as_posix()
    lines = [
        f'path: "{dataset_root}"',
        "train: train/images",
        "val: val/images",
        "test: test/images",
        f"nc: {len(CLASS_NAMES)}",
        "names:",
    ]
    lines.extend(f"  {class_id}: {class_name}" for class_id, class_name in enumerate(CLASS_NAMES))
    (det_dir / "data.yaml").write_text("\n".join(lines) + "\n", encoding="utf-8")


def format_yolo_bbox(label, coords):
    label = int(label)
    if label not in {0, 1, 2}:
        raise ValueError(f"Invalid class label for final 3-class dataset: {label}")

    coords = [float(coord) for coord in coords]
    if len(coords) != 4:
        raise ValueError(f"YOLO bbox must contain 4 values, got {len(coords)}")
    x_center, y_center, width, height = coords
    if not all(0.0 <= value <= 1.0 for value in coords):
        raise ValueError(f"YOLO bbox values must be normalized to [0, 1]: {coords}")
    if width <= 0.0 or height <= 0.0:
        raise ValueError(f"YOLO bbox width/height must be > 0: {coords}")

    return f"{label} {' '.join(f'{coord:.6f}' for coord in coords)}"


def resize_rgb_image(img_rgb, imgsz):
    image = Image.fromarray(img_rgb)
    if image.size != (imgsz, imgsz):
        resampling = getattr(Image, "Resampling", Image).BILINEAR
        image = image.resize((imgsz, imgsz), resampling)
    return image


def split_groups(group_df, test_size, random_state, stage_name):
    stratify = group_df["label"]
    try:
        return train_test_split(
            group_df,
            test_size=test_size,
            stratify=stratify,
            random_state=random_state,
        )
    except ValueError as exc:
        print(f"WARNING: Cannot stratify {stage_name} group split: {exc}")
        print("WARNING: Falling back to non-stratified group split for this stage.")
        return train_test_split(
            group_df,
            test_size=test_size,
            random_state=random_state,
        )


def prepare_group_split(df_filtered, seed=42):
    group_df = (
        df_filtered.groupby("split_group")
        .agg(
            label=("label", dominant_label),
            samples=("label", "size"),
        )
        .reset_index()
    )

    if len(group_df) < 3:
        raise ValueError("Need at least 3 split groups to create train/val/test splits.")

    train_groups, holdout_groups = split_groups(
        group_df,
        test_size=VAL_RATIO + TEST_RATIO,
        random_state=seed,
        stage_name="train/holdout",
    )
    val_groups, test_groups = split_groups(
        holdout_groups,
        test_size=TEST_RATIO / (VAL_RATIO + TEST_RATIO),
        random_state=seed,
        stage_name="validation/test",
    )

    split_by_group = {}
    split_by_group.update({group: "train" for group in train_groups["split_group"]})
    split_by_group.update({group: "val" for group in val_groups["split_group"]})
    split_by_group.update({group: "test" for group in test_groups["split_group"]})

    df_split = df_filtered.copy()
    df_split["split"] = df_split["split_group"].map(split_by_group)
    if df_split["split"].isna().any():
        raise RuntimeError("Some samples were not assigned to a split.")

    group_leakage = df_split.groupby("split_group")["split"].nunique()
    leaked_groups = group_leakage[group_leakage > 1]
    if not leaked_groups.empty:
        raise RuntimeError(
            "Group split leakage detected for these split groups: "
            + ", ".join(leaked_groups.index[:10])
        )

    return (
        df_split[df_split["split"] == "train"].copy(),
        df_split[df_split["split"] == "val"].copy(),
        df_split[df_split["split"] == "test"].copy(),
        group_df,
    )


def prepare_data_clean(
    metadata_path,
    image_source_dir,
    det_dir,
    imgsz,
    image_columns=DEFAULT_IMAGE_COLUMN,
    split_group=DEFAULT_SPLIT_GROUP,
    combine_vv_vh=False,
    vv_column=DEFAULT_VV_COLUMN,
    vh_column=DEFAULT_VH_COLUMN,
    seed=42,
):
    metadata_path = Path(metadata_path).resolve()
    image_source_dir = Path(image_source_dir).resolve()
    det_dir = Path(det_dir).resolve()
    image_columns = parse_image_columns(image_columns)

    if not metadata_path.exists():
        raise FileNotFoundError(f"metadata.csv not found: {metadata_path}")

    df = pd.read_csv(metadata_path)
    required_columns = list(REQUIRED_YOLO_COLUMNS)
    if combine_vv_vh:
        required_columns.extend(
            column for column in [vv_column, vh_column] if column not in required_columns
        )
    else:
        required_columns.extend(column for column in image_columns if column not in required_columns)
    if split_group in {"scene", "scene_patch"} and "scene" not in required_columns:
        required_columns.append("scene")
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise KeyError(
            "metadata.csv is missing required columns for YOLO labels/bboxes: "
            + ", ".join(missing_columns)
        )

    prepared_data = []
    dropped_label = 0
    missing_image = Counter()
    missing_patch = Counter()
    shape_mismatch = 0
    duplicate_image = 0
    seen_image_paths = set()

    print("Preparing YOLO detection dataset")
    print(f"Metadata: {metadata_path}")
    print(f"SAR TIFF source: {image_source_dir}")
    print(f"Output dataset: {det_dir}")
    print(f"YOLO PNG image size: {imgsz}x{imgsz}")
    if combine_vv_vh:
        print(f"Image mode: combined RGB from {vv_column} + {vh_column}")
        print("RGB channels: R=VV, G=VH, B=mean(VV,VH)")
    else:
        print(f"Image columns: {', '.join(image_columns)}")
    print(f"Split group: {split_group}")
    print(f"Split seed: {seed}")

    for _, row in df.iterrows():
        label = map_ship_class(row)
        if label is None:
            dropped_label += 1
            continue

        if combine_vv_vh:
            vv_patch_name = row.get(vv_column, "")
            vh_patch_name = row.get(vh_column, "")
            if pd.isna(vv_patch_name) or str(vv_patch_name).strip() == "":
                missing_patch[vv_column] += 1
                continue
            if pd.isna(vh_patch_name) or str(vh_patch_name).strip() == "":
                missing_patch[vh_column] += 1
                continue

            vv_path = image_source_dir / str(vv_patch_name)
            vh_path = image_source_dir / str(vh_patch_name)
            if not vv_path.exists():
                missing_image[vv_column] += 1
                continue
            if not vh_path.exists():
                missing_image[vh_column] += 1
                continue

            image_key = f"{vv_path.resolve()}|{vh_path.resolve()}".lower()
            if image_key in seen_image_paths:
                duplicate_image += 1
                continue
            seen_image_paths.add(image_key)

            vv_img = tifffile.imread(vv_path)
            vh_img = tifffile.imread(vh_path)
            if vv_img.shape[:2] != vh_img.shape[:2]:
                shape_mismatch += 1
                continue

            bbox_xyxy = axis_aligned_bbox_local(row, image_shape=vv_img.shape, pad=None)
            yolo_coords = bbox_to_yolo(bbox_xyxy, image_shape=vv_img.shape)
            yolo_bbox = format_yolo_bbox(label, yolo_coords)

            prepared_data.append(
                {
                    "img_path": "",
                    "vv_img_path": vv_path,
                    "vh_img_path": vh_path,
                    "patch_name": combined_rgb_patch_name(vv_patch_name),
                    "scene": str(row.get("scene", "")),
                    "category": str(row.get("category", "")),
                    "source_vv_patch_name": str(vv_patch_name),
                    "source_vh_patch_name": str(vh_patch_name),
                    "image_column": f"{vv_column}+{vh_column}",
                    "image_mode": "vv_vh_rgb",
                    "split_group": build_split_group(row, vv_patch_name, split_group),
                    "label": int(label),
                    "yolo_bbox": yolo_bbox,
                }
            )
            continue

        for image_column in image_columns:
            patch_name = row.get(image_column, "")
            if pd.isna(patch_name) or str(patch_name).strip() == "":
                missing_patch[image_column] += 1
                continue

            img_path = image_source_dir / str(patch_name)
            if not img_path.exists():
                missing_image[image_column] += 1
                continue

            image_key = str(img_path.resolve()).lower()
            if image_key in seen_image_paths:
                duplicate_image += 1
                continue
            seen_image_paths.add(image_key)

            img_tiff = tifffile.imread(img_path)
            bbox_xyxy = axis_aligned_bbox_local(row, image_shape=img_tiff.shape, pad=None)
            yolo_coords = bbox_to_yolo(bbox_xyxy, image_shape=img_tiff.shape)
            yolo_bbox = format_yolo_bbox(label, yolo_coords)

            prepared_data.append(
                {
                    "img_path": img_path,
                    "vv_img_path": "",
                    "vh_img_path": "",
                    "patch_name": str(patch_name),
                    "scene": str(row.get("scene", "")),
                    "category": str(row.get("category", "")),
                    "source_vv_patch_name": "",
                    "source_vh_patch_name": "",
                    "image_column": image_column,
                    "image_mode": "single",
                    "split_group": build_split_group(row, patch_name, split_group),
                    "label": int(label),
                    "yolo_bbox": yolo_bbox,
                }
            )

    df_filtered = pd.DataFrame(prepared_data)
    print(f"Rows dropped as non-target classes: {dropped_label}")
    if combine_vv_vh:
        for image_column in [vv_column, vh_column]:
            print(f"Rows missing {image_column}: {missing_patch[image_column]}")
            print(f"Rows with missing image files in {image_column}: {missing_image[image_column]}")
        print(f"Rows with VV/VH shape mismatch: {shape_mismatch}")
    else:
        for image_column in image_columns:
            print(f"Rows missing {image_column}: {missing_patch[image_column]}")
            print(f"Rows with missing image files in {image_column}: {missing_image[image_column]}")
    print(f"Duplicate image references skipped: {duplicate_image}")
    if df_filtered.empty:
        raise ValueError("No YOLO samples remain after label filtering and image lookup.")

    print_distribution("YOLO class distribution after filtering:", df_filtered["label"].tolist())
    class_counts = df_filtered["label"].value_counts()
    too_small = class_counts[class_counts < 2]
    if not too_small.empty:
        raise ValueError(
            "Cannot create stratified YOLO split because these classes have fewer than 2 samples: "
            + ", ".join(f"{CLASS_NAMES[int(class_id)]}={count}" for class_id, count in too_small.items())
        )

    train_df, val_df, test_df, group_df = prepare_group_split(df_filtered, seed=seed)
    print(f"YOLO split ratio: train={TRAIN_RATIO:.0%}, val={VAL_RATIO:.0%}, test={TEST_RATIO:.0%}")
    print(f"Total split groups: {len(group_df)}")
    print(
        "Actual sample split: "
        f"train={len(train_df)} ({len(train_df) / len(df_filtered):.1%}), "
        f"val={len(val_df)} ({len(val_df) / len(df_filtered):.1%}), "
        f"test={len(test_df)} ({len(test_df) / len(df_filtered):.1%})"
    )
    print_distribution("YOLO train class distribution:", train_df["label"].tolist())
    print_distribution("YOLO validation class distribution:", val_df["label"].tolist())
    print_distribution("YOLO test class distribution:", test_df["label"].tolist())

    manifest_frames = []
    for split_name, split_df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        shutil.rmtree(det_dir / split_name, ignore_errors=True)
        (det_dir / split_name / "images").mkdir(parents=True, exist_ok=True)
        (det_dir / split_name / "labels").mkdir(parents=True, exist_ok=True)

        print(f"Writing {split_name} split ({len(split_df)} samples)...")
        for _, row in split_df.iterrows():
            if row["image_mode"] == "vv_vh_rgb":
                img_rgb = dual_pol_to_rgb_uint8(
                    tifffile.imread(row["vv_img_path"]),
                    tifffile.imread(row["vh_img_path"]),
                )
                new_name = Path(row["patch_name"]).name
            else:
                img_rgb = sar_to_rgb_uint8(tifffile.imread(row["img_path"]))
                new_name = Path(row["patch_name"]).with_suffix(".png").name
            resize_rgb_image(img_rgb, imgsz).save(det_dir / split_name / "images" / new_name)
            label_path = det_dir / split_name / "labels" / (Path(new_name).stem + ".txt")
            label_path.write_text(row["yolo_bbox"] + "\n", encoding="utf-8")

        manifest_frames.append(split_df)

    manifest_df = pd.concat(manifest_frames, ignore_index=True)
    manifest_df[
        [
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
    ].to_csv(det_dir / "split_manifest.csv", index=False)
    save_yolo_yaml(det_dir)
    print(f"Split manifest written to: {det_dir / 'split_manifest.csv'}")
    print(f"YOLO data.yaml written to: {det_dir / 'data.yaml'}")

    from yolo_new_gen.validate_dataset import validate_dataset

    validate_dataset(
        det_dir,
        require_image_mode="vv_vh_rgb" if combine_vv_vh else None,
        verbose=True,
    )


def main():
    parser = argparse.ArgumentParser(description="Prepare YOLO detection data from SAR TIFF patches.")
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA_PATH)
    parser.add_argument("--image-dir", type=Path, default=None, help="Source folder containing SAR TIFF patches")
    parser.add_argument(
        "--image-column",
        type=str,
        default=DEFAULT_IMAGE_COLUMN,
        help="Backward-compatible single metadata column containing TIFF file names.",
    )
    parser.add_argument(
        "--image-columns",
        type=str,
        default=None,
        help="Comma-separated metadata columns containing TIFF file names, e.g. patch,patch_vh_actual_file.",
    )
    parser.add_argument(
        "--split-group",
        type=str,
        default=DEFAULT_SPLIT_GROUP,
        choices=["scene_patch", "scene", "patch", "none"],
        help="Group key for train/val/test split to prevent leakage.",
    )
    parser.add_argument(
        "--combine-vv-vh",
        action="store_true",
        help="Create one RGB PNG per row using VV as red, VH as green, and mean(VV,VH) as blue.",
    )
    parser.add_argument(
        "--vv-column",
        type=str,
        default=DEFAULT_VV_COLUMN,
        help="Metadata column containing the VV TIFF file name for --combine-vv-vh.",
    )
    parser.add_argument(
        "--vh-column",
        type=str,
        default=DEFAULT_VH_COLUMN,
        help="Metadata column containing the VH TIFF file name for --combine-vv-vh.",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--imgsz", type=int, default=YOLO_IMAGE_SIZE, help="Output PNG size for YOLO training")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for scene/group splitting")
    args = parser.parse_args()

    image_source_dir = find_image_source_dir(PROJECT_ROOT, args.image_dir)
    prepare_data_clean(
        args.metadata,
        image_source_dir,
        args.output_dir,
        args.imgsz,
        args.image_columns or args.image_column,
        args.split_group,
        args.combine_vv_vh,
        args.vv_column,
        args.vh_column,
        args.seed,
    )


if __name__ == "__main__":
    main()
