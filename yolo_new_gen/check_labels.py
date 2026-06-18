import argparse
import random
import sys
from pathlib import Path

import cv2


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from labels import CLASS_NAMES

DEFAULT_DATASET_DIR = SCRIPT_DIR / "dataset_yolo_det_128_3class_vv_vh_rgb_scene"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "label_check"
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg"}
CLASS_COLORS = {
    0: (0, 255, 0),
    1: (255, 180, 0),
    2: (0, 120, 255),
}


def collect_images(images_dir):
    if not images_dir.exists():
        return []
    return sorted(path for path in images_dir.iterdir() if path.suffix.lower() in IMAGE_EXTENSIONS)


def read_yolo_labels(label_path, width, height):
    boxes = []
    if not label_path.exists():
        return boxes

    for line in label_path.read_text(encoding="utf-8").splitlines():
        parts = line.strip().split()
        if len(parts) != 5:
            continue

        class_id = int(float(parts[0]))
        x_center, y_center, box_w, box_h = [float(value) for value in parts[1:]]
        x1 = int(round((x_center - box_w / 2.0) * width))
        y1 = int(round((y_center - box_h / 2.0) * height))
        x2 = int(round((x_center + box_w / 2.0) * width))
        y2 = int(round((y_center + box_h / 2.0) * height))
        boxes.append(
            (
                class_id,
                max(0, min(width - 1, x1)),
                max(0, min(height - 1, y1)),
                max(0, min(width - 1, x2)),
                max(0, min(height - 1, y2)),
            )
        )
    return boxes


def draw_boxes(image, boxes):
    for class_id, x1, y1, x2, y2 in boxes:
        color = CLASS_COLORS.get(class_id, (0, 255, 255))
        class_name = CLASS_NAMES[class_id] if 0 <= class_id < len(CLASS_NAMES) else str(class_id)
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        label = f"{class_id} {class_name}"
        (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        label_y = max(0, y1 - text_h - 6)
        cv2.rectangle(image, (x1, label_y), (x1 + text_w + 6, label_y + text_h + 6), color, -1)
        cv2.putText(
            image,
            label,
            (x1 + 3, label_y + text_h + 3),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        )


def check_split(dataset_dir, output_dir, split, samples_per_split, rng):
    images_dir = dataset_dir / split / "images"
    labels_dir = dataset_dir / split / "labels"
    split_output_dir = output_dir / split
    split_output_dir.mkdir(parents=True, exist_ok=True)

    images = collect_images(images_dir)
    if not images:
        print(f"WARNING: No images found for split {split}: {images_dir}")
        return 0

    selected = rng.sample(images, min(samples_per_split, len(images)))
    saved = 0
    for image_path in selected:
        image = cv2.imread(str(image_path))
        if image is None:
            print(f"WARNING: Could not read image: {image_path}")
            continue

        height, width = image.shape[:2]
        label_path = labels_dir / f"{image_path.stem}.txt"
        boxes = read_yolo_labels(label_path, width, height)
        draw_boxes(image, boxes)

        output_path = split_output_dir / image_path.name
        cv2.imwrite(str(output_path), image)
        saved += 1

    return saved


def main():
    parser = argparse.ArgumentParser(description="Render random ground-truth YOLO boxes for visual inspection.")
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--samples-per-split", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    total = 0
    for split in ["train", "val", "test"]:
        saved = check_split(args.dataset_dir, args.output_dir, split, args.samples_per_split, rng)
        total += saved
        print(f"{split}: saved {saved} label-check images to {args.output_dir / split}")

    print(f"Done. Saved {total} label-check images.")


if __name__ == "__main__":
    main()
