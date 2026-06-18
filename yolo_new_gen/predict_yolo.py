import argparse
import csv
import cv2
from ultralytics import YOLO
from pathlib import Path


def resolve_device(requested_device):
    requested_device = str(requested_device or "auto").strip().lower()
    if requested_device in {"auto", ""}:
        try:
            import torch

            return "0" if torch.cuda.is_available() else "cpu"
        except Exception:
            return "cpu"
    if requested_device == "cpu":
        return "cpu"
    return requested_device


def predict():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--source", type=str, required=True)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.7)
    parser.add_argument("--max-det", type=int, default=300)
    parser.add_argument("--output", type=str, default="predictions")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--render-size", type=int, default=512)
    args = parser.parse_args()
    device = resolve_device(args.device)

    model = YOLO(args.model)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    prediction_rows = []

    print(f"Predicting with Upscale (untuk label jelas) on {device}...")

    # Ambil daftar gambar
    source_path = Path(args.source)
    if source_path.is_file():
        image_files = [source_path]
    else:
        image_files = []
        for pattern in ["*.png", "*.jpg", "*.jpeg"]:
            image_files.extend(source_path.glob(pattern))
    image_files = sorted(image_files)

    for img_path in image_files:
        results = model.predict(
            source=str(img_path),
            conf=args.conf,
            iou=args.iou,
            max_det=args.max_det,
            device=device,
            verbose=False,
        )
        
        # Baca gambar asli
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"WARNING: Could not read image: {img_path}")
            continue
        
        # Upscale the small SAR patch for readable visual inspection.
        img_h, img_w = img.shape[:2]
        img_large = cv2.resize(img, (args.render_size, args.render_size), interpolation=cv2.INTER_NEAREST)
        scale_x = args.render_size / img_w
        scale_y = args.render_size / img_h
        
        for r in results:
            boxes = r.boxes
            for box in boxes:
                # Ambil koordinat asli untuk CSV, lalu skala ke gambar render.
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                class_name = str(model.names[cls])
                prediction_rows.append(
                    {
                        "image_name": img_path.name,
                        "pred_class_id": cls,
                        "pred_class_name": class_name,
                        "confidence": conf,
                        "x1": x1,
                        "y1": y1,
                        "x2": x2,
                        "y2": y2,
                    }
                )

                x1_render = int(x1 * scale_x)
                y1_render = int(y1 * scale_y)
                x2_render = int(x2 * scale_x)
                y2_render = int(y2 * scale_y)
                label = f"{class_name} {conf:.2f}"
                
                # Gambar Kotak
                cv2.rectangle(img_large, (x1_render, y1_render), (x2_render, y2_render), (0, 255, 0), 2)
                
                # Gambar Label (Teks Putih dengan Background Hitam agar kontras)
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.6
                thickness = 1
                (w, h), _ = cv2.getTextSize(label, font, font_scale, thickness)
                pad = 4
                label_h = h + (pad * 2)
                label_w = w + (pad * 2)
                label_x1 = max(0, min(x1_render, args.render_size - label_w))
                if y1_render - label_h >= 0:
                    label_y1 = y1_render - label_h
                    text_y = y1_render - pad
                else:
                    label_y1 = min(args.render_size - label_h, y2_render)
                    text_y = label_y1 + h + pad
                label_x2 = label_x1 + label_w
                label_y2 = label_y1 + label_h
                cv2.rectangle(img_large, (label_x1, label_y1), (label_x2, label_y2), (0, 0, 0), -1)
                cv2.putText(img_large, label, (label_x1 + pad, text_y), font, font_scale, (255, 255, 255), thickness)

        # Simpan hasil
        save_path = output_dir / img_path.name
        cv2.imwrite(str(save_path), img_large)

    csv_path = output_dir / "predictions.csv"
    fieldnames = ["image_name", "pred_class_id", "pred_class_name", "confidence", "x1", "y1", "x2", "y2"]
    with csv_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(prediction_rows)

    print(f"\nSelesai! {len(image_files)} gambar dengan label besar disimpan di: {args.output}")
    print(f"Prediction table written to: {csv_path}")

if __name__ == "__main__":
    predict()
