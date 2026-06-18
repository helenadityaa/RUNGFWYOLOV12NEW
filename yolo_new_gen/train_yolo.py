import argparse
import os
import random
from pathlib import Path

import numpy as np


def set_global_seed(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    except Exception as exc:
        print(f"WARNING: Could not fully set torch seed: {exc}")


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

    try:
        import torch

        if not torch.cuda.is_available():
            print(
                f"WARNING: Requested CUDA device '{requested_device}', "
                "but torch.cuda.is_available() is False. Falling back to CPU."
            )
            return "cpu"
        if torch.cuda.device_count() == 0:
            print(f"WARNING: Requested CUDA device '{requested_device}', but no CUDA devices were found. Falling back to CPU.")
            return "cpu"
    except Exception:
        print(f"WARNING: Could not check CUDA for device '{requested_device}'. Falling back to CPU.")
        return "cpu"

    return requested_device


def train_yolo():
    parser = argparse.ArgumentParser(description="YOLOv12 Detection Training Script")
    parser.add_argument("--version", type=str, default="12", choices=["12"], help="Only YOLOv12 is supported in this repo.")
    parser.add_argument("--variant", type=str, default="s", choices=["n", "s", "m", "x", "l"])
    parser.add_argument("--task", type=str, default="detect", choices=["detect"])
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=50, choices=[50, 100, 150])
    parser.add_argument("--imgsz", type=int, default=128, help="YOLO training image size")
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", type=str, default="auto", help="Use 'auto', 'cpu', or CUDA ids like '0'.")
    parser.add_argument("--output", type=str, default="runs/detect/yolo_result")
    parser.add_argument("--model", type=str, default="", help="Optional explicit YOLO checkpoint path/name, e.g. yolo12n.pt")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--optimizer", type=str, default="auto")
    parser.add_argument("--weight-decay", type=float, default=0.0005)
    parser.add_argument("--warmup-epochs", type=float, default=3.0)
    parser.add_argument("--mosaic", type=float, default=0.3)
    parser.add_argument("--mixup", type=float, default=0.0)
    parser.add_argument("--close-mosaic", type=int, default=10)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--lr0", type=float, default=0.001, help="Initial learning rate.")
    parser.add_argument("--lrf", type=float, default=0.01, help="Final learning-rate fraction.")
    parser.add_argument("--cos-lr", action="store_true", help="Use cosine learning-rate schedule.")
    parser.add_argument("--patience", type=int, default=100, help="Early-stopping patience.")

    args = parser.parse_args()
    set_global_seed(args.seed)
    device = resolve_device(args.device)

    from ultralytics import YOLO

    if args.model:
        model_pt = args.model
    else:
        model_pt = f"yolo12{args.variant}.pt"

    print("\n--- Memulai Training YOLO ---")
    print(f"Model variant : {args.variant}")
    print(f"Epochs        : {args.epochs}")
    print(f"Dataset path  : {args.data}")
    print(f"Image size    : {args.imgsz}")
    print(f"Batch         : {args.batch}")
    print(f"Seed          : {args.seed}")
    print(f"Deterministic : {args.deterministic}")
    print(f"Optimizer     : {args.optimizer}")
    print(f"LR0           : {args.lr0}")
    print(f"LRF           : {args.lrf}")
    print(f"Weight decay  : {args.weight_decay}")
    print(f"Mosaic        : {args.mosaic}")
    print(f"Mixup         : {args.mixup}")
    print(f"Close mosaic  : {args.close_mosaic}")
    print(f"Workers       : {args.workers}")
    print(f"Cos LR        : {args.cos_lr}")
    print(f"Model         : {model_pt}")
    print(f"Device        : {device}")
    print(f"Output        : {args.output}")
    print(f"------------------------\n")

    model = YOLO(model_pt)

    # Path output mutlak agar tidak ada nesting
    output_path = Path(args.output).resolve()
    project_path = str(output_path.parent)
    folder_name = output_path.name

    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=device,
        project=project_path,
        name=folder_name,
        exist_ok=True,
        plots=True,
        patience=args.patience,
        seed=args.seed,
        deterministic=args.deterministic,
        optimizer=args.optimizer,
        lr0=args.lr0,
        lrf=args.lrf,
        cos_lr=args.cos_lr,
        weight_decay=args.weight_decay,
        warmup_epochs=args.warmup_epochs,
        mosaic=args.mosaic,
        mixup=args.mixup,
        close_mosaic=args.close_mosaic,
        workers=args.workers,
    )

if __name__ == "__main__":
    train_yolo()
