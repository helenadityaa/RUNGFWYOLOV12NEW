import argparse
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def resolve_data_path(script_dir, data_value):
    data_path = Path(data_value)
    if data_path.is_absolute():
        return data_path

    candidates = [
        PROJECT_ROOT / data_path,
        script_dir / data_path,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", type=str, default="12", choices=["12"])
    parser.add_argument("--variant", type=str, default="n", choices=["n", "s", "m", "x", "l"])
    parser.add_argument("--epochs", type=int, default=50, choices=[50, 100, 150])
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--imgsz", type=int, default=128)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument(
        "--data",
        type=str,
        default="yolo_new_gen/dataset_yolo_det_128_3class_vv_vh_rgb_scene/data.yaml",
    )
    parser.add_argument("--output", type=str, default="")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--patience", type=int, default=100)
    parser.add_argument("--lr0", type=float, default=0.001)
    parser.add_argument("--lrf", type=float, default=0.01)
    parser.add_argument("--cos-lr", action="store_true")
    parser.add_argument("--weight-decay", type=float, default=0.0005)
    parser.add_argument("--mosaic", type=float, default=0.3)
    parser.add_argument("--mixup", type=float, default=0.0)
    parser.add_argument("--close-mosaic", type=int, default=10)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    data_path = resolve_data_path(script_dir, args.data)
    if not data_path.exists():
        raise SystemExit(
            f"Detection dataset config not found: {data_path}\n"
            "Pakai --data <path-ke-data.yaml> untuk dataset lain."
        )

    cmd = [
        sys.executable,
        str(script_dir / "train_yolo.py"),
        "--task",
        "detect",
        "--data",
        str(data_path),
        "--version",
        args.version,
        "--variant",
        args.variant,
        "--epochs",
        str(args.epochs),
        "--batch",
        str(args.batch),
        "--imgsz",
        str(args.imgsz),
        "--device",
        args.device,
        "--seed",
        str(args.seed),
        "--patience",
        str(args.patience),
        "--lr0",
        str(args.lr0),
        "--lrf",
        str(args.lrf),
        "--weight-decay",
        str(args.weight_decay),
        "--mosaic",
        str(args.mosaic),
        "--mixup",
        str(args.mixup),
        "--close-mosaic",
        str(args.close_mosaic),
        "--workers",
        str(args.workers),
    ]
    if args.output:
        cmd.extend(["--output", args.output])
    if args.deterministic:
        cmd.append("--deterministic")
    if args.cos_lr:
        cmd.append("--cos-lr")

    print(f"Running Detection: {' '.join(cmd)}")
    subprocess.run(cmd, check=True, cwd=PROJECT_ROOT)

if __name__ == "__main__":
    main()
