"""Evaluate an existing YOLO best.pt on the test split without training."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any, Dict, Iterable, Optional


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run YOLO validation on split=test for an existing model checkpoint."
    )
    parser.add_argument("--model", required=True, help="Path to an existing best.pt model file.")
    parser.add_argument(
        "--data",
        default="yolo_new_gen/dataset_yolo_det_128_3class_vv_vh_rgb_scene/data.yaml",
        help="Path to data.yaml for the final 3-class dataset.",
    )
    parser.add_argument("--imgsz", type=int, default=128, help="Validation image size.")
    parser.add_argument("--batch", type=int, default=16, help="Validation batch size.")
    parser.add_argument("--device", default="cpu", help="Validation device, for example cpu or 0.")
    parser.add_argument("--project", default="runs/detect_test", help="Output project directory.")
    parser.add_argument("--name", default=None, help="Optional validation run name.")
    return parser.parse_args()


def derive_run_name(model_path: Path, provided_name: Optional[str]) -> str:
    if provided_name:
        return provided_name
    if model_path.name == "best.pt" and model_path.parent.name == "weights":
        return f"{model_path.parent.parent.name}_test"
    return f"{model_path.stem}_test"


def get_nested_attr(obj: Any, attr_path: str) -> Optional[float]:
    current = obj
    for part in attr_path.split("."):
        if not hasattr(current, part):
            return None
        current = getattr(current, part)
    try:
        return float(current)
    except (TypeError, ValueError):
        return None


def get_from_results_dict(metrics: Any, keys: Iterable[str]) -> Optional[float]:
    results = getattr(metrics, "results_dict", None)
    if not isinstance(results, dict):
        return None
    normalized = {str(key).strip().lower(): value for key, value in results.items()}
    for key in keys:
        value = normalized.get(key.strip().lower())
        if value is None:
            continue
        try:
            return float(value)
        except (TypeError, ValueError):
            continue
    return None


def extract_metric(metrics: Any, attr_paths: Iterable[str], result_keys: Iterable[str]) -> Optional[float]:
    for attr_path in attr_paths:
        value = get_nested_attr(metrics, attr_path)
        if value is not None:
            return value
    return get_from_results_dict(metrics, result_keys)


def write_metrics_csv(output_path: Path, row: Dict[str, object]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
        writer.writeheader()
        writer.writerow(row)


def resolve_from_repo(path_value: str) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    repo_root = Path(__file__).resolve().parents[1]
    return (repo_root / path).resolve()


def disable_ultralytics_label_cache_writes() -> None:
    """Keep validation from creating labels.cache inside the dataset directory."""
    try:
        import ultralytics.data.dataset as dataset_module
        import ultralytics.data.utils as utils_module
    except Exception:
        return

    def skip_cache_write(*args: object, **kwargs: object) -> None:
        return None

    dataset_module.save_dataset_cache_file = skip_cache_write
    utils_module.save_dataset_cache_file = skip_cache_write


def main() -> None:
    args = parse_args()
    model_path = resolve_from_repo(args.model)
    data_path = resolve_from_repo(args.data)
    project_dir = resolve_from_repo(args.project)
    run_name = derive_run_name(model_path, args.name)

    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    if not data_path.exists():
        raise FileNotFoundError(f"data.yaml not found: {data_path}")

    disable_ultralytics_label_cache_writes()

    from ultralytics import YOLO

    model = YOLO(str(model_path))
    metrics = model.val(
        data=str(data_path),
        split="test",
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=str(project_dir),
        name=run_name,
        exist_ok=True,
    )

    save_dir = Path(getattr(metrics, "save_dir", project_dir / run_name))
    row: Dict[str, object] = {
        "model_path": str(model_path),
        "data_path": str(data_path),
        "split": "test",
        "imgsz": args.imgsz,
        "batch": args.batch,
        "device": args.device,
        "precision": extract_metric(metrics, ["box.mp"], ["metrics/precision(B)", "precision"]),
        "recall": extract_metric(metrics, ["box.mr"], ["metrics/recall(B)", "recall"]),
        "mAP50": extract_metric(metrics, ["box.map50"], ["metrics/mAP50(B)", "mAP50"]),
        "mAP50-95": extract_metric(metrics, ["box.map"], ["metrics/mAP50-95(B)", "mAP50-95"]),
    }
    output_csv = save_dir / "test_metrics.csv"
    write_metrics_csv(output_csv, row)
    print(f"Saved test metrics to {output_csv}")


if __name__ == "__main__":
    main()
