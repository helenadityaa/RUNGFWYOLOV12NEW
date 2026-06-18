"""Collect metrics from existing YOLO training runs without running training."""

from __future__ import annotations

import argparse
import math
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd


OUTPUT_COLUMNS = [
    "run_name",
    "variant",
    "planned_epochs",
    "best_epoch",
    "precision",
    "recall",
    "mAP50",
    "mAP50-95",
    "fitness",
    "train_box_loss",
    "val_box_loss",
    "train_cls_loss",
    "val_cls_loss",
    "run_dir",
]

NOTES = [
    "Rekap ini membaca hasil training yang sudah ada.",
    "Script tidak menjalankan training ulang.",
    "Model terbaik dipilih berdasarkan mAP50-95 tertinggi.",
    "Dataset final menggunakan 3 kelas: Fishing, Cargo, Passenger.",
    "Input citra menggunakan RGB gabungan VV dan VH.",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize existing YOLO training results from runs/detect."
    )
    parser.add_argument("--runs-dir", default="runs/detect", help="Directory containing YOLO runs.")
    parser.add_argument(
        "--output-csv",
        default="summary_existing_runs.csv",
        help="Output CSV summary path.",
    )
    parser.add_argument(
        "--output-excel",
        default="summary_existing_runs.xlsx",
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


def numeric_value(row: pd.Series, column: Optional[str]) -> float:
    if column is None:
        return math.nan
    value = pd.to_numeric(row.get(column), errors="coerce")
    return float(value) if not pd.isna(value) else math.nan


def parse_run_name(run_name: str) -> Dict[str, object]:
    variant = None
    planned_epochs = math.nan
    variant_match = re.search(r"YOLOV?12([A-Z]+)", run_name, flags=re.IGNORECASE)
    epoch_match = re.search(r"(?:^|_)E(\d+)(?:_|$)", run_name, flags=re.IGNORECASE)
    if variant_match:
        variant = variant_match.group(1).upper()
    if epoch_match:
        planned_epochs = int(epoch_match.group(1))
    return {"variant": variant, "planned_epochs": planned_epochs}


def best_row_from_results(results: pd.DataFrame, column_map: Dict[str, Optional[str]]) -> pd.Series:
    sortable = results.copy()
    map_col = column_map.get("mAP50-95")
    map50_col = column_map.get("mAP50")
    epoch_col = column_map.get("epoch")
    sortable["_sort_map"] = pd.to_numeric(sortable[map_col], errors="coerce") if map_col else np.nan
    sortable["_sort_map50"] = pd.to_numeric(sortable[map50_col], errors="coerce") if map50_col else np.nan
    sortable["_sort_epoch"] = pd.to_numeric(sortable[epoch_col], errors="coerce") if epoch_col else np.inf
    sortable["_sort_map"] = sortable["_sort_map"].fillna(-np.inf)
    sortable["_sort_map50"] = sortable["_sort_map50"].fillna(-np.inf)
    sortable["_sort_epoch"] = sortable["_sort_epoch"].fillna(np.inf)
    sortable = sortable.sort_values(
        by=["_sort_map", "_sort_map50", "_sort_epoch"],
        ascending=[False, False, True],
        kind="mergesort",
    )
    return sortable.iloc[0]


def collect_run(results_csv: Path) -> Dict[str, object]:
    run_dir = results_csv.parent
    run_info = parse_run_name(run_dir.name)
    results = pd.read_csv(results_csv)
    results.columns = [column.strip() for column in results.columns]

    column_map = {
        "epoch": find_column(results.columns, ["epoch"]),
        "precision": find_column(results.columns, ["metrics/precision(B)", "precision"]),
        "recall": find_column(results.columns, ["metrics/recall(B)", "recall"]),
        "mAP50": find_column(results.columns, ["metrics/mAP50(B)", "mAP50"]),
        "mAP50-95": find_column(
            results.columns,
            ["metrics/mAP50-95(B)", "metrics/mAP50_95(B)", "mAP50-95", "mAP50_95"],
        ),
        "fitness": find_column(results.columns, ["fitness"]),
        "train_box_loss": find_column(results.columns, ["train/box_loss", "train_box_loss"]),
        "val_box_loss": find_column(results.columns, ["val/box_loss", "val_box_loss"]),
        "train_cls_loss": find_column(results.columns, ["train/cls_loss", "train_cls_loss"]),
        "val_cls_loss": find_column(results.columns, ["val/cls_loss", "val_cls_loss"]),
    }
    best = best_row_from_results(results, column_map)

    map50 = numeric_value(best, column_map["mAP50"])
    map5095 = numeric_value(best, column_map["mAP50-95"])
    fitness = numeric_value(best, column_map["fitness"])
    if math.isnan(fitness) and not math.isnan(map50) and not math.isnan(map5095):
        fitness = (0.1 * map50) + (0.9 * map5095)

    best_epoch_value = numeric_value(best, column_map["epoch"])
    best_epoch = int(best_epoch_value) if not math.isnan(best_epoch_value) else math.nan

    row: Dict[str, object] = {
        "run_name": run_dir.name,
        "variant": run_info["variant"],
        "planned_epochs": run_info["planned_epochs"],
        "best_epoch": best_epoch,
        "precision": numeric_value(best, column_map["precision"]),
        "recall": numeric_value(best, column_map["recall"]),
        "mAP50": map50,
        "mAP50-95": map5095,
        "fitness": fitness,
        "train_box_loss": numeric_value(best, column_map["train_box_loss"]),
        "val_box_loss": numeric_value(best, column_map["val_box_loss"]),
        "train_cls_loss": numeric_value(best, column_map["train_cls_loss"]),
        "val_cls_loss": numeric_value(best, column_map["val_cls_loss"]),
        "run_dir": str(run_dir),
    }
    return row


def collect_existing_runs(runs_dir: Path) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    if not runs_dir.exists():
        print(f"Warning: runs directory not found: {runs_dir}")
        return pd.DataFrame(columns=OUTPUT_COLUMNS)
    for run_dir in sorted(path for path in runs_dir.iterdir() if path.is_dir()):
        results_csv = run_dir / "results.csv"
        if not results_csv.exists():
            continue
        try:
            rows.append(collect_run(results_csv))
        except Exception as exc:  # pragma: no cover - keeps one bad run from stopping the summary.
            print(f"Warning: failed to read {results_csv}: {exc}")
    return pd.DataFrame(rows, columns=OUTPUT_COLUMNS)


def choose_best_model(summary: pd.DataFrame) -> pd.DataFrame:
    if summary.empty:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)
    sortable = summary.copy()
    sortable["_sort_map"] = pd.to_numeric(sortable["mAP50-95"], errors="coerce").fillna(-np.inf)
    sortable["_sort_map50"] = pd.to_numeric(sortable["mAP50"], errors="coerce").fillna(-np.inf)
    sortable["_sort_epochs"] = pd.to_numeric(sortable["planned_epochs"], errors="coerce").fillna(np.inf)
    sortable = sortable.sort_values(
        by=["_sort_map", "_sort_map50", "_sort_epochs"],
        ascending=[False, False, True],
        kind="mergesort",
    )
    return sortable.drop(columns=["_sort_map", "_sort_map50", "_sort_epochs"]).head(1)


def notes_frame() -> pd.DataFrame:
    return pd.DataFrame({"notes": NOTES})


def ensure_parent(path: Path) -> None:
    if path.parent != Path("."):
        path.parent.mkdir(parents=True, exist_ok=True)


def main() -> None:
    args = parse_args()
    runs_dir = Path(args.runs_dir)
    output_csv = Path(args.output_csv)
    output_excel = Path(args.output_excel)

    summary = collect_existing_runs(runs_dir)
    best_model = choose_best_model(summary)

    ensure_parent(output_csv)
    ensure_parent(output_excel)
    summary.to_csv(output_csv, index=False)
    with pd.ExcelWriter(output_excel, engine="openpyxl") as writer:
        summary.to_excel(writer, index=False, sheet_name="Summary_Existing_Runs")
        best_model.to_excel(writer, index=False, sheet_name="Best_Model")
        notes_frame().to_excel(writer, index=False, sheet_name="Notes")

    print(f"Saved training summary CSV to {output_csv}")
    print(f"Saved training summary workbook to {output_excel}")


if __name__ == "__main__":
    main()
