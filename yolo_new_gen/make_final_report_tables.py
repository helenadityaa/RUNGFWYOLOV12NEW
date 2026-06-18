"""Combine post-processing outputs into one Excel workbook for reporting."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Tuple

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create final YOLOv12 report tables from existing post-processing outputs."
    )
    parser.add_argument(
        "--training-summary",
        default="summary_existing_runs.xlsx",
        help="Workbook created by collect_existing_results.py.",
    )
    parser.add_argument(
        "--prediction-summary",
        default="summary_existing_predictions.xlsx",
        help="Workbook created by collect_existing_predictions.py.",
    )
    parser.add_argument(
        "--characteristics-summary",
        default="analysis_outputs/ship_characteristics_summary.xlsx",
        help="Workbook created by analyze_ship_characteristics.py.",
    )
    parser.add_argument(
        "--output-excel",
        default="final_yolov12_report_tables.xlsx",
        help="Final report workbook path.",
    )
    return parser.parse_args()


def read_sheet_or_empty(
    workbook_path: Path,
    sheet_name: str,
    output_sheet_name: str,
    warnings: List[str],
) -> pd.DataFrame:
    if not workbook_path.exists():
        warnings.append(f"Missing input workbook for {output_sheet_name}: {workbook_path}")
        return pd.DataFrame({"warning": [f"Input not found: {workbook_path}"]})
    try:
        return pd.read_excel(workbook_path, sheet_name=sheet_name)
    except ValueError:
        warnings.append(f"Missing sheet {sheet_name} in {workbook_path}")
        return pd.DataFrame({"warning": [f"Sheet not found: {sheet_name} in {workbook_path}"]})
    except Exception as exc:  # pragma: no cover - report tables should still be created.
        warnings.append(f"Could not read {sheet_name} from {workbook_path}: {exc}")
        return pd.DataFrame({"warning": [f"Could not read {workbook_path}: {exc}"]})


def notes_frame(warnings: List[str]) -> pd.DataFrame:
    notes = [
        "Tabel final ini menggabungkan output post-processing yang sudah ada.",
        "Script ini tidak menjalankan training ulang.",
        "Script ini tidak menjalankan predict ulang.",
        "Dataset final menggunakan 3 kelas: Fishing, Cargo, Passenger.",
        "Input citra menggunakan RGB gabungan VV dan VH.",
    ]
    notes.extend(warnings)
    return pd.DataFrame({"notes": notes})


def ensure_parent(path: Path) -> None:
    if path.parent != Path("."):
        path.parent.mkdir(parents=True, exist_ok=True)


def main() -> None:
    args = parse_args()
    training_summary_path = Path(args.training_summary)
    prediction_summary_path = Path(args.prediction_summary)
    characteristics_summary_path = Path(args.characteristics_summary)
    output_excel = Path(args.output_excel)

    warnings: List[str] = []
    sheets: List[Tuple[str, pd.DataFrame]] = [
        (
            "Training_Summary",
            read_sheet_or_empty(
                training_summary_path,
                "Summary_Existing_Runs",
                "Training_Summary",
                warnings,
            ),
        ),
        (
            "Best_Model",
            read_sheet_or_empty(training_summary_path, "Best_Model", "Best_Model", warnings),
        ),
        (
            "Prediction_Summary",
            read_sheet_or_empty(
                prediction_summary_path,
                "Prediction_Summary",
                "Prediction_Summary",
                warnings,
            ),
        ),
        (
            "Ship_Characteristics",
            read_sheet_or_empty(
                characteristics_summary_path,
                "Summary_By_Class",
                "Ship_Characteristics",
                warnings,
            ),
        ),
    ]

    ensure_parent(output_excel)
    with pd.ExcelWriter(output_excel, engine="openpyxl") as writer:
        for sheet_name, frame in sheets:
            frame.to_excel(writer, index=False, sheet_name=sheet_name)
        notes_frame(warnings).to_excel(writer, index=False, sheet_name="Notes")

    for warning in warnings:
        print(f"Warning: {warning}")
    print(f"Saved final report workbook to {output_excel}")


if __name__ == "__main__":
    main()
