import subprocess
import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
DATASET_YAML = SCRIPT_DIR / "dataset_yolo_det_128_3class_vv_vh_rgb_scene" / "data.yaml"
SUMMARY_CSV = PROJECT_ROOT / "summary_comparison_yolov12_3class_vv_vh_rgb_scene.csv"
SUMMARY_XLSX = PROJECT_ROOT / "summary_comparison_yolov12_3class_vv_vh_rgb_scene.xlsx"
VARIANTS = ["n", "s", "m", "x", "l"]
EPOCHS_LIST = [50, 100, 150]
SEED = 42
IMGSZ = 128
BATCH = 16
TASK = "detect"
SUMMARY_COLUMNS = [
    "variant",
    "epochs",
    "precision",
    "recall",
    "mAP50",
    "mAP50-95",
    "fitness",
    "best_epoch",
    "train_box_loss",
    "val_box_loss",
    "train_cls_loss",
    "val_cls_loss",
    "run_dir",
]


def experiment_name(variant, epochs):
    return f"YOLOV12{variant.upper()}_128_E{epochs}_B16_3class_VV_VH_RGB_scene_seed42"


def metric_value(row, names, default=None):
    for name in names:
        if name in row and pd.notna(row[name]):
            return row[name]
    return default


def empty_summary_row(variant, epochs, run_dir):
    return {
        "variant": variant,
        "epochs": epochs,
        "precision": None,
        "recall": None,
        "mAP50": None,
        "mAP50-95": None,
        "fitness": None,
        "best_epoch": None,
        "train_box_loss": None,
        "val_box_loss": None,
        "train_cls_loss": None,
        "val_cls_loss": None,
        "run_dir": str(run_dir),
    }


def best_metrics_from_results(results_csv, variant, epochs, run_dir):
    df_results = pd.read_csv(results_csv)
    df_results.columns = [column.strip() for column in df_results.columns]
    if df_results.empty:
        return empty_summary_row(variant, epochs, run_dir)

    fitness_column = "fitness" if "fitness" in df_results.columns else None
    map_column = "metrics/mAP50-95(B)" if "metrics/mAP50-95(B)" in df_results.columns else None
    selector_column = fitness_column or map_column
    if selector_column:
        best_idx = df_results[selector_column].astype(float).idxmax()
    else:
        best_idx = df_results.index[-1]

    best_row = df_results.loc[best_idx]
    best_epoch = metric_value(best_row, ["epoch"], default=int(best_idx) + 1)

    return {
        "variant": variant,
        "epochs": epochs,
        "precision": metric_value(best_row, ["metrics/precision(B)", "precision"], default=None),
        "recall": metric_value(best_row, ["metrics/recall(B)", "recall"], default=None),
        "mAP50": metric_value(best_row, ["metrics/mAP50(B)", "mAP50"], default=None),
        "mAP50-95": metric_value(best_row, ["metrics/mAP50-95(B)", "mAP50-95"], default=None),
        "fitness": metric_value(best_row, ["fitness"], default=None),
        "best_epoch": best_epoch,
        "train_box_loss": metric_value(best_row, ["train/box_loss"], default=None),
        "val_box_loss": metric_value(best_row, ["val/box_loss"], default=None),
        "train_cls_loss": metric_value(best_row, ["train/cls_loss"], default=None),
        "val_cls_loss": metric_value(best_row, ["val/cls_loss"], default=None),
        "run_dir": str(run_dir),
    }


def run_training(variant, epochs, run_dir):
    cmd = [
        sys.executable,
        str(SCRIPT_DIR / "train_yolo.py"),
        "--version",
        "12",
        "--variant",
        variant,
        "--task",
        TASK,
        "--data",
        str(DATASET_YAML),
        "--epochs",
        str(epochs),
        "--batch",
        str(BATCH),
        "--imgsz",
        str(IMGSZ),
        "--seed",
        str(SEED),
        "--deterministic",
        "--lr0",
        "0.001",
        "--lrf",
        "0.01",
        "--cos-lr",
        "--weight-decay",
        "0.0005",
        "--mosaic",
        "0.3",
        "--mixup",
        "0.0",
        "--close-mosaic",
        "10",
        "--workers",
        "8",
        "--output",
        str(run_dir),
    ]
    subprocess.run(cmd, check=True, cwd=PROJECT_ROOT)


def select_best_model(summary_df):
    metric_df = summary_df.dropna(subset=["mAP50-95"]).copy()
    if metric_df.empty:
        return pd.DataFrame(
            columns=[
                "best_variant",
                "best_epochs",
                "best_precision",
                "best_recall",
                "best_mAP50",
                "best_mAP50_95",
                "best_fitness",
                "best_epoch",
                "run_dir",
                "alasan_pemilihan",
            ]
        )

    metric_df = metric_df.sort_values(
        by=["mAP50-95", "mAP50", "epochs"],
        ascending=[False, False, True],
        kind="mergesort",
    )
    best = metric_df.iloc[0]
    return pd.DataFrame(
        [
            {
                "best_variant": best["variant"],
                "best_epochs": best["epochs"],
                "best_precision": best["precision"],
                "best_recall": best["recall"],
                "best_mAP50": best["mAP50"],
                "best_mAP50_95": best["mAP50-95"],
                "best_fitness": best["fitness"],
                "best_epoch": best["best_epoch"],
                "run_dir": best["run_dir"],
                "alasan_pemilihan": (
                    "Dipilih berdasarkan mAP50-95 tertinggi; jika seri memakai mAP50 lebih tinggi, "
                    "lalu epoch lebih kecil untuk efisiensi."
                ),
            }
        ]
    )


def export_excel(summary_df):
    best_model_df = select_best_model(summary_df)
    config_df = pd.DataFrame(
        [
            {
                "dataset_path": str(DATASET_YAML),
                "class_names": "Fishing, Cargo, Passenger",
                "image_mode": "vv_vh_rgb",
                "split_method": "scene",
                "seed": SEED,
                "imgsz": IMGSZ,
                "batch": BATCH,
                "variants": ", ".join(VARIANTS),
                "epochs_list": ", ".join(map(str, EPOCHS_LIST)),
                "total_experiments": len(VARIANTS) * len(EPOCHS_LIST),
            }
        ]
    )
    notes_df = pd.DataFrame(
        {
            "notes": [
                "Dataset menggunakan 3 kelas: Fishing, Cargo, Passenger.",
                "Input citra menggunakan RGB gabungan VV dan VH.",
                "Split dataset menggunakan scene-based split.",
                "Seed 42 digunakan untuk meningkatkan reproducibility.",
                "Model terbaik dipilih berdasarkan mAP50-95 tertinggi.",
            ]
        }
    )

    try:
        with pd.ExcelWriter(SUMMARY_XLSX, engine="openpyxl") as writer:
            summary_df.to_excel(writer, sheet_name="Summary_All_Experiments", index=False)
            best_model_df.to_excel(writer, sheet_name="Best_Model", index=False)
            config_df.to_excel(writer, sheet_name="Experiment_Config", index=False)
            notes_df.to_excel(writer, sheet_name="Notes", index=False)
    except ImportError:
        print("Please install openpyxl: pip install openpyxl")
        return

    print(f"Excel summary written to: {SUMMARY_XLSX}")


def run_compare():
    if not DATASET_YAML.exists():
        raise SystemExit(f"{DATASET_YAML} tidak ditemukan. Jalankan prepare_data.py final terlebih dahulu.")

    results = []
    for variant in VARIANTS:
        for epochs in EPOCHS_LIST:
            exp_name = experiment_name(variant, epochs)
            run_dir = PROJECT_ROOT / "runs" / "detect" / exp_name
            print(f"\n>>> Menjalankan eksperimen: YOLOv12{variant} epoch {epochs}")
            print(f"Run dir: {run_dir}")

            try:
                run_training(variant, epochs, run_dir)
            except subprocess.CalledProcessError as exc:
                print(f"WARNING: Training gagal untuk {exp_name}: {exc}")

            results_csv = run_dir / "results.csv"
            if results_csv.exists():
                results.append(best_metrics_from_results(results_csv, variant, epochs, run_dir))
                print(f"Selesai: {exp_name} tercatat dari best metric row.")
            else:
                print(f"WARNING: results.csv tidak ditemukan untuk {exp_name}")
                results.append(empty_summary_row(variant, epochs, run_dir))

    summary_df = pd.DataFrame(results, columns=SUMMARY_COLUMNS)
    summary_df.to_csv(SUMMARY_CSV, index=False)
    print("\n--- RINGKASAN HASIL ---")
    print(summary_df)
    print(f"\nCSV summary written to: {SUMMARY_CSV}")
    export_excel(summary_df)


if __name__ == "__main__":
    run_compare()
