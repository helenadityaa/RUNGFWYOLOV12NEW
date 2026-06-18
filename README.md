# OpenSARShip YOLOv12 SAR Patch Detection

Repository ini berisi pipeline untuk membuat dataset YOLO dan training YOLOv12 object detection pada patch SAR GFW/OpenSARShip. Eksperimen final tetap berupa deteksi objek pada patch SAR, bukan klasifikasi dan bukan full-scene detection.

## Struktur Utama

```text
.
|-- labels.py
|-- requirements.txt
|-- new/
|   |-- Patch/
|   `-- metadata/
|       `-- metadata_with_vv_vh_gfw_ais_identity.csv
|-- yolo_exp/
|   `-- bbox_utils.py
|-- yolo_new_gen/
|   |-- prepare_data.py
|   |-- validate_dataset.py
|   |-- check_labels.py
|   |-- train_yolo.py
|   |-- train_yolo_det.py
|   |-- compare_experiments.py
|   |-- predict_yolo.py
|   `-- dataset_yolo_det_128_3class_vv_vh_rgb_scene/
`-- runs/detect/
```

Raw metadata dan raw TIFF tidak diubah oleh script. Filtering kelas dilakukan saat membuat dataset YOLO final.

## Kelas Final

Dataset final memakai 3 kelas target:

| ID | Kelas |
|---:|---|
| 0 | Fishing |
| 1 | Cargo |
| 2 | Passenger |

Kategori di luar tiga kelas target tidak dimasukkan ke dataset YOLO final.

## Input Citra

Input YOLO berupa PNG RGB gabungan dual-pol SAR:

```text
R = VV
G = VH
B = mean(VV, VH)
```

Kolom metadata patch:

- `patch_vv_actual_file`
- `patch_vh_actual_file`

## Dataset Final

Dataset YOLO final:

```text
yolo_new_gen/dataset_yolo_det_128_3class_vv_vh_rgb_scene
```

Split dataset memakai `scene` sebagai group agar satu scene tidak bocor ke lebih dari satu split. Seed default untuk split dan training adalah `42`.

`data.yaml` final harus berisi:

```yaml
nc: 3
names:
  0: Fishing
  1: Cargo
  2: Passenger
```

## Prepare Dataset

```powershell
python yolo_new_gen/prepare_data.py ^
  --metadata new/metadata/metadata_with_vv_vh_gfw_ais_identity.csv ^
  --image-dir new/Patch ^
  --output-dir yolo_new_gen/dataset_yolo_det_128_3class_vv_vh_rgb_scene ^
  --imgsz 128 ^
  --split-group scene ^
  --combine-vv-vh ^
  --seed 42
```

Setelah selesai, script menjalankan validasi dataset. Validasi manual dapat dijalankan dengan:

```powershell
python yolo_new_gen/validate_dataset.py
```

Untuk inspeksi visual bounding box ground truth:

```powershell
python yolo_new_gen/check_labels.py --seed 42
```

Output inspeksi disimpan ke:

```text
label_check/train
label_check/val
label_check/test
```

## Training Satu Run

```powershell
python yolo_new_gen/train_yolo.py ^
  --version 12 ^
  --variant n ^
  --task detect ^
  --data yolo_new_gen/dataset_yolo_det_128_3class_vv_vh_rgb_scene/data.yaml ^
  --epochs 50 ^
  --imgsz 128 ^
  --batch 16 ^
  --device auto ^
  --seed 42 ^
  --deterministic ^
  --lr0 0.001 ^
  --lrf 0.01 ^
  --cos-lr ^
  --weight-decay 0.0005 ^
  --mosaic 0.3 ^
  --mixup 0.0 ^
  --close-mosaic 10 ^
  --output runs/detect/YOLOV12N_128_E50_B16_3class_VV_VH_RGB_scene_seed42
```

## Perbandingan Eksperimen

Eksperimen final membandingkan semua kombinasi:

- Varian YOLOv12: `n`, `s`, `m`, `x`, `l`
- Epoch: `50`, `100`, `150`
- Total: 15 percobaan

Jalankan semua eksperimen:

```powershell
python yolo_new_gen/compare_experiments.py
```

Summary disimpan ke:

```text
summary_comparison_yolov12_3class_vv_vh_rgb_scene.csv
summary_comparison_yolov12_3class_vv_vh_rgb_scene.xlsx
```

Excel berisi sheet:

- `Summary_All_Experiments`
- `Best_Model`
- `Experiment_Config`
- `Notes`

Model terbaik dipilih dari evaluasi, berdasarkan mAP50-95 tertinggi. Jika mAP50-95 seri, dipilih mAP50 lebih tinggi, lalu epoch lebih kecil.

## Prediksi

Contoh prediksi pada folder test:

```powershell
python yolo_new_gen/predict_yolo.py ^
  --model runs/detect/YOLOV12N_128_E50_B16_3class_VV_VH_RGB_scene_seed42/weights/best.pt ^
  --source yolo_new_gen/dataset_yolo_det_128_3class_vv_vh_rgb_scene/test/images ^
  --conf 0.25 ^
  --device auto ^
  --output predictions/YOLOV12N_test_3class_vv_vh_rgb_scene
```

Selain gambar hasil prediksi, script juga menyimpan:

```text
predictions.csv
```

Kolom CSV minimal: `image_name`, `pred_class_id`, `pred_class_name`, `confidence`, `x1`, `y1`, `x2`, `y2`.

## Catatan

File besar, dataset lokal, weight model, dan output training tetap sebaiknya tidak dipush ke Git biasa. Source code dan dokumentasi cukup untuk mereproduksi dataset final dari metadata dan patch lokal yang valid.
