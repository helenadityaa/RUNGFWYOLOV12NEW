from pathlib import Path
import sys

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from labels import map_ship_class

CLASS_BBOX_PAD = {
    0: 5.0,  # Fishing
    1: 2.0,  # Cargo
    2: 2.0,  # Passenger
}


def _finite_float(value, default=None):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return default
    if not np.isfinite(value):
        return default
    return value


def patch_scene_bounds(row):
    upper_left_x = _finite_float(row.get("UpperLeft_x"))
    upper_left_y = _finite_float(row.get("UpperLeft_y"))
    lower_right_x = _finite_float(row.get("LowerRight_x"))
    lower_right_y = _finite_float(row.get("LowerRight_y"))
    if None not in (upper_left_x, upper_left_y, lower_right_x, lower_right_y):
        min_x = min(upper_left_x, lower_right_x)
        max_x = max(upper_left_x, lower_right_x)
        min_y = min(upper_left_y, lower_right_y)
        max_y = max(upper_left_y, lower_right_y)
        return min_x, min_y, max_x, max_y

    center_x = _finite_float(row.get("Center_x"), 0.0)
    center_y = _finite_float(row.get("Center_y"), 0.0)
    half = 32.0
    return center_x - half, center_y - half, center_x + half, center_y + half


def patch_scene_size(row):
    min_x, min_y, max_x, max_y = patch_scene_bounds(row)
    return (max_x - min_x), (max_y - min_y)


def head_tail_local(row):
    min_x, min_y, _, _ = patch_scene_bounds(row)
    center_x = _finite_float(row.get("Center_x"), min_x)
    center_y = _finite_float(row.get("Center_y"), min_y)
    head_x = _finite_float(row.get("Head_x"), center_x) - min_x
    head_y = _finite_float(row.get("Head_y"), center_y) - min_y
    tail_x = _finite_float(row.get("Tail_x"), center_x) - min_x
    tail_y = _finite_float(row.get("Tail_y"), center_y) - min_y
    return np.array([head_x, head_y], dtype=np.float32), np.array([tail_x, tail_y], dtype=np.float32)


def _safe_ratio(width_value, length_value):
    try:
        width_value = float(width_value)
        length_value = float(length_value)
    except (TypeError, ValueError):
        return None
    if width_value <= 0 or length_value <= 0:
        return None
    return width_value / length_value


def estimate_ship_width_pixels(row, head_xy, tail_xy, default_ratio=0.18, min_width=3.0):
    pixel_length = float(np.hypot(*(tail_xy - head_xy)))
    ratios = [
        _safe_ratio(row.get("AIS_Width"), row.get("AIS_Length")),
        _safe_ratio(row.get("Breadth_extreme"), row.get("Length_overall")),
    ]
    ratio = next((r for r in ratios if r is not None), default_ratio)
    return max(min_width, pixel_length * ratio)


def oriented_box_corners_local(row):
    head_xy, tail_xy = head_tail_local(row)
    vector = tail_xy - head_xy
    norm = float(np.hypot(*vector))
    if norm < 1.0:
        center_x = float(row["Center_x"])
        center_y = float(row["Center_y"])
        min_x, min_y, _, _ = patch_scene_bounds(row)
        cx = center_x - min_x
        cy = center_y - min_y
        half = 6.0
        return np.array(
            [
                [cx - half, cy - half],
                [cx + half, cy - half],
                [cx + half, cy + half],
                [cx - half, cy + half],
            ],
            dtype=np.float32,
        )

    unit = vector / norm
    perp = np.array([-unit[1], unit[0]], dtype=np.float32)
    half_width = estimate_ship_width_pixels(row, head_xy, tail_xy) / 2.0
    corners = np.array(
        [
            head_xy + perp * half_width,
            head_xy - perp * half_width,
            tail_xy - perp * half_width,
            tail_xy + perp * half_width,
        ],
        dtype=np.float32,
    )
    return corners


def axis_aligned_bbox_local(row, image_shape=None, pad=None):
    if pad is None:
        class_id = map_ship_class(row)
        pad = CLASS_BBOX_PAD.get(class_id, 2.0)
    corners = oriented_box_corners_local(row)
    x1 = float(corners[:, 0].min() - pad)
    y1 = float(corners[:, 1].min() - pad)
    x2 = float(corners[:, 0].max() + pad)
    y2 = float(corners[:, 1].max() + pad)

    scene_w, scene_h = patch_scene_size(row)
    if image_shape is not None:
        img_h, img_w = image_shape[:2]
        sx = float(img_w) / float(scene_w)
        sy = float(img_h) / float(scene_h)
        x1 *= sx
        x2 *= sx
        y1 *= sy
        y2 *= sy
        x1 = np.clip(x1, 0.0, float(img_w))
        x2 = np.clip(x2, 0.0, float(img_w))
        y1 = np.clip(y1, 0.0, float(img_h))
        y2 = np.clip(y2, 0.0, float(img_h))

    return x1, y1, x2, y2


def bbox_to_yolo(bbox_xyxy, image_shape):
    img_h, img_w = image_shape[:2]
    if img_h <= 0 or img_w <= 0:
        raise ValueError(f"Invalid image shape for YOLO bbox conversion: {image_shape}")

    x1, y1, x2, y2 = [float(value) for value in bbox_xyxy]
    x1, x2 = sorted((x1, x2))
    y1, y2 = sorted((y1, y2))
    x1 = float(np.clip(x1, 0.0, float(img_w)))
    x2 = float(np.clip(x2, 0.0, float(img_w)))
    y1 = float(np.clip(y1, 0.0, float(img_h)))
    y2 = float(np.clip(y2, 0.0, float(img_h)))

    width = max(1e-6, x2 - x1)
    height = max(1e-6, y2 - y1)
    center_x = x1 + width / 2.0
    center_y = y1 + height / 2.0
    return (
        float(np.clip(center_x / img_w, 0.0, 1.0)),
        float(np.clip(center_y / img_h, 0.0, 1.0)),
        float(np.clip(width / img_w, 0.0, 1.0)),
        float(np.clip(height / img_h, 0.0, 1.0)),
    )


def sar_to_rgb_uint8(img):
    img = np.asarray(img)
    if img.ndim == 3 and img.shape[0] in (1, 2, 3, 4):
        img = np.transpose(img, (1, 2, 0))
    if img.ndim == 2:
        img = np.stack([img, img, img], axis=-1)
    if img.ndim != 3:
        raise ValueError(f"Unsupported image shape: {img.shape}")
    if img.shape[-1] == 1:
        img = np.repeat(img, 3, axis=-1)
    elif img.shape[-1] == 2:
        img = np.concatenate([img, np.zeros_like(img[..., :1])], axis=-1)
    elif img.shape[-1] > 3:
        img = img[..., :3]

    img = img.astype(np.float32)
    lo, hi = np.percentile(img, [1, 99])
    if hi <= lo:
        hi = lo + 1e-6
    img = np.clip((img - lo) / (hi - lo), 0.0, 1.0)
    return (img * 255.0).astype(np.uint8)


def sar_channel_to_uint8(img):
    img = np.asarray(img)
    if img.ndim == 3 and img.shape[0] in (1, 2, 3, 4):
        img = np.transpose(img, (1, 2, 0))
    if img.ndim == 3:
        img = img[..., 0]
    if img.ndim != 2:
        raise ValueError(f"Unsupported SAR channel shape: {img.shape}")

    img = img.astype(np.float32)
    lo, hi = np.percentile(img, [1, 99])
    if hi <= lo:
        hi = lo + 1e-6
    img = np.clip((img - lo) / (hi - lo), 0.0, 1.0)
    return (img * 255.0).astype(np.uint8)


def dual_pol_to_rgb_uint8(vv_img, vh_img):
    vv = sar_channel_to_uint8(vv_img)
    vh = sar_channel_to_uint8(vh_img)
    if vv.shape != vh.shape:
        raise ValueError(f"VV/VH shape mismatch: {vv.shape} vs {vh.shape}")

    mixed = ((vv.astype(np.uint16) + vh.astype(np.uint16)) // 2).astype(np.uint8)
    return np.stack([vv, vh, mixed], axis=-1)


def patch_name_from_any(value):
    return Path(str(value)).name
