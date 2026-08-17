# -*- coding: utf-8 -*-
"""校准文件加载：统一输出【归一化坐标】键值表。

支持两种现有格式：
  - configs/calibration.json             （已归一化：{"minimap_center": [nx, ny], ...}）
  - configs/calibration_absolute.json    （绝对像素：{"screen_width": W, "screen_height": H, "points": {...}}）
"""
import json
from pathlib import Path

DEFAULT_CALIB = Path("configs/calibration.json")
DEFAULT_CALIB_ABS = Path("configs/calibration_absolute.json")


def load_calibration(path=None):
    """返回 (calib_dict, source_path)。calib_dict 为归一化坐标点表，键见 configs/calibration.json。"""
    p = Path(path) if path else DEFAULT_CALIB
    if p.exists():
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
        # 归一化格式：直接是 {label: [nx, ny]}
        if "move_stick_center" in data:
            return data, p
        raise ValueError(f"校准文件 {p} 缺少 move_stick_center，格式不符")

    p_abs = DEFAULT_CALIB_ABS
    if p_abs.exists():
        with open(p_abs, "r", encoding="utf-8") as f:
            data = json.load(f)
        w, h = data["screen_width"], data["screen_height"]
        norm = {k: [x / w, y / h] for k, (x, y) in data["points"].items()}
        return norm, p_abs

    raise FileNotFoundError(
        "未找到校准文件，请先运行 scripts/control/calibrate.py 或 calibrate_absolute.py"
    )


def point_norm(calib, label, default=None):
    """取归一化点，缺失时返回 default。"""
    return calib.get(label, default)
