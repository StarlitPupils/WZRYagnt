# -*- coding: utf-8 -*-
"""UI 数值读取器（v0）：血条比例 + 技能冷却遮罩占比。

注意：这是 M1 增量组件，阈值需用真实对局画面标定；
本模块不依赖小地图模块，可独立运行。
"""
import cv2
import numpy as np


def _roi(frame, cx, cy, r):
    h, w = frame.shape[:2]
    x0, x1 = max(0, int(cx - r)), min(w, int(cx + r))
    y0, y1 = max(0, int(cy - r)), min(h, int(cy + r))
    if x1 <= x0 or y1 <= y0:
        return None
    return frame[y0:y1, x0:x1]


def find_hp_bar(frame, search=(0.08, 0.03, 0.35, 0.12), min_green_frac=0.05):
    """在画面顶部区域自动定位己方血条，返回 (x, y, w, h, hp_ratio) 或 None。

    王者荣耀血条：顶部靠左的绿色分段条。策略：
      1. 在搜索区域取绿色掩码（HSV H 40-85）；
      2. 按行投影找绿色最密集的水平带；
      3. 在该带内按列投影求绿色区间长度 -> hp_ratio = 绿色长度/条总长。
    """
    h, w = frame.shape[:2]
    x0, y0 = int(search[0] * w), int(search[1] * h)
    x1, y1 = int(search[2] * w), int(search[3] * h)
    roi = frame[y0:y1, x0:x1]
    if roi.size == 0:
        return None
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    green = cv2.inRange(hsv, (40, 80, 90), (85, 255, 255))
    frac = float((green > 0).mean())
    if frac < min_green_frac:
        return None
    # 行投影：绿色像素最多的行带
    row_sum = green.sum(axis=1)
    best_row = int(np.argmax(row_sum))
    band = green[max(0, best_row - 4):best_row + 5, :]
    col_present = (band > 0).any(axis=0)
    cols = np.nonzero(col_present)[0]
    if len(cols) == 0:
        return None
    bar_x0, bar_x1 = cols.min(), cols.max()
    bar_w = max(1, bar_x1 - bar_x0 + 1)
    # 血条总长按绿色覆盖的完整区间估算；hp 比例 = 绿色列占比（分段条近似）
    hp_ratio = float(col_present.sum()) / bar_w
    return (x0 + bar_x0, y0 + best_row, bar_w, 9, hp_ratio)


def skill_cooldown_fraction(frame, skill_points, roi_r=26):
    """技能冷却遮罩占比：对每个技能按钮 ROI 计算"变暗"比例。

    王者荣耀冷却时按钮被暗色扇形遮罩覆盖。v0 启发式：
      dark = 像素 V 值 < (ROI 最高 V 的 55%) 的比例；
      就绪技能图标明亮 -> dark 低；冷却 -> dark 高。
    返回 {skill_id: {"dark_frac": float, "mean_v": float}}。
    """
    out = {}
    for sid, (cx, cy) in skill_points.items():
        roi = _roi(frame, cx, cy, roi_r)
        if roi is None:
            out[sid] = {"dark_frac": None, "mean_v": None}
            continue
        v = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)[:, :, 2].astype(np.float32)
        vmax = v.max()
        if vmax < 30:
            out[sid] = {"dark_frac": 1.0, "mean_v": float(v.mean())}
            continue
        out[sid] = {"dark_frac": float((v < vmax * 0.55).mean()),
                    "mean_v": float(v.mean())}
    return out


def read_ui(frame, skill_points):
    """组合入口。skill_points: {1: (x, y), 2: ..., 3: ...}（像素坐标）。"""
    result = {}
    hp = find_hp_bar(frame)
    result["hp_bar"] = hp
    result["skills"] = skill_cooldown_fraction(frame, skill_points)
    return result
