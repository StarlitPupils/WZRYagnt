# -*- coding: utf-8 -*-
"""自己英雄 HP/MP 检测（英雄头顶血条蓝条）。

原理（真机验证）：
  - 王者荣耀镜头跟随自己英雄，自己始终在画面中央偏下
  - 中央区域（x500-800, y300-550）高饱和大块 = 自己英雄
  - 英雄头顶上方：绿色血条（HP）+ 蓝色量条（MP）
  - 多帧验证：绿条 37-160px、蓝条 29-132px 稳定检出

用法：
    from wzry.vision.self_bars import detect_self_bars
    hp, mp = detect_self_bars(frame)   # (ratio, ratio) 或 (None, None)
"""
from __future__ import annotations

import cv2
import numpy as np

# 自己英雄搜索区（中央偏下）
HERO_REGION = (500, 300, 800, 550)
MIN_HERO_AREA = 300
# 头顶血条/蓝条搜索（英雄中心上方）
HEAD_OFFSET_Y = (100, 160)   # 血条在英雄中心上方 100-160px
HEAD_X_HALF = 100


def _find_self_hero(frame):
    """中央区域找自己英雄（高饱和大块），返回 (cx, cy) 或 None。"""
    x0, y0, x1, y1 = HERO_REGION
    roi = frame[y0:y1, x0:x1]
    if roi.size == 0:
        return None
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    sat = (hsv[..., 1] > 80).astype(np.uint8)
    n, lab, st, cent = cv2.connectedComponentsWithStats(sat, 8)
    best = None
    for i in range(1, n):
        area = st[i, cv2.CC_STAT_AREA]
        if area >= MIN_HERO_AREA and (best is None or area > best[0]):
            best = (area, int(cent[i][0]) + x0, int(cent[i][1]) + y0)
    return (best[1], best[2]) if best else None


def _find_bar(head_roi, hue_cond):
    """在头顶区域找指定色相的横条，返回 (ratio, 条长, 区域宽)。

    ratio = 条长 / 区域宽（近似血量/蓝量比例）
    """
    if head_roi is None or head_roi.size == 0:
        return None, 0, 0
    hsv = cv2.cvtColor(head_roi, cv2.COLOR_BGR2HSV)
    H, S, V = hsv[..., 0].astype(int), hsv[..., 1].astype(int), hsv[..., 2].astype(int)
    mask = hue_cond(H, S, V)
    # 逐行找最长连续段，取所有行最大值
    best_len = 0
    best_ratio = 0.0
    for ry in range(mask.shape[0]):
        row = mask[ry]
        run = 0
        for x in range(row.shape[0]):
            if row[x]:
                run += 1
                best_len = max(best_len, run)
            else:
                run = 0
        if run > 0:
            best_len = max(best_len, run)
    if best_len > 0:
        best_ratio = best_len / max(1, mask.shape[1])
    return best_ratio, best_len, mask.shape[1]


def detect_self_bars(frame):
    """检测自己英雄 HP/MP（头顶绿条/蓝条）。

    返回 (hp_ratio, mp_ratio, hero_pos) 或 (None, None, None)。
    """
    hero = _find_self_hero(frame)
    if hero is None:
        return None, None, None
    cx, cy = hero
    h, w = frame.shape[:2]
    # 头顶区域：英雄中心上方 100-160px，横向 ±100px
    y0 = max(0, cy - HEAD_OFFSET_Y[1])
    y1 = max(0, cy - HEAD_OFFSET_Y[0])
    x0 = max(0, cx - HEAD_X_HALF)
    x1 = min(w, cx + HEAD_X_HALF)
    if y1 <= y0:
        return None, None, hero
    head = frame[y0:y1, x0:x1]

    def is_green(H, S, V):
        return (H >= 35) & (H <= 90) & (S > 60) & (V > 60)

    def is_blue(H, S, V):
        return (H >= 90) & (H <= 135) & (S > 60) & (V > 60)

    hp, hp_len, _ = _find_bar(head, is_green)
    mp, mp_len, _ = _find_bar(head, is_blue)
    # 阈值：条长至少 25px 才算有效（防场景色）
    if hp_len < 25:
        hp = None
    if mp_len < 25:
        mp = None
    return hp, mp, hero


if __name__ == "__main__":
    import numpy as np
    for f in ["s01.png", "s05.png", "s08.png", "s11.png"]:
        data = np.fromfile(rf"E:\WZRYagent\temp\ann\{f}", dtype=np.uint8)
        img = cv2.imdecode(data, cv2.IMREAD_COLOR)
        if img is None:
            print(f"{f}: 无法读取")
            continue
        hp, mp, pos = detect_self_bars(img)
        print(f"{f}: HP={hp} MP={mp} 英雄位置={pos}")
