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


def find_hp_bar(frame, search=(0.08, 0.04, 0.40, 0.15), min_bar_len=40):
    """在画面顶部区域自动定位己方血条，返回 (x, y, w, h, hp_ratio) 或 None。

    王者荣耀血条：左上角英雄头像右侧的水平绿色长条（绿色偏暗，阈值需放宽）。
    策略（水平条结构检测，抗场景绿色干扰）：
      1. 搜索区绿色掩码（HSV H 35-90, S≥60, V≥70）；
      2. 水平核（31x3）开运算滤掉块状场景绿，只留长横条；
      3. 逐行找最长连续绿色段，作为血条；
      4. hp_ratio = 绿色段长度 / (绿色段+右侧空槽延伸)。
    """
    h, w = frame.shape[:2]
    x0, y0 = int(search[0] * w), int(search[1] * h)
    x1, y1 = int(search[2] * w), int(search[3] * h)
    roi = frame[y0:y1, x0:x1]
    if roi.size == 0:
        return None
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    green = cv2.inRange(hsv, (35, 60, 70), (90, 255, 255))
    # 水平条结构：31x3 核开运算（保留 >31px 的横条）
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (31, 3))
    bars = cv2.morphologyEx(green, cv2.MORPH_OPEN, kernel)
    # 逐行最长连续段
    best = None
    for ry in range(bars.shape[0]):
        row = bars[ry]
        # 连续段：(起点, 长度)
        segs = []
        in_run = False
        for x in range(row.shape[0]):
            if row[x] and not in_run:
                start = x
                in_run = True
            elif not row[x] and in_run:
                segs.append((start, x - start))
                in_run = False
        if in_run:
            segs.append((start, row.shape[0] - start))
        for (sx, slen) in segs:
            if slen >= min_bar_len and (best is None or slen > best[0]):
                best = (slen, ry, sx)
    if best is None:
        return None
    slen, ry, sx = best
    # 血条总长：绿色段 + 右侧空血槽延伸（灰度 >60 的连续区）
    roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    bar_end = sx + slen
    while bar_end < roi_gray.shape[1] and roi_gray[ry, bar_end] > 60:
        bar_end += 1
    bar_total = max(slen, bar_end - sx)
    hp_ratio = float(slen) / bar_total
    return (x0 + sx, y0 + ry, bar_total, 3, round(float(hp_ratio), 3))


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
