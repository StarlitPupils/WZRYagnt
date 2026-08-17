# -*- coding: utf-8 -*-
"""阵营开局判断：对局确认后第一帧，检测己方泉水颜色（英雄出生在泉水中）。

蓝方泉水偏蓝、红方泉水偏红：取屏幕中心 ROI 平均色 B/R 比值判定。
返回 'blue' / 'red' / None（无法判定）。

发育路方向：蓝方在右下（小地图 ~(0.72, 0.82)），红方镜像左上（~(0.28, 0.18)）。
"""
import cv2
import numpy as np


def detect_camp_from_center(frame, roi_frac=0.25):
    """屏幕中心 ROI 主色调 -> 'blue' / 'red' / None。"""
    h, w = frame.shape[:2]
    rw, rh = int(w * roi_frac), int(h * roi_frac)
    cx, cy = w // 2, h // 2
    roi = frame[cy - rh // 2:cy + rh // 2, cx - rw // 2:cx + rw // 2]
    if roi.size == 0:
        return None
    b, g, r = roi.reshape(-1, 3).mean(axis=0)
    if b > r * 1.08:
        return "blue"
    if r > b * 1.08:
        return "red"
    return None


def lane_dir_for_camp(camp: str):
    """阵营 -> 发育路方向（小地图归一化坐标）。蓝方右下，红方左上镜像。"""
    return (0.72, 0.82) if camp == "blue" else (0.28, 0.18)
