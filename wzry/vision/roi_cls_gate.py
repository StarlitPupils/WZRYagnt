# -*- coding: utf-8 -*-
"""接入 ROI 分类器 (v4.1): 小地图红/蓝/绿候选经分类器重判类目。

在 MMTrackerV7.update() 的 heroes 输出后:
  - 每个 enemy 候选 (归一化点) 抠 32x32 ROI -> clf 预测:
      mm_red(>=可信) -> 保留 enemy
      其他(blue/green/噪声) -> 剔除(假红点纠正)
  - ally 候选类似: mm_blue -> ally; 其他剔除
  - self 候选: mm_green -> self; 其他剔除
分类器为轻量 yolo8n-cls (0.9ms), 6Hz 小地图 * <=5 候选 = 实时无忧。
"""
import json
import threading
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
_lock = threading.Lock()
_clf = None
_loaded_path = None
_CONF = {"mm_red": 0.55, "mm_blue": 0.55, "mm_green": 0.55}


def _load_clf(path=None):
    global _clf, _loaded_path
    path = path or str(ROOT / "runs" / "cls" / "roi_cls" / "weights" / "best.pt")
    with _lock:
        if _clf is None or _loaded_path != path:
            from ultralytics import YOLO
            _clf = YOLO(path)
            _loaded_path = path
        return _clf


def classify_roi(img_bgr):
    """对抠图 ROI 分类 -> (cls_name, conf)。"""
    try:
        clf = _load_clf()
        r = clf.predict(img_bgr, verbose=False)[0]
        return clf.names[int(r.probs.top1)], float(r.probs.top1conf)
    except Exception:
        return None, 0.0


def filter_dots(mm_img, reds, blues, greens, size=242):
    """对 (px,py) 归一候选列表分类过滤。输入输出均为 (x,y,conf) 归一化 0-1。

    返回 (keep_red, keep_blue, keep_green) 各为过滤后的候选。
    """
    mm = mm_img[0:size, 0:size] if mm_img is not None and mm_img.shape[0] >= size else mm_img
    # 小图无效 -> 原样返回
    try:
        if mm is None or mm.size == 0:
            return reds, blues, greens
        h, w = mm.shape[:2]
    except Exception:
        return reds, blues, greens

    def _classify(cands):
        keep = []
        for (nx, ny, conf) in cands:
            px = int(nx * w)
            py = int(ny * h)
            r_half = 17
            x0, y0 = max(0, px - r_half), max(0, py - r_half)
            x1, y1 = min(w, px + r_half), min(h, py + r_half)
            roi = mm[y0:y1, x0:x1]
            if roi.size == 0:
                continue
            roi = cv2.resize(roi, (64, 64), interpolation=cv2.INTER_AREA)
            name, c = classify_roi(roi)
            if name is None:
                keep.append((nx, ny, conf))   # 分类器故障不拦截
            else:
                keep.append((nx, ny, conf, name, c))
        return keep

    kr = _classify(reds)
    kb = _classify(blues)
    kg = _classify(greens)
    # 按类名保留
    red_out = [(x, y, c) for (x, y, c, n, cc) in kr if n == "mm_red" and cc >= _CONF["mm_red"]]
    blue_out = [(x, y, c) for (x, y, c, n, cc) in kb if n == "mm_blue" and cc >= _CONF["mm_blue"]]
    green_out = [(x, y, c) for (x, y, c, n, cc) in kg if n == "mm_green" and cc >= _CONF["mm_green"]]
    return red_out, blue_out, green_out
