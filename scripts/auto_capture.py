# -*- coding: utf-8 -*-
"""一边打一边学: 自动采集 + 自我标注 (v11.7 YoloDetector 兼容)。

det 可能传入: YOLO 模型 (有 .predict) 或 YoloDetector (wrapper 有 .model)
统一: _pred = det.model if hasattr(det,'model') else det; _predfn = _pred.predict
"""
import json
import threading
import time
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "auto_labeled"
MM = (0, 0, 232, 232)
_lock = threading.Lock()
_last_mm = {}
_last_scr = {}
CLS = {"mm_red": 12, "mm_blue": 13, "mm_green": 14,
       "enemy_hero": 0, "ally_hero": 1, "self": 11, "neutral_monster": 8}


def clamp(x, d):
    return round(x, d)


def _vote(store, key, cls, conf):
    with _lock:
        store[key] = cls
        return True


def capture_once(frame, det):
    """自标: 小地图(mm点) + 全屏(英雄) -> auto_labeled。兼容 YOLO / YoloDetector。"""
    h, w = frame.shape[:2]
    mm = frame[MM[1]:MM[3], MM[0]:MM[2]]
    stamp = f"{int(time.time()*1000)}"
    lines = []
    # 兼容 wrapper: YoloDetector.model = YOLO
    try:
        _pred = det.model if hasattr(det, "model") else det
        _predfn = getattr(_pred, "predict", None)
        if _predfn is None:
            return False
    except Exception:
        return False
    # ① 小地图内
    try:
        r = _predfn(mm, conf=0.5, verbose=False)[0]
        nms = r.names
        for b in r.boxes:
            nm = nms[int(b.cls[0])]
            if nm in ("mm_red", "mm_blue", "mm_green"):
                x1, y1, x2, y2 = [int(v) for v in b.xyxy[0].tolist()]
                cx, cy = (x1 + x2) / 2 / 232, (y1 + y2) / 2 / 232
                bw, bh = (x2 - x1) / 232, (y2 - y1) / 232
                key = f"{clamp(cx,3)}|{clamp(cy,3)}"
                if _vote(_last_mm, key, nm, float(b.conf[0])):
                    lines.append(f"{CLS[nm]} {cx:.4f} {cy:.4f} {bw:.4f} {bh:.4f}")
    except Exception:
        pass
    # ② 全屏
    try:
        dets = _predfn(frame, conf=0.7, verbose=False)[0]
        nms = dets.names
        for b in dets.boxes:
            nm = nms[int(b.cls[0])]
            if nm in ("enemy_hero", "ally_hero", "self", "neutral_monster"):
                x1, y1, x2, y2 = [int(v) for v in b.xyxy[0].tolist()]
                cx, cy = (x1 + x2) / 2 / w, (y1 + y2) / 2 / h
                bw, bh = (x2 - x1) / w, (y2 - y1) / h
                key = f"{clamp(cx,2)}|{clamp(cy,2)}|{nm}"
                if _vote(_last_scr, key, nm, float(b.conf[0])):
                    lines.append(f"{CLS[nm]} {cx:.4f} {cy:.4f} {bw:.4f} {bh:.4f}")
    except Exception:
        pass
    if lines:
        OUT.mkdir(parents=True, exist_ok=True)
        img_p = OUT / f"auto_{stamp}.png"
        cv2.imwrite(str(img_p), frame)
        (OUT / f"auto_{stamp}.txt").write_text("\n".join(lines), encoding="utf-8")
        return True
    return False
