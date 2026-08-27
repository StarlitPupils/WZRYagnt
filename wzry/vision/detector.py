# -*- coding: utf-8 -*-
"""YOLO 检测封装：加载 ultralytics 模型，输出结构化检测结果。

阵营修正：小兵/塔两阵营外观几乎相同，YOLO 常混淆阵营。
但 我方=蓝方(蓝色饰件/血条) / 敌方=红方(红色饰件/血条)，框内蓝/红像素
占比是硬判据 → 阵营翻转修正。
"""
import time
from dataclasses import dataclass, field
from typing import List, Optional

import cv2
import numpy as np

CAMP_PAIRS = {"enemy_minion": "ally_minion", "ally_minion": "enemy_minion",
              "enemy_turret": "ally_turret", "ally_turret": "enemy_turret",
              "enemy_crystal": "ally_crystal", "ally_crystal": "enemy_crystal"}
# 注: 英雄类(0/1)不参与颜色修正——英雄框内蓝饰件/技能特效多，会误翻；
#     小兵/塔/水晶两阵营外观相同，蓝/红饰件占比是硬判据
# 类索引: 0 enemy_hero 1 ally_hero 2 enemy_minion 3 ally_minion
#         4 enemy_turret 5 ally_turret 6 enemy_crystal 7 ally_crystal (偶数=敌)
PAIR_ORDER = ["enemy_minion", "ally_minion", "enemy_turret", "ally_turret",
              "enemy_crystal", "ally_crystal"]
CAMP_EPS = 0.02


def camp_blue_red_fracs(frame, x1, y1, x2, y2):
    """框内蓝色/红色像素占比（BGR 帧）。"""
    h, w = frame.shape[:2]
    x1, y1 = max(0, int(x1)), max(0, int(y1))
    x2, y2 = min(w, int(x2)), min(h, int(y2))
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return 0.0, 0.0
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    H, S = hsv[..., 0].astype(int), hsv[..., 1].astype(int)
    blue = (((H >= 70) & (H <= 135) & (S > 60))).mean()
    red = ((((H <= 15) | (H >= 165)) & (S > 60))).mean()
    return float(blue), float(red)


def camp_correct(frame, dets, eps=CAMP_EPS):
    """按框内蓝/红占比修正阵营类（偶数类=敌, 奇数类=我）。

    小兵: 若某色占比>0.40 视为背景(河道/崖壁)污染 → 保持模型原判。
    """
    out = []
    for d in dets:
        if d.cls in CAMP_PAIRS:
            idx = PAIR_ORDER.index(d.cls)
            blue, red = camp_blue_red_fracs(frame, *d.xyxy)
            if not d.cls.endswith("minion"):
                if (blue - red) > eps and idx % 2 == 0:
                    d.cls = CAMP_PAIRS[d.cls]
                elif (red - blue) > eps and idx % 2 == 1:
                    d.cls = CAMP_PAIRS[d.cls]
            else:
                # 小兵: 背景污染保护
                if (blue - red) > eps and idx % 2 == 0 and blue <= 0.40:
                    d.cls = CAMP_PAIRS[d.cls]
                elif (red - blue) > eps and idx % 2 == 1 and red <= 0.40:
                    d.cls = CAMP_PAIRS[d.cls]
        out.append(d)
    return out


@dataclass
class Det:
    cls: str
    conf: float
    xyxy: List[float]          # [x1, y1, x2, y2] 像素
    center: List[float]        # [cx, cy]

    def to_dict(self):
        return {"class": self.cls, "confidence": round(float(self.conf), 4),
                "bbox": [round(v, 1) for v in self.xyxy],
                "center": [round(v, 1) for v in self.center]}


class YoloDetector:
    def __init__(self, model_path, conf=0.3, iou=0.5, device=None, half=False):
        from ultralytics import YOLO
        self.model = YOLO(str(model_path))
        self.conf = conf
        self.iou = iou
        self.device = device or ("0" if self._cuda_ok() else "cpu")
        self.half = half
        self.last_infer_ms = 0.0

    @staticmethod
    def _cuda_ok():
        try:
            import torch
            return torch.cuda.is_available()
        except Exception:
            return False

    def detect(self, frame: np.ndarray) -> List[Det]:
        t0 = time.perf_counter()
        results = self.model.predict(frame, conf=self.conf, iou=self.iou,
                                     device=self.device, verbose=False,
                                     half=self.half)[0]
        self.last_infer_ms = (time.perf_counter() - t0) * 1000.0
        dets: List[Det] = []
        if results.boxes is None:
            return dets
        boxes = results.boxes.xyxy.cpu().numpy()
        confs = results.boxes.conf.cpu().numpy()
        cls_ids = results.boxes.cls.cpu().numpy().astype(int)
        names = results.names
        for box, c, cid in zip(boxes, confs, cls_ids):
            x1, y1, x2, y2 = (float(v) for v in box)
            dets.append(Det(
                cls=names[cid], conf=float(c),
                xyxy=[x1, y1, x2, y2],
                center=[(x1 + x2) / 2, (y1 + y2) / 2],
            ))
        return hero_dedup_ui(camp_correct(frame, dets))


def _iou(a, b):
    ax1, ay1, ax2, ay2 = a; bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    inter = (ix2 - ix1) * (iy2 - iy1)
    return inter / max(1e-6, (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter)


def hero_dedup_ui(dets):
    """英雄类后处理：
    1) 跨类去重: 我英/敌英同框(同一英雄双阵营标签) → 保留高置信
    2) 同类合并(IoU>0.5)
    3) UI 死区: 右缘/底部 UI 圈(攻击键/回城等)误检英雄 → 删除
    """
    dets = list(dets)
    # 1) 跨类去重
    heros = [(i, d) for i, d in enumerate(dets) if d.cls in ("enemy_hero", "ally_hero")]
    for a in range(len(heros)):
        for b in range(a + 1, len(heros)):
            (ia, da), (ib, db) = heros[a], heros[b]
            if da.cls == db.cls:
                continue
            if _iou(da.xyxy, db.xyxy) > 0.45:
                drop = db if da.conf >= db.conf else da   # 删低置信
                if drop in dets:
                    dets.remove(drop)
    # 3) UI 死区
    dead = []
    for d in dets:
        if d.cls in ("enemy_hero", "ally_hero"):
            x1, y1, x2, y2 = d.xyxy
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            h = y2 - y1
            if ((cx > 1105 and cy > 380) or (cy > 570) or (cx > 1105 and h > 200)
                    or (cx < 205 and 225 < cy < 475)):   # 左缘 UI(商店/金币)
                dead.append(d)
    for d in dead:
        if d in dets:
            dets.remove(d)
    # 2) 同类合并
    for cls in ("enemy_hero", "ally_hero"):
        pool = [d for d in dets if d.cls == cls]
        kept = []
        for d in sorted(pool, key=lambda x: -x.conf):
            if not any(_iou(d.xyxy, k.xyxy) > 0.5 for k in kept):
                kept.append(d)
        dets = [d for d in dets if d.cls != cls] + kept
    return dets
