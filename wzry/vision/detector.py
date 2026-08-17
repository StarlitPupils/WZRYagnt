# -*- coding: utf-8 -*-
"""YOLO 检测封装：加载 ultralytics 模型，输出结构化检测结果。"""
import time
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np


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
        return dets
