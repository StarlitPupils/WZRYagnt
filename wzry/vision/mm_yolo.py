# -*- coding: utf-8 -*-
"""小地图元素 YOLO 推理（v4.0 强化训练版理解层）。

用户提供抠图 -> 合成数据集 -> YOLOv8n 训练 -> 本模块推理。

类别（9 类）：
  mm_self / mm_ally / mm_enemy / mm_ally_tower / mm_enemy_tower /
  mm_monster / mm_buff / mm_ally_minion / mm_enemy_minion

输出小地图归一化坐标（0-1，相对小地图框左上角）。
"""
import json
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]

MM_CLASSES = ["mm_self", "mm_ally", "mm_enemy",
              "mm_ally_tower", "mm_enemy_tower",
              "mm_monster", "mm_buff",
              "mm_ally_minion", "mm_enemy_minion"]


class MMYoloDetector:
    """小地图 YOLO 检测封装。输入整帧，输出结构化小地图信息。"""

    def __init__(self, weights=None, conf=0.4, iou=0.5):
        from ultralytics import YOLO
        self.weights = weights or str(ROOT / "runs" / "mm_detect" / "mm_v1"
                                      / "weights" / "best.pt")
        self.model = YOLO(self.weights)
        self.conf = conf
        self.iou = iou
        self.last_ms = 0.0
        print(f"[mm-yolo] 加载 {self.weights}")

    def detect(self, frame, mm_box=None):
        """检测小地图元素。

        mm_box: 小地图框 (x0, y0, x1, y1) 全屏像素；None 时用默认 (0,0,232,232)。
        返回: {"found": bool, "center": (cx,cy), "radius": r,
               "dots": {"self": [...], "ally": [...], "enemy": [...],
                        "monster": [...], "buff": [...]},
               "towers": {"ally": [...], "enemy": [...]},
               "minions": {"ally": [...], "enemy": [...]}}  坐标=小地图归一化 0-1
        """
        import time
        h, w = frame.shape[:2]
        if mm_box is None:
            # 默认小地图（1280x720 标定 0-232）
            mm_box = (0, 0, min(232, w), min(232, h))
        x0, y0, x1, y1 = mm_box
        crop = frame[y0:y1, x0:x1]
        if crop.size == 0:
            return {"found": False}
        t0 = time.time()
        res = self.model.predict(crop, conf=self.conf, iou=self.iou,
                                 imgsz=320, verbose=False)[0]
        self.last_ms = (time.time() - t0) * 1000

        out = {"found": True, "center": ((x0 + x1) / 2, (y0 + y1) / 2),
               "radius": (x1 - x0) / 2,
               "dots": {"self": [], "ally": [], "enemy": [],
                        "monster": [], "buff": []},
               "towers": {"ally": [], "enemy": []},
               "minions": {"ally": [], "enemy": []}}
        ch, cw = crop.shape[:2]
        names = res.names
        for box in res.boxes:
            cls_id = int(box.cls[0])
            cls = names[cls_id] if names else MM_CLASSES[cls_id]
            x1b, y1b, x2b, y2b = [float(v) for v in box.xyxy[0]]
            cx = ((x1b + x2b) / 2) / cw
            cy = ((y1b + y2b) / 2) / ch
            conf = float(box.conf[0])
            rec = {"n": [round(cx, 4), round(cy, 4)], "conf": round(conf, 3)}
            if cls == "mm_self":
                out["dots"]["self"].append(rec)
            elif cls == "mm_ally":
                out["dots"]["ally"].append(rec)
            elif cls == "mm_enemy":
                out["dots"]["enemy"].append(rec)
            elif cls == "mm_monster":
                out["dots"]["monster"].append(rec)
            elif cls == "mm_buff":
                out["dots"]["buff"].append(rec)
            elif cls == "mm_ally_tower":
                out["towers"]["ally"].append(rec)
            elif cls == "mm_enemy_tower":
                out["towers"]["enemy"].append(rec)
            elif cls == "mm_ally_minion":
                out["minions"]["ally"].append(rec)
            elif cls == "mm_enemy_minion":
                out["minions"]["enemy"].append(rec)
        return out

    def to_state(self, det):
        """转换为主循环 state_dict.minimap 兼容格式。

        det: detect() 返回值。
        返回 minimap dict（与 MinimapTracker.update 输出同构）：
          {"found", "center", "radius", "dots": {"green","blue","red","yellow"},
           "towers": [[nx,ny], ...]}
        """
        if not det.get("found"):
            return {"found": False}
        dots = {"green": [d["n"] for d in det["dots"]["self"]],
                "blue": [d["n"] for d in det["dots"]["ally"]],
                "red": [d["n"] for d in det["dots"]["enemy"]],
                "yellow": ([d["n"] for d in det["dots"]["monster"]]
                           + [d["n"] for d in det["dots"]["buff"]])}
        towers = ([t["n"] for t in det["towers"]["ally"]]
                  + [t["n"] for t in det["towers"]["enemy"]])
        return {"found": True, "center": det["center"], "radius": det["radius"],
                "dots": dots, "towers": towers}
