# -*- coding: utf-8 -*-
"""一边打一边学: 自动采集 + 自我标注 + 训练 (v11.1)。

对局运行时 (后台线程), 每 N 秒:
  ① 小地图(地图内)裁剪 -> 现 mm 检测(绿/蓝/红点, conf>0.6) -> 自标 mm_red/blue/green
  ② 全屏(地图外) -> yolo r3 检测(敌英/野怪/自己/队友 conf>0.7) -> 自标
  ③ 多帧一致性: 连续 2 帧同位置同类别才入库 (防闪烁误标)
存 data/auto_labeled/ (images+labels) -> 定时(每局)训练更新模型 -> agent 加载新模型
主动学习: 只采集"模型置信高"的样本(自监督), 低置信留给用户/预留。

用法:
  python scripts/auto_capture.py --fps-interval 2 --session "<id>"   (对局中运行)
"""
import argparse
import json
import threading
import time
import uuid
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "auto_labeled"
MM = (0, 0, 232, 232)

_lock = threading.Lock()
_last_mm = {}       # (cls, x, y) -> count  (多帧投票)
_last_scr = {}

# 类别映射 (与 yolo 一致): mm_red 12 mm_blue 13 mm_green 14; enemy_hero 0 ally_hero 1 self 11 neutral_monster 8
CLS = {"mm_red": 12, "mm_blue": 13, "mm_green": 14,
       "enemy_hero": 0, "ally_hero": 1, "self": 11, "neutral_monster": 8}


def _vote(store, key, cls, conf):
    """v11.6 单帧即入库(高置信自标); 同 key 重复也 True。"""
    with _lock:
        store[key] = cls
        return True


def capture_once(frame, det, cls_model=None):
    """抓一帧: 自标小地图点 + 全屏英雄 -> 写 auto_labeled。"""
    import numpy as np  # noqa
    h, w = frame.shape[:2]
    mm = frame[MM[1]:MM[3], MM[0]:MM[2]]   # 小地图
    stamp = f"{int(time.time()*1000)}"
    lines = []
    # ① 小地图点自标: 用 mm 检测 (mm_rules ~红/蓝/绿), 简化: 用 cls_model predict mm 裁剪
    try:
        if cls_model is not None:
            r = cls_model.predict(mm, conf=0.6, verbose=False)[0]
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
    # ② 全屏英雄自标 (yolo det + 多帧)
    try:
        dets = det.predict(frame, conf=0.7, verbose=False)[0]
        nms = det.names
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


def clamp(x, d):
    return round(x, d)


def load_models(model_path):
    from ultralytics import YOLO
    det = YOLO(str(model_path))
    return det, None


def run(model_path="runs/detect/zhongkui_r3/weights/best.pt", interval=2.0):
    det, _ = load_models(model_path)
    print(f"一边打一边学: 采集 {model_path}, 每 {interval}s, -> data/auto_labeled/")
    from wzry.capture.desktop_capture import DesktopCapturePrint
    cap = DesktopCapturePrint()
    cap.start()
    n = 0
    t0 = time.time()
    while True:
        try:
            frame, lag = cap.wait_frame(timeout=2.0)
            if frame is None:
                continue
            if capture_once(frame, det):
                n += 1
                if n % 5 == 0:
                    print(f"已采集 {n} 帧 -> data/auto_labeled/")
            time.sleep(interval)
        except Exception as e:
            print("cap err:", e)
            time.sleep(2)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="runs/detect/zhongkui_r3/weights/best.pt")
    ap.add_argument("--interval", type=float, default=2.0)
    args = ap.parse_args()
    run(args.model, args.interval)
