# -*- coding: utf-8 -*-
"""从多个 demo 视频抽帧（对局中段均匀采样，跳过开场/结算）。"""
import sys
from pathlib import Path
import numpy as np
import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

VIDEOS = [
    "data/demos/20260818_122514/stream_0001.mkv",
    "data/demos/20260818_122818/stream_0001.mkv",
    "data/demos/20260818_122941/stream_0001.mkv",
    "data/demos/20260818_123054/stream_0001.mkv",
]
OUT = ROOT / "temp" / "frames2"
OUT.mkdir(parents=True, exist_ok=True)

PER_VIDEO = 8
for vi, vp in enumerate(VIDEOS):
    cap = cv2.VideoCapture(vp)
    if not cap.isOpened():
        print(vp, "打不开")
        continue
    # 读全部帧计数（属性不可靠时顺序读）
    frames = []
    n = 0
    while True:
        ok, f = cap.read()
        if not ok:
            break
        frames.append(f)
        n += 1
        if n > 200000:
            break
    cap.release()
    print(vp, "可读帧数:", n)
    if n == 0:
        continue
    idxs = [int(n * (0.15 + 0.7 * k / (PER_VIDEO))) for k in range(PER_VIDEO)]
    for k, idx in enumerate(idxs):
        f = frames[idx]
        out = OUT / f"v{vi}_f{k:02d}.png"
        cv2.imwrite(str(out), f)
    print("  → 抽取", idxs)
print("完成")
