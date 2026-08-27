# -*- coding: utf-8 -*-
"""连续对局段(612-760)时序验证。"""
import sys
from pathlib import Path
import numpy as np
import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from wzry.vision.mm_tracker_v7 import MMTrackerV7

cap = cv2.VideoCapture(str(ROOT / "data" / "demos" / "20260818_125747" / "stream_0001.mkv"))
cap.set(cv2.CAP_PROP_POS_FRAMES, 612)
frames = []
for _ in range(150):
    ok, f = cap.read()
    if not ok:
        break
    frames.append(f)
cap.release()
print("帧数:", len(frames))

tr = MMTrackerV7()
tot = 0.0
save_at = 100
for idx, f in enumerate(frames):
    r = tr.update(f)
    tot += tr.last_ms
    v7 = r["v7"]
    if idx % 15 == 0:
        print(f"f{idx:02d}: 己{len(v7['self'])} 友{len(v7['ally'])} 敌{len(v7['enemy'])} "
              f"我兵{len(v7['minions']['ally'])} 敌兵{len(v7['minions']['enemy'])} "
              f"({tr.last_ms:.0f}ms) 敌位={[(round(d['n'][0],2),round(d['n'][1],2)) for d in v7['enemy']]}")
print(f"平均 {tot/len(frames):.0f}ms/帧")
