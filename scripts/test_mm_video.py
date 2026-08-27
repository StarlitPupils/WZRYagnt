# -*- coding: utf-8 -*-
"""连续视频段时序验证：弧段吸收/伪点过滤/耗时。"""
import sys
from pathlib import Path
import numpy as np
import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from wzry.vision.mm_tracker_v7 import MMTrackerV7

cap = cv2.VideoCapture(str(ROOT / "data" / "demos" / "20260818_125747" / "stream_0001.mkv"))
cap.set(cv2.CAP_PROP_POS_FRAMES, 2400)   # 跳过开场, 中段
frames = []
for _ in range(80):
    ok, f = cap.read()
    if not ok:
        break
    frames.append(f)
cap.release()
print("帧数:", len(frames))

tr = MMTrackerV7()
tot = 0.0
stat = {"self": [], "ally": [], "enemy": [], "mally": [], "menemy": []}
for idx, f in enumerate(frames):
    r = tr.update(f)
    tot += tr.last_ms
    v7 = r["v7"]
    for k, key in (("self", "self"), ("ally", "ally"), ("enemy", "enemy"),
                   ("mally", "minions/ally"), ("menemy", "minions/enemy")):
        if key.startswith("minions"):
            val = len(v7["minions"][key.split("/")[1]])
        else:
            val = len(v7[key])
        stat[k].append(val)
    if idx % 10 == 0:
        print(f"f{idx:02d}: 己{stat['self'][-1]} 友{stat['ally'][-1]} 敌{stat['enemy'][-1]} "
              f"我兵{stat['mally'][-1]} 敌兵{stat['menemy'][-1]} ({tr.last_ms:.0f}ms)")
print(f"平均 {tot/len(frames):.0f}ms/帧")
for k in stat:
    arr = np.array(stat[k])
    print(f"{k}: 均值{arr.mean():.1f} 中位{np.median(arr):.0f} 最大{arr.max()}")
