# -*- coding: utf-8 -*-
"""MMTrackerV7 顺序回放测试: s01-s12 时间序列, 输出各帧英雄/小兵数与耗时。"""
import sys
from pathlib import Path
import numpy as np
import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from wzry.vision.mm_tracker_v7 import MMTrackerV7

tr = MMTrackerV7()
tot_ms = 0.0
for i in range(1, 13):
    fp = ROOT / "temp" / "ann" / f"s{i:02d}.png"
    if not fp.exists():
        continue
    img = cv2.imdecode(np.fromfile(str(fp), dtype=np.uint8), cv2.IMREAD_COLOR)
    m = tr.update(img[:720, :1280])
    tot_ms += tr.last_ms
    v7 = m["v7"]
    print(f"s{i:02d} 自己{len(v7['self'])} 队友{len(v7['ally'])} 敌人{len(v7['enemy'])} "
          f"我兵{len(v7['minions']['ally'])} 敌兵{len(v7['minions']['enemy'])} "
          f"({tr.last_ms:.0f}ms)")
    if i in (1, 4, 5):
        print("   敌:", [(round(p[0], 2), round(p[1], 2)) for p, in [(d['n'],) for d in v7['enemy']]])
print(f"平均 {tot_ms/12:.0f}ms/帧")
