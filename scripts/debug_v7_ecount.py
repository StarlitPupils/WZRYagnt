# -*- coding: utf-8 -*-
"""每帧敌人数 vs GT。"""
import sys
from pathlib import Path
import numpy as np
import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from wzry.vision.mm_rules_v7 import MMDetectorV7

det = MMDetectorV7()
for i in range(1, 13):
    fp = ROOT / "temp" / "ann" / f"s{i:02d}.png"
    if not fp.exists():
        continue
    img = cv2.imdecode(np.fromfile(str(fp), dtype=np.uint8), cv2.IMREAD_COLOR)
    r = det.detect(img[:720, :1280])
    gt = sum(1 for line in fp.with_suffix(".txt").read_text(encoding="utf-8").splitlines()
             if line.strip() and int(line.split()[0]) == 13)
    pos = [(d["n"][0], d["n"][1]) for d in r["dots"]["enemy"]]
    print(f"s{i:02d} enemy det={len(pos)} gt={gt} pos={[(round(a,2), round(b,2)) for a, b in pos]}")
