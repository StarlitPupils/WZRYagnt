# -*- coding: utf-8 -*-
"""打印 ally/enemy/self 候选两两距离 <15px 的对。"""
import sys
from pathlib import Path
import numpy as np
import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from wzry.vision.mm_rules_v7 import MMDetectorV7

det = MMDetectorV7()
for i in (1, 2, 3):
    img = cv2.imdecode(np.fromfile(str(ROOT / "temp" / "ann" / f"s{i:02d}.png"), dtype=np.uint8), cv2.IMREAD_COLOR)
    r = det.detect(img[:720, :1280])
    pix = {"ally": [(int(d["n"][0] * 232), int(d["n"][1] * 232)) for d in r["dots"]["ally"]],
           "enemy": [(int(d["n"][0] * 232), int(d["n"][1] * 232)) for d in r["dots"]["enemy"]],
           "self": [(int(d["n"][0] * 232), int(d["n"][1] * 232)) for d in r["dots"]["self"]]}
    print(f"s{i:02d} ally={pix['ally']} enemy={pix['enemy']} self={pix['self']}")
    for a in pix["ally"]:
        for e in pix["enemy"]:
            d = ((a[0] - e[0]) ** 2 + (a[1] - e[1]) ** 2) ** 0.5
            if d < 16:
                print(f"  ally{a}~enemy{e} dist={d:.1f}")
