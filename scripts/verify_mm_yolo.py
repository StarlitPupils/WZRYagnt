# -*- coding: utf-8 -*-
"""小地图 YOLO 真实帧验证：检测结果画到小地图上，输出对比图。

用法：
    venv\\Scripts\\python.exe scripts\\verify_mm_yolo.py <帧图> [权重]
"""
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from wzry.vision.mm_yolo import MMYoloDetector  # noqa: E402

COLORS = {"self": (0, 255, 0), "ally": (255, 128, 0), "enemy": (0, 0, 255),
          "monster": (0, 255, 255), "buff": (255, 0, 255)}
LABELS = {"self": "SELF", "ally": "ALLY", "enemy": "ENEMY",
          "monster": "MONSTER", "buff": "BUFF"}


def main():
    path = sys.argv[1]
    weights = sys.argv[2] if len(sys.argv) > 2 else None
    img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
    det = MMYoloDetector(weights=weights)
    r = det.detect(img)
    vis = img.copy()
    if not r.get("found"):
        print("未找到小地图")
        return
    x0, y0 = int(r["center"][0] - r["radius"]), int(r["center"][1] - r["radius"])
    r2 = int(r["radius"] * 2)
    cv2.rectangle(vis, (x0, y0), (x0 + r2, y0 + r2), (255, 200, 0), 2)
    for grp, color in COLORS.items():
        for d in r["dots"][grp]:
            px = int(x0 + d["n"][0] * r2)
            py = int(y0 + d["n"][1] * r2)
            cv2.circle(vis, (px, py), 6, color, -1)
            cv2.putText(vis, f"{LABELS[grp]} {d['conf']:.2f}", (px + 8, py),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
    for side, color in (("ally", (200, 100, 0)), ("enemy", (0, 100, 200))):
        for t in r["towers"][side]:
            px = int(x0 + t["n"][0] * r2)
            py = int(y0 + t["n"][1] * r2)
            cv2.rectangle(vis, (px - 6, py - 6), (px + 6, py + 6), color, -1)
            cv2.putText(vis, f"{side.upper()}_TW {t['conf']:.2f}", (px + 8, py),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1)
    for side, color in (("ally", (150, 255, 150)), ("enemy", (150, 150, 255))):
        for m in r["minions"][side]:
            px = int(x0 + m["n"][0] * r2)
            py = int(y0 + m["n"][1] * r2)
            cv2.circle(vis, (px, py), 3, color, -1)
    out = ROOT / "temp" / f"mm_verify_{Path(path).stem}.png"
    cv2.imwrite(str(out), vis)
    print("已保存:", out)
    print("self=%d ally=%d enemy=%d monster=%d buff=%d | 蓝塔=%d 红塔=%d | 我兵=%d 敌兵=%d (%.0fms)" % (
        len(r["dots"]["self"]), len(r["dots"]["ally"]), len(r["dots"]["enemy"]),
        len(r["dots"]["monster"]), len(r["dots"]["buff"]),
        len(r["towers"]["ally"]), len(r["towers"]["enemy"]),
        len(r["minions"]["ally"]), len(r["minions"]["enemy"]), det.last_ms))


if __name__ == "__main__":
    main()
