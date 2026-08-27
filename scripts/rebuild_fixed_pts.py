# -*- coding: utf-8 -*-
"""从 s01-s12 标注重聚合塔/野怪/buff 固定点 (minimap 类 14/15/16/17)。"""
import os
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MM = 242.0


def main():
    pts = {"blue_t": [], "red_t": [], "mon": [], "buff": []}
    for f in sorted((ROOT / "temp" / "ann").glob("s*.txt")):
        for l in f.read_text(encoding="utf-8").splitlines():
            p = l.split()
            if len(p) < 5:
                continue
            c = int(p[0])
            # GT minimap 类=全屏归一化 -> 小地图像素 = x*1280 (map at 0..242px)
            x, y = float(p[1]) * 1280.0, float(p[2]) * 720.0
            if c == 14:
                pts["blue_t"].append((x, y))
            elif c == 15:
                pts["red_t"].append((x, y))
            elif c == 16:
                pts["mon"].append((x, y))
            elif c == 17:
                pts["buff"].append((x, y))

    def cluster(arr, d=12):
        out = []
        for x, y in arr:
            for o in out:
                if (o[0] - x) ** 2 + (o[1] - y) ** 2 < d * d:
                    o[0] = (o[0] + x) / 2
                    o[1] = (o[1] + y) / 2
                    break
            else:
                out.append([x, y])
        return out

    for k, v in pts.items():
        c = cluster(v)
        print(k, len(v), "->", len(c))
    # 输出 python 常量
    def fmt(arr):
        return "[" + ", ".join(f"({x / MM:.3f}, {y / MM:.3f})" for x, y in
                               cluster(arr, 20).__iter__() and [0] or []) if False else \
            "[" + ", ".join(f"({x / MM:.3f}, {y / MM:.3f})" for x, y in cluster(arr)) + "]"
    txt = (f"BLUE_TOWER_PTS = {fmt(pts['blue_t'])}\n"
           f"RED_TOWER_PTS = {fmt(pts['red_t'])}\n"
           f"MONSTER_PTS = {fmt(pts['mon'])}\n"
           f"BUFF_PTS = {fmt(pts['buff'])}\n")
    (ROOT / "temp" / "fixed_pts.py").write_text(txt, encoding="utf-8")
    print(txt)


if __name__ == "__main__":
    main()
