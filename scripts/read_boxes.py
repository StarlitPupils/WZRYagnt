# -*- coding: utf-8 -*-
"""自动读取人工标注: 从画好彩色矩形框的图读取标注 -> YOLO 标签。

约定颜色(在图上画框时用):
    红  = 敌方英雄
    黄  = 野怪
    蓝  = 我方英雄
    紫  = 自己
用法:
    1) 把框好的图放 temp/label_me/annotated/<原名>.png (或任意路径, --dir 指定)
    2) python scripts/read_boxes.py            # 默认扫 temp/label_me/annotated/
    3) 输出 temp/label_me/yolo/<原名>.txt
   win 画图/任意截图工具画矩形框即可; 也可以直接在原图上画(用 --overwrite 源)
"""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
COLOR_MAP = {  # BGR 近似 + 容差
    "enemy": ((0, 0, 255), 70),
    "monster": ((0, 255, 255), 70),
    "ally": ((255, 100, 0), 80),
    "self": ((255, 0, 200), 80),
}
CLS = {"enemy": 0, "ally": 1, "self": 1, "monster": 8}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="temp/label_me/annotated")
    args = ap.parse_args()
    d = ROOT / args.dir
    out = ROOT / "temp" / "label_me" / "yolo"
    out.mkdir(parents=True, exist_ok=True)
    n_total = 0
    for f in sorted(d.glob("*.png")):
        img = cv2.imdecode(np.fromfile(str(f), dtype=np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            continue
        h, w = img.shape[:2]
        found = []
        for name, (bgr, tol) in COLOR_MAP.items():
            diff = np.abs(img.astype(np.int16) - np.array(bgr, np.int16)).sum(axis=-1)
            m = (diff < tol * 3).astype(np.uint8) * 255
            m = cv2.morphologyEx(m, cv2.MORPH_OPEN,
                                 cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)))
            n, lab, st, cent = cv2.connectedComponentsWithStats(m, 8)
            for i in range(1, n):
                x, y, ww, hh, area = st[i]
                # 矩形框特征: 长/宽>=25, 面积中等(框线面积远小于框内)
                if ww > 25 and hh > 25 and ww < w * 0.9 and area > 60:
                    # 框线样式: 空心矩形(角点颜色占比) 或 实心小目标 -> 视作框
                    found.append((name, x, y, x + ww, y + hh))
        # 去重(同类重叠)
        names = ["enemy", "monster", "ally", "self"]
        keep = []
        for fnd in found:
            dup = False
            for k in keep:
                if same(fnd, k, names):
                    dup = True
                    break
            if not dup:
                keep.append(fnd)
        lines = []
        for (nm, x1, y1, x2, y2) in keep:
            cx, cy = ((x1 + x2) / 2) / w, ((y1 + y2) / 2) / h
            bw, bh = (x2 - x1) / w, (y2 - y1) / h
            lines.append(f"{CLS[nm]} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
        if lines:
            (out / (f.stem + ".txt")).write_text("\n".join(lines) + "\n", encoding="utf-8")
            print(f"{f.stem}: {len(lines)} 框  {[l.split()[0] for l in lines]}")
            n_total += len(lines)
        else:
            print(f"{f.stem}: 未识别到框(检查颜色约定: 红=敌 黄=野 蓝=友 紫=我)")
    print(f"总计 {n_total} 框 -> temp/label_me/yolo/")


def same(a, b, names):
    if a[0] != b[0]:
        return False
    ax1, ay1, ax2, ay2 = a[1:]; bx1, by1, bx2, by2 = b[1:]
    return abs(ax1 - bx1) < 20 and abs(ay1 - by1) < 20 and abs((ax2 - ax1) - (bx2 - bx1)) < 20


if __name__ == "__main__":
    main()
