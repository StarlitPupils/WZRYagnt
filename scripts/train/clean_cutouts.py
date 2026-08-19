# -*- coding: utf-8 -*-
"""抠图预处理：自动去除 Snipaste 截图背景，生成带 alpha 的干净抠图。

输入: data/mm_cutouts/<class>/*.png （用户截图，标记居中，背景为深色小地图）
输出: data/mm_cutouts_clean/<class>/*.png （带 alpha，仅标记区域）

标记特征：高饱和亮色像素（绿/蓝/红圈、塔、野怪点）。
"""
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
CLASSES = ["mm_self", "mm_ally", "mm_enemy",
           "mm_ally_tower", "mm_enemy_tower",
           "mm_monster", "mm_buff",
           "mm_ally_minion", "mm_enemy_minion"]


def clean_cutout(img):
    """返回带 alpha 的裁剪图（标记区域+羽化透明背景）。"""
    h, w = img.shape[:2]
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    H, S, V = hsv[..., 0].astype(int), hsv[..., 1].astype(int), hsv[..., 2].astype(int)
    # 标记 = 高饱和亮色（排除深色地图背景）
    mask = ((S > 60) & (V > 50)).astype(np.uint8) * 255
    # 取最大连通域（标记本体）
    n, lab, st, cent = cv2.connectedComponentsWithStats(mask, 8)
    if n <= 1:
        return None
    areas = [(st[i, cv2.CC_STAT_AREA], i) for i in range(1, n)]
    areas.sort(reverse=True)
    main = areas[0][1]
    m = (lab == main).astype(np.uint8) * 255
    # 外接框（含箭头）
    ys, xs = np.where(m > 0)
    x0, y0, x1, y1 = xs.min(), ys.min(), xs.max(), ys.max()
    pad = max(2, int((x1 - x0) * 0.15))
    x0, y0 = max(0, x0 - pad), max(0, y0 - pad)
    x1, y1 = min(w - 1, x1 + pad), min(h - 1, y1 + pad)
    crop = img[y0:y1 + 1, x0:x1 + 1]
    cm = m[y0:y1 + 1, x0:x1 + 1]
    # 羽化 alpha：标记 255，周围膨胀过渡
    alpha = cv2.GaussianBlur(cm.astype(np.float32), (7, 7), 2)
    alpha = np.clip(alpha * 1.5, 0, 255).astype(np.uint8)
    rgba = cv2.cvtColor(crop, cv2.COLOR_BGR2BGRA)
    rgba[..., 3] = alpha
    return rgba


def main():
    src = ROOT / "data" / "mm_cutouts"
    dst = ROOT / "data" / "mm_cutouts_clean"
    total = 0
    for cls in CLASSES:
        d = src / cls
        if not d.exists():
            continue
        od = dst / cls
        od.mkdir(parents=True, exist_ok=True)
        n = 0
        for f in sorted(d.glob("*.png")) + sorted(d.glob("*.jpg")):
            data = np.fromfile(str(f), dtype=np.uint8)
            img = cv2.imdecode(data, cv2.IMREAD_UNCHANGED)
            if img is None:
                continue
            if img.ndim == 3 and img.shape[2] == 4:
                img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
            rgba = clean_cutout(img)
            if rgba is None:
                print(f"  {cls}/{f.name}: 无法定位标记")
                continue
            out = od / (f.stem + ".png")
            cv2.imwrite(str(out), rgba)
            n += 1
        print(f"{cls}: {n} 张")
        total += n
    print(f"完成，共 {total} 张 -> {dst}")


if __name__ == "__main__":
    main()
