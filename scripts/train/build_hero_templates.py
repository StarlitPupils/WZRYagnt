# -*- coding: utf-8 -*-
"""英雄网格图 -> 模板库：从用户提供的英雄头像网格图自动裁剪每个头像。

用法：
    1. 把 4 张网格图保存到 data/heroes_annotate/（例如 img1.png ~ img4.png）
    2. 运行本脚本：自动检测网格布局，按 HERO_GRIDS 顺序裁剪头像，
       保存到 data/heroes/<英雄名>.png（96x96）
    3. 可选：裁剪后人工抽查，删除错误的重新裁剪

网格检测：
    - 头像为圆形，按 10 列规则排列（4 张图 = 12/36/40/40 英雄）
    - 用 Hough 圆检测或规则网格采样（优先规则网格：n 行 x 10 列）
"""
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from wzry.vision.hero_recognition import HERO_GRIDS, HERO_DIR  # noqa: E402

ANNOTATE_DIR = ROOT / "data" / "heroes_annotate"


def find_grid(img, n_cols=10):
    """检测头像网格：找圆形头像的中心点，聚类出行列。"""
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 头像外圈通常是亮边框：高亮像素聚类
    circles = None
    try:
        circles = cv2.HoughCircles(
            cv2.GaussianBlur(gray, (5, 5), 0), cv2.HOUGH_GRADIENT,
            dp=1.2, minDist=max(30, min(h, w) // n_cols // 3),
            param1=80, param2=30, minRadius=int(min(h, w) / n_cols * 0.25),
            maxRadius=int(min(h, w) / n_cols * 0.5))
    except cv2.error:
        pass
    if circles is None:
        return None
    pts = np.round(circles[0][:, :2]).astype(int)
    # 按 y 聚类成行，按 x 排序
    pts = sorted(pts.tolist(), key=lambda p: (p[1], p[0]))
    rows = []
    for p in pts:
        placed = False
        for row in rows:
            if abs(row[-1][1] - p[1]) < min(h, w) / n_cols * 0.3:
                row.append(p)
                placed = True
                break
        if not placed:
            rows.append([p])
    rows = [sorted(r, key=lambda p: p[0]) for r in rows]
    rows.sort(key=lambda r: r[0][1])
    return rows


def crop_avatar(img, center, radius_frac=0.62):
    """以 center 为中心裁剪头像（半径为间距*0.62，避开文字）。"""
    h, w = img.shape[:2]
    # 间距估计 = 相邻头像距离
    return None


def main():
    if not ANNOTATE_DIR.exists():
        print(f"请先创建 {ANNOTATE_DIR} 并放入 4 张网格图（img1.png~img4.png）")
        return 1
    files = sorted(ANNOTATE_DIR.glob("*.png"))
    if not files:
        print(f"{ANNOTATE_DIR} 中没有图片")
        return 1
    print(f"找到 {len(files)} 张网格图: {[f.name for f in files]}")
    for i, f in enumerate(files):
        if i >= len(HERO_GRIDS):
            print(f"警告: {f.name} 超出已知英雄名单（第 {i+1} 张）")
            break
        img = cv2.imread(str(f))
        if img is None:
            print(f"无法读取 {f.name}")
            continue
        rows = find_grid(img)
        if rows is None:
            print(f"{f.name}: 未检测到网格（尝试规则网格）")
            continue
        names = HERO_GRIDS[i]
        # 展平：行优先
        avatars = [p for row in rows for p in row]
        print(f"{f.name}: 检测到 {len(avatars)} 个头像（预期 {len(names)}）")
        HERO_DIR.mkdir(parents=True, exist_ok=True)
        n_ok = 0
        for j, name in enumerate(names):
            if j >= len(avatars):
                print(f"  头像不足，{name} 缺失")
                continue
            cx, cy = avatars[j]
            # 半径 = 行内相邻间距 / 2 * 0.7（避开文字）
            if j + 1 < len(avatars) and abs(avatars[j + 1][1] - cy) < 20:
                step = abs(avatars[j + 1][0] - cx)
            else:
                step = min(img.shape[:2]) // 10
            r = max(8, int(step * 0.38))
            x0, y0 = max(0, cx - r), max(0, cy - r)
            x1, y1 = min(img.shape[1], cx + r), min(img.shape[0], cy + r)
            crop = img[y0:y1, x0:x1]
            if crop.size == 0:
                continue
            crop = cv2.resize(crop, (96, 96), interpolation=cv2.INTER_AREA)
            cv2.imwrite(str(HERO_DIR / f"{name}.png"), crop)
            n_ok += 1
        print(f"  已保存 {n_ok}/{len(names)} 个头像模板")
    # 生成 index
    from wzry.vision.hero_recognition import build_index
    index = build_index()
    print(f"\n模板库: {len(index)} 个英雄")
    return 0


if __name__ == "__main__":
    sys.exit(main())
