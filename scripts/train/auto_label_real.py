# -*- coding: utf-8 -*-
"""真实对局帧自动标注（轨迹法）：移动的彩色圈=英雄标记（可靠真值）。

原理：英雄标记随英雄移动（位置逐帧变化），塔/野怪/地图元素静止。
对连续帧序列检测彩色候选 -> 跨帧关联 -> 移动方差大的 = 英雄标记。

输出: data/mm_real_annot/ 下（帧图 + YOLO txt），供混合训练。

用法:
  venv\\Scripts\\python.exe scripts\\train\\auto_label_real.py [--max-frames 800]
"""
import argparse
import shutil
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]

COLOR_DEFS = {
    "mm_self": ((35, 90), 90, 80),      # 绿
    "mm_ally": ((90, 135), 90, 80),     # 蓝
    "mm_enemy": ((165, 180), 90, 80),   # 红（含 wrap）
}
CLS_IDX = {"mm_self": 0, "mm_ally": 1, "mm_enemy": 2}


def find_candidates(mm):
    """返回 [(cls, cx, cy, size), ...] 英雄标记候选（大圈 comp>=0.45）。"""
    hsv = cv2.cvtColor(mm, cv2.COLOR_BGR2HSV)
    H, S, V = hsv[..., 0].astype(int), hsv[..., 1].astype(int), hsv[..., 2].astype(int)
    out = []
    for cls, (hlo, hhi), smin, vmin in [
        (c, h, s, v) for c, (h, s, v) in COLOR_DEFS.items()]:
        if hhi > 180:
            m = ((H >= hlo) | (H < hhi - 180)) & (S > smin) & (V > vmin)
        else:
            m = (H >= hlo) & (H < hhi) & (S > smin) & (V > vmin)
        mask = m.astype(np.uint8) * 255
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
        n, lab, st, cent = cv2.connectedComponentsWithStats(mask, 8)
        for i in range(1, n):
            area = int(st[i, cv2.CC_STAT_AREA])
            w_ = int(st[i, cv2.CC_STAT_WIDTH])
            h_ = int(st[i, cv2.CC_STAT_HEIGHT])
            comp = area / max(1.0, w_ * h_)
            if not (60 <= area <= 500 and comp >= 0.45):
                continue
            cx, cy = int(cent[i][0]), int(cent[i][1])
            out.append((cls, cx, cy, max(w_, h_)))
    return out


def track_and_label(video, out_dir, max_frames=800, step=3, window=30):
    """扫描视频，用移动轨迹识别英雄标记，输出标注样本。"""
    cap = cv2.VideoCapture(video)
    frames = []
    n = 0
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        n += 1
        if n % step != 0:
            continue
        # 对局判断：左上角小地图区有绿色底（小地图草地）
        h, w = fr.shape[:2]
        roi = fr[0:int(0.322 * h), 0:int(0.181 * w)]
        hsv0 = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        H0 = hsv0[..., 0].astype(int)
        S0 = hsv0[..., 1].astype(int)
        V0 = hsv0[..., 2].astype(int)
        green_bg = int(np.sum((H0 >= 35) & (H0 <= 90) & (S0 > 80) & (V0 > 90)))
        if green_bg < 50:
            continue
        frames.append(fr)
        if len(frames) >= max_frames:
            break
    cap.release()
    if not frames:
        print("无对局帧")
        return 0
    print(f"读取 {len(frames)} 帧对局画面")

    # 每帧候选
    all_cands = []
    for fr in frames:
        mm = fr[0:232, 0:232]
        all_cands.append(find_candidates(mm))

    # 逐帧最近邻跟踪（量化 key 会丢失移动信息，改用连续坐标）
    tracks = {}     # tid -> {"pts": [(cx, cy)...], "cls": str, "start": fi}
    prev = {}       # tid -> (cx, cy)
    next_id = 0
    for fi, cands in enumerate(all_cands):
        matched = set()
        for cls, cx, cy, size in cands:
            best, bd = None, 64.0
            for tid, (tcx, tcy) in prev.items():
                d = (cx - tcx) ** 2 + (cy - tcy) ** 2
                if d < bd:
                    bd, best = d, tid
            if best is not None and bd < 25.0:  # <5px 视为同一标记
                tracks[best]["pts"].append((cx, cy))
                prev[best] = (cx, cy)
                matched.add(best)
            else:
                prev[next_id] = (cx, cy)
                tracks[next_id] = {"pts": [(cx, cy)], "cls": cls, "start": fi}
                next_id += 1
        for tid in list(prev.keys()):
            if tid not in matched:
                del prev[tid]

    # 移动判定：轨迹跨度（max-min）>= 7px = 英雄标记（塔/地图元素静止）
    labeled = {}
    for tid, tr in tracks.items():
        pts = tr["pts"]
        if len(pts) < 6:
            continue
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        span = max(max(xs) - min(xs), max(ys) - min(ys))
        if span < 7.0:
            continue
        fi0 = tr["start"]
        labeled.setdefault(fi0, []).append((tr["cls"], pts[0][0], pts[0][1]))

    # 输出样本
    out_dir.mkdir(parents=True, exist_ok=True)
    img_dir = out_dir / "images"
    lbl_dir = out_dir / "labels"
    img_dir.mkdir(exist_ok=True)
    lbl_dir.mkdir(exist_ok=True)
    n_saved = 0
    for fi, fr in enumerate(frames):
        labels = labeled.get(fi)
        if not labels:
            continue
        name = f"real_{fi:05d}"
        cv2.imwrite(str(img_dir / f"{name}.png"), fr[0:232, 0:232])
        with open(lbl_dir / f"{name}.txt", "w") as f:
            for cls, cx, cy in labels:
                f.write(f"{CLS_IDX[cls]} {cx/232:.4f} {cy/232:.4f} {16/232:.4f} {16/232:.4f}\n")
        n_saved += 1
    print(f"输出 {n_saved} 张标注样本 -> {out_dir}")
    return n_saved


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-frames", type=int, default=800)
    args = ap.parse_args()
    videos = list((ROOT / "data" / "demos").glob("*/stream_0001.mkv"))
    total = 0
    for v in videos:
        print(f"处理 {v.parent.name} ...")
        total += track_and_label(str(v), ROOT / "data" / "mm_real_annot",
                                 max_frames=args.max_frames)
    print(f"总计 {total} 张")


if __name__ == "__main__":
    main()
