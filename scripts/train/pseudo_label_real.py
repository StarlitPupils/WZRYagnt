# -*- coding: utf-8 -*-
"""真实帧伪标注（v4 模型高置信检测 -> 混合训练 v5）。

流程：
  1. 扫描演示视频 + 录像的对局帧（小地图绿色底判定）
  2. 用 mm_v4 模型检测，保留 conf>=0.55 的 self/ally/enemy 检测
  3. 输出 data/mm_real_pseudo/（帧 + txt），供混合训练

用法：
  venv\\Scripts\\python.exe scripts\\train\\pseudo_label_real.py [--conf 0.55] [--max-frames 1500]
"""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--conf", type=float, default=0.55)
    ap.add_argument("--max-frames", type=int, default=1500)
    args = ap.parse_args()

    from ultralytics import YOLO
    model = YOLO(str(ROOT / "runs" / "mm_detect" / "mm_v4" / "weights" / "best.pt"))

    out_dir = ROOT / "data" / "mm_real_pseudo"
    img_dir = out_dir / "images"
    lbl_dir = out_dir / "labels"
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)

    videos = list((ROOT / "data" / "demos").glob("*/stream_0001.mkv"))
    videos.append(ROOT / "temp" / "stream_0001.mkv")  # 实机录像
    n_saved = 0
    for v in videos:
        if not v.exists():
            continue
        print(f"处理 {v.parent.name} ...")
        cap = cv2.VideoCapture(str(v))
        n = 0
        while True:
            ok, fr = cap.read()
            if not ok:
                break
            n += 1
            if n % 10 != 0:
                continue
            h, w = fr.shape[:2]
            roi = fr[0:int(0.322 * h), 0:int(0.181 * w)]
            hsv0 = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
            H0, S0, V0 = (hsv0[..., 0].astype(int), hsv0[..., 1].astype(int),
                          hsv0[..., 2].astype(int))
            if int(np.sum((H0 >= 35) & (H0 <= 90) & (S0 > 80) & (V0 > 90))) < 50:
                continue  # 非对局
            mm = fr[0:232, 0:232]
            res = model.predict(mm, conf=args.conf, imgsz=320, verbose=False)[0]
            labels = []
            for box in res.boxes:
                cls = int(box.cls[0])
                if cls not in (0, 1, 2):   # 只留英雄标记类
                    continue
                x1b, y1b, x2b, y2b = [float(v) for v in box.xyxy[0]]
                cx = (x1b + x2b) / 2 / 232
                cy = (y1b + y2b) / 2 / 232
                bw = (x2b - x1b) / 232
                bh = (y2b - y1b) / 232
                labels.append(f"{cls} {cx:.4f} {cy:.4f} {bw:.4f} {bh:.4f}")
            if not labels:
                continue
            name = f"pseudo_{n_saved:05d}"
            cv2.imwrite(str(img_dir / f"{name}.png"), mm)
            (lbl_dir / f"{name}.txt").write_text("\n".join(labels), encoding="utf-8")
            n_saved += 1
            if n_saved >= args.max_frames:
                break
        cap.release()
    print(f"伪标注完成: {n_saved} 张 -> {out_dir}")


if __name__ == "__main__":
    main()
