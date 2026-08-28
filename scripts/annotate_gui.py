# -*- coding: utf-8 -*-
"""可视化标注器 (OpenCV GUI): 在图上拖框选类别 -> 保存 YOLO 归一化 txt。

用法:
    python scripts/annotate_gui.py <图片目录> [--close 2]
        <图片目录> 默认 data/labeling/in_mm/  (小地图内素材)
        out_mm 素材: python scripts/annotate_gui.py data/labeling/out_mm/images/

操作:
    鼠标左键 按下->拖动->松开 = 画一个框
    框出现后按 数字 选类别 (0-17, 见图例)
    回车/Enter = 确认当前框;  Backspace = 删除最后一个框
    n/N = 下一张;  p/P = 上一张;  s/S = 跳过(不标);  q/Q = 退出
    每张图保存同名 .txt (与 label_tools.py 格式一致)

类别:
    0 enemy_hero  1 ally_hero   2 enemy_minion 3 ally_minion
    4 enemy_turret 5 ally_turret 6 enemy_crystal 7 ally_crystal
    8 neutral_monster 9 hook_aim 10 skill_effect 11 self
    -- 小地图内专用 --
    12 mm_red     13 mm_blue    14 mm_green    15 mm_yellow
    16 mm_monster 17 mm_tower
"""
import argparse
import glob
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CLASSES = [
    "enemy_hero", "ally_hero", "enemy_minion", "ally_minion",
    "enemy_turret", "ally_turret", "enemy_crystal", "ally_crystal",
    "neutral_monster", "hook_aim", "skill_effect", "self",
    "mm_red", "mm_blue", "mm_green", "mm_yellow", "mm_monster", "mm_tower",
]

WIN = "annotate"
COLORS = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (0, 255, 255),
          (255, 0, 255), (255, 255, 0), (128, 128, 255), (255, 128, 128),
          (128, 0, 128), (0, 128, 255), (255, 128, 0), (0, 255, 128),
          (80, 80, 255), (255, 80, 80), (80, 255, 80), (255, 200, 80),
          (200, 80, 255), (128, 200, 128)]


class Annotator:
    def __init__(self, img_dir, cls_filter=None):
        self.dir = Path(img_dir)
        self.files = sorted(self.dir.glob("*.png"))
        if not self.files:
            print("目录无图片:", self.dir)
            sys.exit(1)
        self.idx = 0
        self.img = None
        self.boxes = []          # [(x1,y1,x2,y2,cls)]
        self.drag_start = None
        self.pending_cls = 0

    def load(self):
        p = self.dir / "labels" if False else self.files[self.idx]
        self.img = cv2.imread(str(p))
        self.img_big = self.img
        self.h, self.w = self.img.shape[:2]
        txt = self.dir.parent / f"{p.stem}.txt"
        # 未标目录: 同级姐妹目录存 labels 或图片旁 txt (label_tools 约定: 图片旁同名 txt)
        txt = p.with_suffix(".txt")
        self.boxes = []
        if txt.exists():
            for ln in txt.read_text(encoding="utf-8").strip().splitlines():
                parts = ln.split()
                if len(parts) != 5:
                    continue
                cls = int(float(parts[0]))
                cx, cy, bw, bh = (float(v) for v in parts[1:])
                x1 = int((cx - bw / 2) * self.w)
                y1 = int((cy - bh / 2) * self.h)
                x2 = int((cx + bw / 2) * self.w)
                y2 = int((cy + bh / 2) * self.h)
                self.boxes.append((x1, y1, x2, y2, cls))
        self.pending_cls = 0

    def save(self):
        p = self.files[self.idx]
        txt = p.with_suffix(".txt")
        lines = []
        for (x1, y1, x2, y2, cls) in self.boxes:
            x1, x2 = max(0, x1), min(self.w, x2)
            y1, y2 = max(0, y1), min(self.h, y2)
            if x2 <= x1 or y2 <= y1:
                continue
            cx = (x1 + x2) / 2 / self.w
            cy = (y1 + y2) / 2 / self.h
            bw = (x2 - x1) / self.w
            bh = (y2 - y1) / self.h
            lines.append(f"{cls} {cx:.4f} {cy:.4f} {bw:.4f} {bh:.4f}")
        txt.write_text("\n".join(lines), encoding="utf-8")

    def render(self):
        vis = self.img.copy()
        for (x1, y1, x2, y2, cls) in self.boxes:
            col = COLORS[cls % len(COLORS)]
            cv2.rectangle(vis, (x1, y1), (x2, y2), col, 2)
            cv2.putText(vis, f"{cls}:{CLASSES[cls]}", (x1, max(14, y1 - 4)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1)
        cv2.putText(vis, f"{self.idx+1}/{len(self.files)} {self.files[self.idx].name}",
                    (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
        cv2.putText(vis, f"next cls: {self.pending_cls}:{CLASSES[self.pending_cls]}",
                    (8, 46), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        cv2.putText(vis, "LMB drag=box  digits=cls  Enter=ok  BS=del  n/p=next  s=skip  q=quit",
                    (8, self.h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
        return vis

    def mouse(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drag_start = (x, y)
        elif event == cv2.EVENT_LBUTTONUP and self.drag_start:
            x1, y1 = self.drag_start
            x2, y2 = x, y
            if abs(x2 - x1) >= 4 and abs(y2 - y1) >= 4:
                self.boxes.append((min(x1, x2), min(y1, y2),
                                   max(x1, x2), max(y1, y2), self.pending_cls))
            self.drag_start = None
        elif event == cv2.EVENT_MOUSEMOVE and self.drag_start:
            vis = self.render()
            x1, y1 = self.drag_start
            cv2.rectangle(vis, (x1, y1), (x, y), (0, 255, 255), 1)
            cv2.imshow(WIN, vis)

    def run(self):
        cv2.namedWindow(WIN)
        cv2.setMouseCallback(WIN, self.mouse)
        while True:
            self.load()
            while True:
                cv2.imshow(WIN, self.render())
                k = cv2.waitKey(20) & 0xFF
                if k == ord("q"):
                    cv2.destroyAllWindows()
                    return False
                elif k == ord("n") or k == ord("N"):
                    self.save()
                    self.idx = min(self.idx + 1, len(self.files) - 1)
                    break
                elif k == ord("p") or k == ord("P"):
                    self.save()
                    self.idx = max(self.idx - 1, 0)
                    break
                elif k == ord("s") or k == ord("S"):
                    self.idx = min(self.idx + 1, len(self.files) - 1)
                    break
                elif k == 13:      # Enter: 确认框(保持类别递增切换)
                    self.pending_cls = min(self.pending_cls + 1, len(CLASSES) - 1)
                elif k == 8:       # Backspace: 删最后一个框
                    if self.boxes:
                        self.boxes.pop()
                elif 48 <= k <= 57:
                    self.pending_cls = k - 48
                elif k == ord(" ") and self.pending_cls + 10 < len(CLASSES):
                    self.pending_cls += 10
                elif k == ord("-") and self.pending_cls - 10 >= 0:
                    self.pending_cls -= 10
                elif k == 255:
                    pass
                # else ignore
            if not (0 <= self.idx < len(self.files)):
                break
        cv2.destroyAllWindows()
        return True


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("dir", nargs="?", default="data/labeling/in_mm")
    args = ap.parse_args()
    print("标注器启动, 目录:", args.dir)
    print("注意: 框=ROI 抠图训练用, 只需框住目标即可; 类别数字见类表")
    app = Annotator(args.dir)
    app.run()
    print("完成, 标注保存为同名 .txt (cls cx cy w h 归一化)")
