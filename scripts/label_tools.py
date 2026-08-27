# -*- coding: utf-8 -*-
"""强化训练素材准备 (v1): 从历史对局帧切分并生成标注骨架。

用法:
    python scripts/label_tools.py  \u2014prepare          # 切分素材 -> data/labeling/{in_mm,out_mm}/
    python scripts/label_tools.py  \u2014check <目录>      # 检查标注一致性

产出:
  data/labeling/in_mm/  小地图内裁剪(240x240)  -> 用户标注 红点蓝点绿点黄点/塔/野怪
  data/labeling/out_mm/ 全屏帧(1280x720)      -> 用户标注 敌英/队友/自/野怪/塔
  每张同名 .txt (YOLO 空骨架), 用户填: cls x_center y_center w h (归一化)

类别(与 yolo zhongkui_11cls 一致):
  0 enemy_hero 1 ally_hero 2 enemy_minion 3 ally_minion 4 enemy_turret
  5 ally_turret 6 enemy_crystal 7 ally_crystal 8 neutral_monster
  9 hook_aim 10 skill_effect 11 self
  小地图专用: 12 mm_red 13 mm_blue 14 mm_green 15 mm_yellow 16 mm_monster 17 mm_tower
"""
import argparse
import glob
import shutil
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MM_ROI = (0, 0, 232, 232)   # x0,y0,x1,y1 of minimap on 1280x720


def prepare(frames_glob="temp/self_check/live_*.png", keep=99999):
    """切分素材: 全屏 + 小地图裁剪, 每张配空 YOLO txt。"""
    out_in = ROOT / "data" / "labeling" / "in_mm"
    out_out = ROOT / "data" / "labeling" / "out_mm" / "images"
    out_out_lb = ROOT / "data" / "labeling" / "out_mm" / "labels"
    for d in (out_in, out_out, out_out_lb):
        d.mkdir(parents=True, exist_ok=True)
    # v1.1 素材来源: self_check(最新) + keep(早期局) 合并去重
    sources = []
    for g in (frames_glob, "temp/keep/live_*.png"):
        sources += glob.glob(str(ROOT / g))
    files = sorted(set(sources))[-keep:]
    n = 0
    for f in files:
        img = cv2.imread(f)
        if img is None:
            continue
        if img.shape[0] < 232 or img.shape[1] < 232:
            continue
        n += 1
        name = Path(f).stem
        # 全屏 (小地图外素材)
        cv2.imwrite(str(out_out / f"{name}.png"), img)
        (out_out_lb / f"{name}.txt").write_text("", encoding="utf-8")
        # 小地图内 (cv2: img[y0:y1, x0:x1])
        mm = img[MM_ROI[1]:MM_ROI[3], MM_ROI[0]:MM_ROI[2]]
        cv2.imwrite(str(out_in / f"{name}_mm.png"), mm)
        (out_in / f"{name}_mm.txt").write_text("", encoding="utf-8")
    print(f"素材就绪: {n} 帧 -> data/labeling/")
    print("  小地图内: data/labeling/in_mm/  (标 红点/蓝点/绿点/黄点/塔)")
    print("  小地图外: data/labeling/out_mm/images/ (标 en/ally/self/monster/turret)")
    print("  每张同名 .txt 为 YOLO 空骨架, 填: cls cx cy w h (归一化 0-1)")


def check(d):
    d = Path(d)
    ok = bad = 0
    for txt in sorted(d.glob("*.txt")):
        try:
            for ln in txt.read_text(encoding="utf-8").strip().splitlines():
                parts = ln.split()
                assert len(parts) == 5, f"{txt.name}: {ln}"
                cls = int(float(parts[0]))
                vals = [float(x) for x in parts[1:]]
                assert all(0.0 <= v <= 1.0 for v in vals), f"{txt.name}: 越界 {ln}"
                assert 0 <= cls <= 17, f"{txt.name}: cls 越界"
            ok += 1
        except Exception as e:
            bad += 1
            print("BAD:", e)
    print(f"检查: OK {ok} / BAD {bad}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--prepare", action="store_true")
    ap.add_argument("--check", metavar="DIR")
    args = ap.parse_args()
    if args.prepare:
        prepare()
    elif args.check:
        check(args.check)
    else:
        ap.print_help()
