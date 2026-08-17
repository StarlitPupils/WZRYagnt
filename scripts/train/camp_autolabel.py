# -*- coding: utf-8 -*-
"""阵营自动分类工具：按单位头顶血条颜色（绿=友方/红=敌方）自动修正
minion(2)/turret(4) 的阵营歧义（2->3 ally_minion, 4->5 ally_turret）。

用法：
    venv\\Scripts\\python.exe scripts\\train\\camp_autolabel.py [--apply] [--camp blue|red]
        [--min-conf 0.30] [--out-dir temp/camp_check]

  --apply     写入修正（先备份 *.bak）
  --camp      玩家阵营（默认 blue；玩红方时判定反转）
  --min-conf  绿/红占比差的置信阈值（低于此值不动，留人工）
  --out-dir   保存血条区域裁剪图（人工抽查用，拼成图集）

原理：王者荣耀单位头顶有血条，友方绿色、敌方红色（以玩家阵营为准）。
血条 ROI = 检测框顶部上方一小条区域。
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

IMG_DIR = ROOT / "data" / "screenshots" / "zhongkui"
AMBIGUOUS = {2: "enemy_minion", 3: "ally_minion", 4: "enemy_turret", 5: "ally_turret"}


def bar_colors(roi_bgr):
    """血条 ROI 的绿/红像素占比（归一化 0-1）。"""
    if roi_bgr is None or roi_bgr.size == 0:
        return 0.0, 0.0
    hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)
    h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    green = ((h >= 40) & (h <= 85) & (s >= 80) & (v >= 80)).mean()
    red = (((h <= 10) | (h >= 170)) & (s >= 80) & (v >= 80)).mean()
    return float(green), float(red)


def bar_roi(frame, box, h_frac=0.12, w_frac=0.70, dy_frac=0.02):
    """检测框顶部上方的血条区域。box=(x1,y1,x2,y2)。"""
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = (int(v) for v in box)
    bw, bh = x2 - x1, y2 - y1
    cx = (x1 + x2) / 2
    rx0 = max(0, int(cx - bw * w_frac / 2))
    rx1 = min(w, int(cx + bw * w_frac / 2))
    ry0 = max(0, int(y1 - bh * dy_frac))
    ry1 = max(0, int(y1 - bh * dy_frac + bh * h_frac))
    if ry1 <= ry0 or rx1 <= rx0:
        return None
    return frame[ry0:ry1, rx0:rx1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--camp", default="blue", choices=["blue", "red"])
    ap.add_argument("--min-conf", type=float, default=0.30)
    ap.add_argument("--out-dir", default=str(ROOT / "temp" / "camp_check"))
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    n_total = n_sugg = n_apply = n_uncertain = 0
    img_files = sorted(IMG_DIR.glob("*.jpg")) + sorted(IMG_DIR.glob("*.png"))
    for img_path in img_files:
        txt = img_path.with_suffix(".txt")
        if not txt.exists():
            continue
        frame = cv2.imread(str(img_path))
        if frame is None:
            continue
        h, w = frame.shape[:2]
        lines = txt.read_text(encoding="utf-8").strip().splitlines()
        new_lines = []
        changed = False
        for ln in lines:
            p = ln.split()
            if len(p) != 5:
                new_lines.append(ln)
                continue
            cid = int(p[0])
            if cid not in (2, 4):
                new_lines.append(ln)
                continue
            xc, yc, bw, bh = (float(v) for v in p[1:])
            box = ((xc - bw / 2) * w, (yc - bh / 2) * h,
                   (xc + bw / 2) * w, (yc + bh / 2) * h)
            roi = bar_roi(frame, box)
            green, red = bar_colors(roi)
            diff = green - red
            n_total += 1
            sugg = None
            lean = ""
            if args.camp == "red":
                diff = -diff
            if diff >= args.min_conf:
                sugg = cid + 1          # 2->3, 4->5（友方，高置信）
            elif diff <= -args.min_conf:
                sugg = cid              # 保持敌方（高置信）
            elif diff >= 0.05:
                lean = "偏友方"         # 弱倾向：仅供人工参考，不自动改
            elif diff <= -0.05:
                lean = "偏敌方"
            else:
                lean = "无法判定"
            if sugg is None:
                n_uncertain += 1
            elif sugg != cid:
                n_sugg += 1
            rows.append([img_path.name, cid, AMBIGUOUS.get(cid, cid),
                         sugg if sugg else cid, sugg if sugg else cid,
                         round(green, 3), round(red, 3),
                         ("改" if sugg and sugg != cid else "保持") if sugg else lean])
            # 保存裁剪图（供抽查）
            if roi is not None:
                crop = cv2.resize(roi, (120, 30))
                cv2.imwrite(str(out_dir / f"{img_path.stem}_{cid}_{n_total:04d}.png"), crop)
            if sugg is not None and sugg != cid:
                new_lines.append(f"{sugg} {' '.join(p[1:])}")
                changed = True
            else:
                new_lines.append(ln)
        if args.apply and changed:
            bak = txt.with_suffix(".txt.bak")
            if not bak.exists():
                bak.write_text(txt.read_text(encoding="utf-8"), encoding="utf-8")
            txt.write_text("\n".join(new_lines) + "\n", encoding="utf-8")
            n_apply += 1

    print(f"歧义框总数: {n_total}")
    print(f"建议改为友方(2->3/4->5): {n_sugg}")
    print(f"不确定(留人工): {n_uncertain}")
    if args.apply:
        print(f"已修改标签文件: {n_apply}（原文件备份为 .txt.bak）")
    print(f"血条裁剪图已保存到: {out_dir}（可快速拼图抽查准确率）")
    report = out_dir / "camp_suggestions.csv"
    with open(report, "w", newline="", encoding="utf-8-sig") as f:
        cw = csv.writer(f)
        cw.writerow(["file", "old_id", "old_class", "new_id", "new_class",
                     "green", "red", "verdict"])
        cw.writerows(rows)
    print(f"明细: {report}")
    print("\n建议：先用不带 --apply 跑一遍，抽查 temp/camp_check 下的裁剪图；")
    print("准确率满意后加 --apply 写入。玩家为红方时用 --camp red。")


if __name__ == "__main__":
    main()
