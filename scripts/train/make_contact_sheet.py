# -*- coding: utf-8 -*-
"""联系表生成器：把 camp_autolabel 的"建议修改"裁剪图拼成一张大图，供快速人工抽查。

用法：
    venv\\Scripts\\python.exe scripts\\train\\make_contact_sheet.py
        [--csv temp/camp_check/camp_suggestions.csv] [--img-dir temp/camp_check]
        [--cols 12] [--out temp/camp_check/contact_sheet.png]

输出：拼图（每格 120x30 血条裁剪图 + 文件名/新类标注），打开一张图即可抽查全部建议。
"""
import argparse
import csv
import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=str(ROOT / "temp" / "camp_check" / "camp_suggestions.csv"))
    ap.add_argument("--img-dir", default=str(ROOT / "temp" / "camp_check"))
    ap.add_argument("--cols", type=int, default=12)
    ap.add_argument("--out", default=str(ROOT / "temp" / "camp_check" / "contact_sheet.png"))
    args = ap.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        print(f"CSV 不存在: {csv_path}（先跑 camp_autolabel.py）")
        return 1
    img_dir = Path(args.img_dir)

    # 收集"建议修改"的行（verdict == 改）
    items = []  # (filename_prefix, new_class_name, crop_path)
    with open(csv_path, encoding="utf-8-sig") as f:
        for row in csv.reader(f):
            if not row or row[0] == "file":
                continue
            if len(row) < 8 or row[7] != "改":
                continue
            fname, old_id, old_cls, new_id, new_cls = row[0], int(row[1]), row[2], row[3], row[4]
            # 裁剪图命名: {stem}_{old_id}_{seq:04d}.png，需与 CSV 顺序对应（同目录同序）
            items.append((fname, new_cls))

    if not items:
        print("没有建议修改的条目")
        return 0

    # 按 CSV 顺序匹配裁剪图（命名规律一致）
    crops = sorted(img_dir.glob("*.png"))
    if len(crops) < len(items):
        print(f"裁剪图不足: {len(crops)} 张 < 建议 {len(items)} 条（换 --img-dir 指向对应目录）")
        # 退回：尽力按前缀匹配
        crops = []
        for fname, _ in items:
            cands = list(img_dir.glob(f"{Path(fname).stem}_*.png"))
            if cands:
                crops.append(cands[0])
    else:
        crops = crops[:len(items)]

    cell_w, cell_h, pad = 130, 46, 4
    cols = args.cols
    rows = (len(items) + cols - 1) // cols
    sheet = np_white = None
    import numpy as np
    sheet = np.full((rows * cell_h + pad, cols * cell_w + pad, 3), 240, dtype=np.uint8)
    for i, ((fname, new_cls), crop_path) in enumerate(zip(items, crops)):
        r, c = divmod(i, cols)
        x0, y0 = c * cell_w + pad, r * cell_h + pad
        cell = sheet[y0:y0 + cell_h, x0:x0 + cell_w].copy()
        if crop_path and Path(crop_path).exists():
            img = cv2.imread(str(crop_path))
            if img is not None:
                cell[2:32, 2:122] = cv2.resize(img, (120, 30))
        cv2.putText(cell, f"{Path(fname).stem[:18]} -> {new_cls}",
                    (2, cell_h - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.32, (0, 0, 0), 1)
        sheet[y0:y0 + cell_h, x0:x0 + cell_w] = cell

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), sheet)
    print(f"联系表已生成: {out}  ({len(items)} 条建议, {cols} 列 {rows} 行)")
    print("抽查方法：打开大图，逐格看血条颜色是否与 '-> 新类' 一致；")
    print("若多数正确即可运行 camp_autolabel.py --apply 落盘。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
