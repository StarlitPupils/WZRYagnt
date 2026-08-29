# -*- coding: utf-8 -*-
"""AI 审核标注 (v11.2): 主动学习确认。
从 auto_labeled (或 labeling) 采样候选框 -> 生成 montage (每格: 抠图 + 模型预测标签)
-> 由 Assistant (我) 读图确认/纠正 -> 修正写回标签 (可靠数据才入训练)。

用法:
  python scripts/ai_verify_labels.py --src data/auto_labeled --out temp/audit --max 40
  读 temp/audit/montage.png -> 我确认 -> 在 temp/audit/fixes.json 写修正
  python scripts/ai_verify_labels.py --apply temp/audit/fixes.json
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CLS_NAME = {0: "enemy_hero", 1: "ally_hero", 8: "neutral_monster", 11: "self",
            12: "mm_red", 13: "mm_blue", 14: "mm_green", 15: "mm_yellow"}
NAME_CLS = {v: k for k, v in CLS_NAME.items()}


def build_audit(src_dirs, out_dir, max_items=40):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tiles = []
    meta = []   # (frame_path, gt_cls, box_px)
    for sd in src_dirs:
        d = Path(sd)
        if not d.exists():
            continue
        for imgf in sorted(d.glob("*.png")):
            txt = imgf.with_suffix(".txt")
            if not txt.exists():
                continue
            img = cv2.imread(str(imgf))
            if img is None:
                continue
            h, w = img.shape[:2]
            for ln in txt.read_text(encoding="utf-8").strip().splitlines():
                p = ln.split()
                if len(p) != 5:
                    continue
                cls = int(float(p[0]))
                cx, cy, bw, bh = (float(v) for v in p[1:])
                x1, y1 = int((cx - bw / 2) * w), max(0, int((cy - bh / 2) * h))
                x2, y2 = min(w, int((cx + bw / 2) * w)), min(h, int((cy + bh / 2) * h))
                roi = img[y1:y2, x1:x2]
                if roi.size == 0:
                    continue
                rh, rw = roi.shape[:2]
                cell = np.zeros((130, 130, 3), np.uint8)
                sc = min(118 / rw, 118 / rh)
                nw, nh = int(rw * sc), int(rh * sc)
                cell[6:6 + nh, 6:6 + nw] = cv2.resize(roi, (nw, nh))
                name = CLS_NAME.get(cls, f"cls{cls}")
                cv2.putText(cell, f"{name}", (8, 126), cv2.FONT_HERSHEY_SIMPLEX,
                            0.4, (0, 255, 255), 1)
                tiles.append(cell)
                meta.append({"img": str(imgf), "cls": name, "box": [x1, y1, x2, y2]})
                if len(tiles) >= max_items:
                    break
            if len(tiles) >= max_items:
                break
        if len(tiles) >= max_items:
            break
    if not tiles:
        print("无样本")
        return
    rows = []
    for i in range(0, len(tiles), 8):
        row = tiles[i:i + 8]
        while len(row) < 8:
            row.append(np.zeros((130, 130, 3), np.uint8))
        rows.append(np.hstack(row))
    mont = np.vstack(rows)
    montage_p = out_dir / "montage.png"
    cv2.imwrite(str(montage_p), mont)
    (out_dir / "candidates.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1),
                                             encoding="utf-8")
    print(f"审核 montage: {montage_p} ({len(tiles)} 候选)")
    print("请 AI/人 读图确认 -> 在 temp/audit/fixes.json 写 {'<img>': {'<box>_new_cls': ...}} 或直接改")


def apply_fixes(fixes_path):
    """按 fixes.json 修正标注 (img -> 新 cls 替换旧)。"""
    fixes = json.loads(Path(fixes_path).read_text(encoding="utf-8"))
    n = 0
    for img_p, new_cls in fixes.items():
        txt = Path(img_p).with_suffix(".txt")
        if not txt.exists():
            continue
        lines = []
        for ln in txt.read_text(encoding="utf-8").strip().splitlines():
            p = ln.split()
            if len(p) == 5:
                lines.append(f"{NAME_CLS[new_cls]} {' '.join(p[1:])}")
        txt.write_text("\n".join(lines), encoding="utf-8")
        n += 1
    print(f"应用修正: {n} 标签")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", nargs="+", default=["data/auto_labeled"])
    ap.add_argument("--out", default="temp/audit")
    ap.add_argument("--max", type=int, default=40)
    ap.add_argument("--apply", metavar="FIXES")
    args = ap.parse_args()
    if args.apply:
        apply_fixes(args.apply)
    else:
        build_audit(args.src, args.out, args.max)
