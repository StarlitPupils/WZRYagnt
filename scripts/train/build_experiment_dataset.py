# -*- coding: utf-8 -*-
"""实验数据集构建：复制 yolo_dataset -> zhongkui_v2，并把 camp_autolabel 的
高置信建议（CSV verdict=改）应用到副本标签（不修改用户主数据）。

用法：
    venv\\Scripts\\python.exe scripts\\train\\build_experiment_dataset.py
        [--csv temp/camp_check_hero/camp_suggestions.csv] [--name zhongkui_v2]
"""
import argparse
import csv
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

SRC = ROOT / "data" / "yolo_dataset" / "zhongkui"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=str(ROOT / "temp" / "camp_check_hero" / "camp_suggestions.csv"))
    ap.add_argument("--name", default="zhongkui_v2")
    args = ap.parse_args()

    dst = ROOT / "data" / "yolo_dataset" / args.name
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(SRC, dst)

    # 读建议（只取 verdict=改；按 文件名+旧类+中心坐标 精确匹配，避免同文件多框误改/漏改）
    changes = {}  # (stem, old_id, xc, yc) -> new_id
    with open(args.csv, encoding="utf-8-sig") as f:
        for row in csv.reader(f):
            if not row or row[0] == "file" or len(row) < 10:
                continue
            if row[7] == "改":
                changes[(Path(row[0]).stem, int(row[1]),
                         float(row[8]), float(row[9]))] = int(row[3])

    # 应用：labels/{train,val}/*.txt 按 (文件名, 旧id, 中心坐标容差) 匹配改类
    # 标签从 screenshots 原样拷贝，坐标应完全一致；容差只防浮点舍入差异
    TOL = 0.002
    n_applied = 0
    for txt in sorted((dst / "labels").rglob("*.txt")):
        lines = txt.read_text(encoding="utf-8").strip().splitlines()
        new_lines = []
        for ln in lines:
            p = ln.split()
            key = None
            if len(p) == 5:
                key = (txt.stem, int(p[0]), float(p[1]), float(p[2]))
                # 容差匹配：找 changes 中最近的同文件同旧类建议
                cand = None
                for (s, oid, xc, yc), nid in changes.items():
                    if s == key[0] and oid == key[1] and abs(xc - key[2]) <= TOL and abs(yc - key[3]) <= TOL:
                        cand = nid
                        break
            if cand is not None:
                new_lines.append(f"{cand} {' '.join(p[1:])}")
                n_applied += 1
            else:
                new_lines.append(ln)
        txt.write_text("\n".join(new_lines) + ("\n" if new_lines else ""), encoding="utf-8")

    # data.yaml（11 类）
    with open(ROOT / "configs" / "classes.json", "r", encoding="utf-8-sig") as f:
        classes = json.load(f)
    (dst / "data.yaml").write_text(
        f"# {args.name} experiment dataset (11-class, camp suggestions applied)\n"
        f"path: {dst.as_posix()}\n"
        f"train: images/train\nval: images/val\nnc: 11\nnames: {json.dumps(classes, ensure_ascii=False)}\n",
        encoding="utf-8")

    print(f"实验数据集: {dst}")
    print(f"应用建议: {n_applied} 条（原数据未动）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
