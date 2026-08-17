# -*- coding: utf-8 -*-
"""标签迁移：4 类 id（enemy_hero/hook_aim/minion/turret）-> 11 类 id（classes.json）。

背景：
  - 旧标注（data/screenshots/zhongkui/*.txt 与 data/yolo_dataset/zhongkui/labels/**）是
    训练 best.pt 时的 4 类 id：0=enemy_hero, 1=hook_aim, 2=minion, 3=turret。
  - 目标体系是 classes.json 的 11 类。minion/turret 缺少阵营信息：
    默认映射到敌方（enemy_minion=2 / enemy_turret=4），并把含歧义框的文件
    写入审查清单，供人工用 annotate.py 复核修正（ally 需改为 3/5）。
  - 已迁移的文件（存在 >3 的 id）自动跳过，不会二次映射。

用法：
    venv\\Scripts\\python.exe scripts\\train\\migrate_labels_11class.py [--backup] [--apply]
      --apply   实际写入（默认只预览统计）
      --backup  写入前把原文件复制为 *.4class.bak
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

OLD_NAMES = {0: "enemy_hero", 1: "hook_aim", 2: "minion", 3: "turret"}
DEFAULT_MAP = {0: 0, 1: 9, 2: 2, 3: 4}   # minion->enemy_minion, turret->enemy_turret
AMBIGUOUS = {2, 3}                       # 旧 id 中阵营不明确的类

TARGET_DIRS = [
    ROOT / "data" / "screenshots" / "zhongkui",
    ROOT / "data" / "yolo_dataset" / "zhongkui" / "labels",
]


def load_classes():
    with open(ROOT / "configs" / "classes.json", "r", encoding="utf-8-sig") as f:
        return json.load(f)


def iter_label_files():
    seen = set()
    for d in TARGET_DIRS:
        if not d.exists():
            continue
        for txt in d.rglob("*.txt"):
            if txt.name in ("data.yaml",):
                continue
            if txt.resolve() in seen:
                continue
            seen.add(txt.resolve())
            yield txt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true", help="实际写入（默认仅预览）")
    ap.add_argument("--backup", action="store_true", help="写入前备份为 *.4class.bak")
    args = ap.parse_args()

    classes = load_classes()
    n_files = n_boxes = n_skipped = n_ambiguous = 0
    stats = {}
    ambiguous_files = []

    for txt in iter_label_files():
        try:
            lines = txt.read_text(encoding="utf-8").strip().splitlines()
        except Exception:
            continue
        if not lines or all(not l.strip() for l in lines):
            continue
        parsed = []
        already = False
        for ln in lines:
            p = ln.split()
            if len(p) != 5:
                continue
            cid = int(p[0])
            if cid > 3:
                already = True
            parsed.append((cid, p[1:]))
        if already:
            n_skipped += 1
            continue

        n_files += 1
        has_amb = any(cid in AMBIGUOUS for cid, _ in parsed)
        if has_amb:
            n_ambiguous += 1
            ambiguous_files.append(str(txt.relative_to(ROOT)))

        if args.apply:
            if args.backup:
                bak = txt.with_suffix(".4class.bak")
                if not bak.exists():
                    bak.write_text(txt.read_text(encoding="utf-8"), encoding="utf-8")
            new_lines = []
            for cid, rest in parsed:
                new_id = DEFAULT_MAP.get(cid, cid)
                new_lines.append(f"{new_id} {' '.join(rest)}")
                stats[new_id] = stats.get(new_id, 0) + 1
                n_boxes += 1
            txt.write_text("\n".join(new_lines) + ("\n" if new_lines else ""), encoding="utf-8")
        else:
            for cid, _ in parsed:
                stats[DEFAULT_MAP.get(cid, cid)] = stats.get(DEFAULT_MAP.get(cid, cid), 0) + 1
                n_boxes += 1

    print(f"迁移目标文件: {n_files} 个, 框总数: {n_boxes}, 已迁移跳过: {n_skipped} 个")
    print("迁移后各类框数（11 类 id -> 类名）:")
    for cid in range(len(classes)):
        if cid in stats:
            print(f"  {cid:>2} {classes[cid]:<16} {stats[cid]}")
    print(f"\n含歧义框（minion/turret，需人工复核阵营）的文件: {n_ambiguous} 个")

    if args.apply:
        out = ROOT / "data" / "yolo_dataset" / "zhongkui" / "AMBIGUOUS_REVIEW.txt"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text("\n".join(ambiguous_files) + "\n", encoding="utf-8")
        print(f"审查清单已写入: {out.relative_to(ROOT)}")
    else:
        print("\n（预览模式，未写入。加 --apply 实际执行，--backup 同时备份原文件）")


if __name__ == "__main__":
    main()
