# -*- coding: utf-8 -*-
"""Scan zhongkui dataset labels (train + val) and report class statistics.

Counts boxes per class, average boxes per image, and images without any
annotation (empty txt). Prints to console and writes
data/yolo_dataset/zhongkui/class_stats.txt (utf-8).
"""
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]  # E:/WZRYagent
LABELS_DIR = ROOT / "data" / "yolo_dataset" / "zhongkui" / "labels"
OUT_TXT = ROOT / "data" / "yolo_dataset" / "zhongkui" / "class_stats.txt"
CLASSES_JSON = ROOT / "configs" / "classes.json"


def main():
    names = []
    if CLASSES_JSON.exists():
        try:
            names = json.loads(CLASSES_JSON.read_text(encoding="utf-8-sig"))  # file has BOM
        except (json.JSONDecodeError, OSError) as exc:
            print(f"[warn] could not read {CLASSES_JSON}: {exc}")

    # The zhongkui labels are currently in the 4-class model space (ids 0-3,
    # see data/yolo_dataset/zhongkui/data.yaml). Only fall back to the 11-class
    # names when an id beyond 3 appears (i.e. data was converted to 11-class).
    MODEL4_NAMES = {0: 'enemy_hero', 1: 'hook_aim', 2: 'minion', 3: 'turret'}

    splits = ["train", "val"]
    per_class = Counter()
    total_boxes = 0
    total_imgs = 0
    empty_imgs = 0
    split_stats = {}

    for split in splits:
        split_dir = LABELS_DIR / split
        if not split_dir.is_dir():
            print(f"[warn] missing label dir: {split_dir}")
            continue
        txts = sorted(split_dir.glob("*.txt"))
        n_imgs = len(txts)
        n_boxes = 0
        n_empty = 0
        for t in txts:
            lines = [ln.strip() for ln in t.read_text(encoding="utf-8").splitlines() if ln.strip()]
            if not lines:
                n_empty += 1
                continue
            for ln in lines:
                parts = ln.split()
                if not parts:
                    continue
                try:
                    cid = int(float(parts[0]))
                except ValueError:
                    continue
                per_class[cid] += 1
                n_boxes += 1
        total_imgs += n_imgs
        total_boxes += n_boxes
        empty_imgs += n_empty
        split_stats[split] = (n_imgs, n_boxes, n_empty)

    lines = []

    def emit(s):
        lines.append(s)
        print(s)

    emit("Zhongkui dataset class statistics")
    emit(f"Total labeled images (txt files): {total_imgs}")
    for split in splits:
        if split in split_stats:
            n_imgs, n_boxes, n_empty = split_stats[split]
            emit(f"  [{split}] images={n_imgs} boxes={n_boxes} empty={n_empty}")
    emit(f"Total boxes: {total_boxes}")
    if total_imgs:
        emit(f"Avg boxes per image: {total_boxes / total_imgs:.2f}")
    emit(f"Images without annotations (empty txt): {empty_imgs}")
    emit("")
    max_id = max(per_class) if per_class else -1
    if names and max_id > 3:
        emit("Naming: 11-class system (configs/classes.json)")
        label_of = lambda cid: names[cid] if cid < len(names) else f"unknown_{cid}"
    else:
        emit("Naming: 4-class model space (data.yaml); labels not yet converted to 11-class")
        label_of = lambda cid: MODEL4_NAMES.get(cid, f"unknown_{cid}")
    emit("Per-class box counts:")
    for cid in sorted(per_class):
        emit(f"  {cid:>2} {label_of(cid):<16} {per_class[cid]}")

    OUT_TXT.parent.mkdir(parents=True, exist_ok=True)
    OUT_TXT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nWritten to {OUT_TXT}")


if __name__ == "__main__":
    main()
