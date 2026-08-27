# -*- coding: utf-8 -*-
"""构建 v6 数据集: v4(11类) + 用户手工标注素材(含新类11=自己, 12类)。"""
import shutil
from pathlib import Path
import random

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "data" / "yolo_dataset" / "zhongkui_v4"
DST = ROOT / "data" / "yolo_dataset" / "zhongkui_v6"
IMG = ROOT / "temp" / "label_me"

NAMES12 = ["enemy_hero", "ally_hero", "enemy_minion", "ally_minion",
           "enemy_turret", "ally_turret", "enemy_crystal", "ally_crystal",
           "neutral_monster", "hook_aim", "skill_effect", "self"]

def main():
    if DST.exists():
        shutil.rmtree(DST)
    for sub in ("images/train", "images/val", "labels/train", "labels/val"):
        (DST / sub).mkdir(parents=True, exist_ok=True)
    # 1) 复制 v4 全部
    for p in SRC.rglob("*"):
        if p.is_file():
            rel = p.relative_to(SRC)
            dst = DST / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(p, dst)
    # 2) 添加手动标注素材 (train, 非拆分: 新素材直接进 train)
    n_add = 0
    for t in sorted(IMG.glob("*.txt")):
        lines = [l for l in t.read_text(encoding="utf-8").splitlines() if l.strip()]
        if not lines:
            continue
        img = IMG / (t.stem + ".png")
        if not img.exists():
            continue
        shutil.copy2(img, DST / "images" / "train" / (t.stem + ".png"))
        with open(DST / "labels" / "train" / (t.stem + ".txt"), "w",
                  encoding="utf-8") as f:
            for l in lines:
                f.write(l + "\n")
        n_add += 1
    # 3) yaml (12类) -> 同时覆盖 DST/data.yaml (训练脚本读该文件)
    yaml_path = ROOT / "data" / "zhongkui_v6.yaml"
    yaml_text = (
        f"path: {DST.as_posix()}\ntrain: images/train\nval: images/val\n"
        f"nc: 12\nnames: {NAMES12}\n")
    yaml_path.write_text(yaml_text, encoding="utf-8")
    (DST / "data.yaml").write_text(yaml_text, encoding="utf-8")
    # 4) 统计
    n_train = len(list((DST / "images" / "train").glob("*.png")))
    n_val = len(list((DST / "images" / "val").glob("*.png")))
    print(f"v6 数据集: train {n_train} / val {n_val} (新增标注素材 {n_add} 张)")
    print("yaml:", yaml_path)


if __name__ == "__main__":
    main()
