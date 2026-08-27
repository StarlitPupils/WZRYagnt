# -*- coding: utf-8 -*-
"""构建 v4 微调数据集: v3 train + 用户真值12帧(仅GT标签,前缀gt_) 全部入 train。"""
import sys
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

V3 = ROOT / "data" / "yolo_dataset" / "zhongkui_v3"
GT = ROOT / "temp" / "ann"
DST = ROOT / "data" / "yolo_dataset" / "zhongkui_v4"
for sub in ("images/train", "images/val", "labels/train", "labels/val"):
    (DST / sub).mkdir(parents=True, exist_ok=True)

# v3 train 全部拷贝
n = 0
for lp in sorted((V3 / "labels" / "train").glob("*.txt")):
    ip = V3 / "images" / "train" / (lp.stem + ".png")
    if not ip.exists():
        continue
    shutil.copy2(str(ip), str(DST / "images" / "train" / f"{lp.stem}.png"))
    shutil.copy2(str(lp), str(DST / "labels" / "train" / f"{lp.stem}.txt"))
    n += 1
# 真值12帧 → train (仅类0-10)
for i in range(1, 13):
    ip = GT / f"s{i:02d}.png"
    tp = GT / f"s{i:02d}.txt"
    if not ip.exists():
        continue
    lines = [ln for ln in tp.read_text(encoding="utf-8").splitlines()
             if len(ln.split()) == 5 and int(ln.split()[0]) <= 10]
    shutil.copy2(str(ip), str(DST / "images" / "train" / f"gt_s{i:02d}.png"))
    (DST / "labels" / "train" / f"gt_s{i:02d}.txt").write_text("\n".join(lines), encoding="utf-8")
    n += 1
# val 用 v3 val (s01-s03 标签只作参考; 微调不看 val 权重)
for lp in sorted((V3 / "labels" / "val").glob("*.txt")):
    ip = V3 / "images" / "val" / (lp.stem + ".png")
    if ip.exists():
        shutil.copy2(str(ip), str(DST / "images" / "val" / f"{lp.stem}.png"))
        shutil.copy2(str(lp), str(DST / "labels" / "val" / f"{lp.stem}.txt"))
yaml = f"""# zhongkui_v4 微调集
path: {str(DST).replace(chr(92), '/')}
train: images/train
val: images/val
nc: 11
names: ["enemy_hero", "ally_hero", "enemy_minion", "ally_minion", "enemy_turret", "ally_turret", "enemy_crystal", "ally_crystal", "neutral_monster", "hook_aim", "skill_effect"]
"""
(DST / "data.yaml").write_text(yaml, encoding="utf-8")
print("帧数:", n)
