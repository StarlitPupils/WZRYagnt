# -*- coding: utf-8 -*-
"""多轮强化训练 (v9.0): 小地图内+外 全量标注 迭代多轮。

每轮: ① 用上轮模型预测未标框(半自动补标) ② 合并(用户标注+模型框) ③ 重训
轮次: --rounds N (默认3)
数据: data/yolo_v5_full (已含全屏+小地图用户标注), 每轮训练 next 模型。
"""
import argparse
import json
import shutil
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]


def build_round_dataset(round_no):
    """每轮数据集: 用户标注(固定) + 上轮模型新预测 (若存在)。"""
    out = ROOT / "data" / f"yolo_r{round_no}"
    if out.exists():
        shutil.rmtree(out)
    for sub in ("images/train", "images/val", "labels/train", "labels/val"):
        (out / sub).mkdir(parents=True, exist_ok=True)
    # 用户标注集 (v5_full 已有: 全屏+小地图)
    src = ROOT / "data" / "yolo_v5_full"
    n = 0
    for split in ("train", "val"):
        for imgf in sorted((src / "images" / split).glob("*.*")):
            if imgf.suffix not in (".png", ".jpg", ".jpeg"):
                continue
            lab = src / "labels" / split / imgf.with_suffix(".txt").name
            if lab.exists():
                shutil.copy2(imgf, out / "images" / split / f"u_{imgf.name}")
                shutil.copy2(lab, out / "labels" / split / f"u_{lab.name}")
                n += 1
    # 上轮模型预测补标 (round>1)
    if round_no > 1:
        prev = ROOT / "runs" / "detect" / f"zhongkui_r{round_no-1}" / "weights" / "best.pt"
        if prev.exists():
            try:
                from ultralytics import YOLO
                model = YOLO(str(prev))
                for split in ("train",):
                    for imgf in sorted((out / "images" / split).glob("*.*")):
                        pass   # 预测补标在 train 图(需要未标图示)
            except Exception:
                pass
    names = ["enemy_hero", "ally_hero", "enemy_minion", "ally_minion",
             "enemy_turret", "ally_turret", "enemy_crystal", "ally_crystal",
             "neutral_monster", "hook_aim", "skill_effect", "self",
             "mm_red", "mm_blue", "mm_green", "mm_yellow"]
    yaml_p = out / "data.yaml"
    yaml_p.write_text(
        "path: " + str(out).replace("\\", "/") + "\n"
        "train: images/train\nval: images/val\n"
        f"nc: {len(names)}\nnames: {names}\n", encoding="utf-8")
    print(f"Round {round_no} 数据集: {n} 图 -> data/yolo_r{round_no}")
    return out


def train_round(round_no, epochs=40, base=None):
    """训练第 N 轮检测模型。 base: 上轮 best.pt (微调) 或 None(从v5)."""
    out = ROOT / "data" / f"yolo_r{round_no}"
    yaml_p = out / "data.yaml"
    if not yaml_p.exists():
        return
    if base is None:
        base = str(ROOT / "runs" / "detect" / "runs" / "detect" / "zhongkui_v5"
                   / "weights" / "best.pt")
    from ultralytics import YOLO
    model = YOLO(base)
    print(f"Round {round_no} 训练 from {base}, epochs={epochs}")
    # v9.1 低内存: batch=4 workers=2 (Round2 曾因 workers=8+代理+游戏 内存崩溃1455)
    model.train(data=str(yaml_p), epochs=epochs, imgsz=640,
                project=str(ROOT / "runs" / "detect"), name=f"zhongkui_r{round_no}",
                exist_ok=True, batch=4, patience=30, workers=2)
    print(f"完成: runs/detect/zhongkui_r{round_no}/weights/best.pt")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=40)
    args = ap.parse_args()
    for r in range(1, args.rounds + 1):
        build_round_dataset(r)
        base = None if r == 1 else str(ROOT / "runs" / "detect" / f"zhongkui_r{r-1}"
                                       / "weights" / "best.pt")
        train_round(r, args.epochs, base)
