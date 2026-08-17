# -*- coding: utf-8 -*-
"""录像抽帧数据扩充：从对局录像均匀抽帧，用现有检测器预标注，扩充训练数据。

产出：
  data/screenshots/replay/*.jpg  抽帧图
  data/screenshots/replay/*.txt  预标注（11 类 id：enemy_hero->0, hook_aim->9,
                                 minion->2(enemy_minion), turret->4(enemy_turret)，
                                 阵营歧义交给 camp_autolabel.py 处理）

用法：
    venv\\Scripts\\python.exe scripts\\train\\extract_replay_frames.py
        [--video temp/tmphjhl7fk3.mp4] [--count 200] [--skip-first 600] [--conf 0.35]
"""
import argparse
import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

# 模型 4 类 -> 11 类 id（minion/turret 默认敌方，待 camp_autolabel 复核）
MODEL_NAMES = ["enemy_hero", "hook_aim", "minion", "turret"]
NAME_TO_ID = {"enemy_hero": 0, "hook_aim": 9, "minion": 2, "turret": 4}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", default=str(ROOT / "temp" / "tmphjhl7fk3.mp4"))
    ap.add_argument("--count", type=int, default=200)
    ap.add_argument("--skip-first", type=int, default=600, help="跳过开头帧（加载/选人）")
    ap.add_argument("--conf", type=float, default=0.35)
    ap.add_argument("--out", default=str(ROOT / "data" / "screenshots" / "replay"))
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"无法打开录像: {args.video}")
        return 1
    n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    n_use = max(1, n_total - args.skip_first)
    stride = max(1, n_use // args.count)
    frames = [args.skip_first + i * stride for i in range(min(args.count, n_use))]
    print(f"录像 {n_total} 帧，抽 {len(frames)} 帧（stride={stride}）")

    from wzry.vision.detector import YoloDetector
    det = YoloDetector(ROOT / "runs" / "detect" / "zhongkui_detector_finetune" / "weights" / "best.pt",
                       conf=args.conf)

    n_boxes = 0
    for i, pos in enumerate(frames):
        cap.set(cv2.CAP_PROP_POS_FRAMES, pos)
        ok, frame = cap.read()
        if not ok:
            continue
        name = f"replay_{pos:06d}"
        cv2.imwrite(str(out_dir / f"{name}.jpg"), frame)
        dets = det.detect(frame)
        if dets:
            h, w = frame.shape[:2]
            lines = []
            for d in dets:
                cid = NAME_TO_ID.get(d.cls)
                if cid is None:
                    continue
                x1, y1, x2, y2 = d.xyxy
                xc = ((x1 + x2) / 2) / w
                yc = ((y1 + y2) / 2) / h
                bw = (x2 - x1) / w
                bh = (y2 - y1) / h
                lines.append(f"{cid} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}")
            if lines:
                (out_dir / f"{name}.txt").write_text("\n".join(lines), encoding="utf-8")
                n_boxes += len(lines)
        if (i + 1) % 50 == 0:
            print(f"  进度 {i + 1}/{len(frames)}")
    cap.release()
    print(f"完成: {len(frames)} 帧 -> {out_dir}")
    print(f"预标注框数: {n_boxes}（minion/turret 阵营可用 camp_autolabel.py 复核）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
