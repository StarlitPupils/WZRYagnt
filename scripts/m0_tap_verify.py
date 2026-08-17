# -*- coding: utf-8 -*-
"""设备像素空间点击验证（训练营内安全）。

对校准绝对坐标点逐个执行：截图 -> tap -> 截图，比较按钮 ROI 像素变化，
验证"校准点是否真的落在按钮上、点击是否生效"。

用法：
    venv\\Scripts\\python.exe scripts\\m0_tap_verify.py [--point skill1] [--all]
"""
import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from wzry.capture.adb import AdbCapture  # noqa: E402
from wzry.control.executor import AdbExecutor  # noqa: E402


def roi_change(before, after, center, r=45):
    h, w = before.shape[:2]
    cx, cy = int(center[0]), int(center[1])
    x0, x1 = max(0, cx - r), min(w, cx + r)
    y0, y1 = max(0, cy - r), min(h, cy + r)
    if x1 <= x0 or y1 <= y0:
        return 0.0
    return float(np.abs(before[y0:y1, x0:x1].astype(np.int16)
                        - after[y0:y1, x0:x1].astype(np.int16)).mean())


def load_absolute_calib():
    import json
    with open(ROOT / "configs" / "calibration_absolute.json", "r", encoding="utf-8") as f:
        data = json.load(f)
    return data["points"], data["screen_width"], data["screen_height"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--point", default=None, help="验证单个点（如 skill1）")
    ap.add_argument("--all", action="store_true", help="验证全部校准点")
    ap.add_argument("--rounds", type=int, default=2, help="每点点击次数")
    args = ap.parse_args()

    pts, cw, ch = load_absolute_calib()
    cap = AdbCapture()
    ex = AdbExecutor(serial=cap.serial)

    keys = [args.point] if args.point else (list(pts) if args.all else ["skill1", "skill2", "skill3", "attack", "move_stick_center"])

    print(f"设备: {cap.serial}  校准画面: {cw}x{ch}")
    for key in keys:
        if key not in pts:
            print(f"  跳过未知点: {key}")
            continue
        px, py = pts[key]
        changes = []
        ok_all = True
        for _ in range(args.rounds):
            before, _ = cap.get_frame()
            if before is None:
                print(f"  {key:<18} 截图失败")
                ok_all = False
                break
            ok = ex.tap(px, py, source="tap_verify")
            time.sleep(0.35)
            after, _ = cap.get_frame()
            if after is None:
                ok_all = False
                break
            changes.append(roi_change(before, after, (px, py)))
        if changes:
            avg = sum(changes) / len(changes)
            verdict = "OK" if avg > 2.0 else ("可疑" if avg > 0.8 else "无反馈")
            print(f"  {key:<18} ({px:>4},{py:>4}) ROI变化 {avg:6.2f}  [{verdict}]")
        elif not ok_all:
            print(f"  {key:<18} 验证失败")

    ex.close()


if __name__ == "__main__":
    main()
