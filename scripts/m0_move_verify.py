# -*- coding: utf-8 -*-
"""移动验证（训练营内安全）：swipe 摇杆 -> 检测小地图己方蓝点质心位移。

用法：
    venv\\Scripts\\python.exe scripts\\m0_move_verify.py [--dir up] [--hold-ms 600]
方向可选: up/down/left/right/left_up/right_up/left_down/right_down
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


def load_absolute_calib():
    import json
    with open(ROOT / "configs" / "calibration_absolute.json", "r", encoding="utf-8") as f:
        data = json.load(f)
    return data["points"], data["screen_width"], data["screen_height"]


def blue_dot_centroid(frame, mini_center=(110, 93), r=62):
    """小地图 ROI 内蓝色像素质心（归一化到 ROI 内 0-1）。"""
    cx, cy = mini_center
    h, w = frame.shape[:2]
    x0, x1 = max(0, cx - r), min(w, cx + r)
    y0, y1 = max(0, cy - r), min(h, cy + r)
    roi = frame[y0:y1, x0:x1]
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    m = cv2.inRange(hsv, (100, 80, 40), (130, 255, 255))
    ys, xs = np.nonzero(m)
    if len(xs) < 20:
        return None, int((m > 0).sum())
    return (float(xs.mean()) / max(1, x1 - x0), float(ys.mean()) / max(1, y1 - y0)), int((m > 0).sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="up")
    ap.add_argument("--hold-ms", type=int, default=600)
    args = ap.parse_args()

    pts, cw, ch = load_absolute_calib()
    cap = AdbCapture()
    ex = AdbExecutor(serial=cap.serial)

    start = pts["move_stick_center"]
    dir_key = f"dir_{args.dir}"
    if dir_key not in pts:
        print(f"方向必须为: up/down/left/right/left_up/right_up/left_down/right_down")
        return
    target = pts[dir_key]

    before, _ = cap.get_frame()
    if before is None:
        print("截图失败")
        return
    c0, n0 = blue_dot_centroid(before)
    print(f"before: 蓝点质心={c0} 像素数={n0}")

    # 按住摇杆向目标方向拖动
    ex.swipe(*start, *target, duration=args.hold_ms, source="move_verify")
    time.sleep(args.hold_ms / 1000 + 0.3)
    during, _ = cap.get_frame()
    c1, n1 = blue_dot_centroid(during)
    print(f"during: 蓝点质心={c1} 像素数={n1}")

    # 停止（同点 1ms swipe 抬起）
    ex.swipe(*start, *start, duration=1, source="move_verify")
    time.sleep(0.3)
    after, _ = cap.get_frame()
    c2, n2 = blue_dot_centroid(after)
    print(f"after:  蓝点质心={c2} 像素数={n2}")

    if c0 and c1:
        dist = ((c1[0] - c0[0]) ** 2 + (c1[1] - c0[1]) ** 2) ** 0.5
        print(f"\n蓝点位移: {dist:.4f} (ROI 归一化) -> {'移动生效' if dist > 0.01 else '未检测到位移'}")
    else:
        print("\n蓝点检测不足，无法判断（小地图可能在别处或颜色阈值不准）")

    ex.close()


if __name__ == "__main__":
    main()
