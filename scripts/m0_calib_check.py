# -*- coding: utf-8 -*-
"""M0 校准可视化 + 坐标映射经验验证。

用法：
    venv\\Scripts\\python.exe scripts\\m0_calib_check.py [--save temp/calib_check.png]
    venv\\Scripts\\python.exe scripts\\m0_calib_check.py --tap-verify [--tap-point attack]

--tap-verify：在训练营内点击指定校准点，用"点击前后 ROI 像素变化"经验判定
  窗口像素 -> 设备像素的正确映射（含旋转候选），选出变化最大的映射。
"""
import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from wzry.calib import load_calibration  # noqa: E402
from wzry.capture.window import WindowCapture  # noqa: E402
from wzry.control.executor import AdbExecutor, discover_mumu_adb_devices  # noqa: E402


def draw_points(img, pts_norm, color=(0, 255, 0), radius=8):
    h, w = img.shape[:2]
    for label, (nx, ny) in pts_norm.items():
        x, y = int(nx * w), int(ny * h)
        cv2.circle(img, (x, y), radius, color, 2)
        cv2.putText(img, label, (x + radius + 2, y - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)


def roi_change(before, after, center, r=40):
    """以 center(窗口像素) 为中心的正方形 ROI 平均绝对像素差。"""
    h, w = before.shape[:2]
    cx, cy = int(center[0]), int(center[1])
    x0, x1 = max(0, cx - r), min(w, cx + r)
    y0, y1 = max(0, cy - r), min(h, cy + r)
    if x1 <= x0 or y1 <= y0:
        return 0.0
    return float(np.abs(before[y0:y1, x0:x1].astype(np.int16)
                        - after[y0:y1, x0:x1].astype(np.int16)).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--save", default=str(ROOT / "temp" / "calib_check.png"))
    ap.add_argument("--tap-verify", action="store_true",
                    help="经验验证窗口->设备坐标映射（训练营内安全）")
    ap.add_argument("--tap-point", default="attack",
                    help="验证点击的校准点，默认 attack")
    args = ap.parse_args()

    cap = WindowCapture()
    frame, ms = cap.get_frame()
    if frame is None:
        print("未找到 MuMu 窗口")
        return
    win_h, win_w = frame.shape[:2]
    print(f"窗口画面: {win_w}x{win_h} (采集 {ms:.0f} ms)")

    calib, src = load_calibration()
    print(f"校准源: {src}  校准点: {len(calib)} 个")

    # 1) 覆盖图：把归一化校准点画到当前窗口画面上
    overlay = frame.copy()
    draw_points(overlay, calib)
    out = Path(args.save)
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), overlay)
    print(f"覆盖图已保存: {out}（请人工核对：minimap_center 应在小地图圆心，"
          f"skill1/2/3 应在技能按钮，move_stick_center 应在摇杆中心）")

    # 1.5) 校准点落位合理性（程序化）：小地图应暗、摇杆/技能区应与周围有对比
    print("\n[落位合理性检查]")
    def roi_gray(img, nx, ny, r=18):
        h, w = img.shape[:2]
        cx, cy = int(nx * w), int(ny * h)
        x0, x1 = max(0, cx - r), min(w, cx + r)
        y0, y1 = max(0, cy - r), min(h, cy + r)
        return float(img[y0:y1, x0:x1].mean())
    for key in ("minimap_center", "move_stick_center", "skill1", "skill2", "skill3", "attack"):
        if key in calib:
            v = roi_gray(frame, *calib[key])
            hint = ""
            if key == "minimap_center" and v > 120:
                hint = "  <- 小地图应是暗色，若偏亮则校准或画面不符"
            print(f"  {key:<18} ROI平均灰度 {v:6.1f}{hint}")

    if not args.tap_verify:
        cap.close()
        return

    # 2) 经验验证映射
    serials = discover_mumu_adb_devices()
    if not serials:
        print("无 ADB 设备，跳过 tap 验证")
        cap.close()
        return
    ex = AdbExecutor(serial=serials[0])
    dev_w, dev_h = ex.get_device_size()
    print(f"设备分辨率: {dev_w}x{dev_h}")

    pt = calib.get(args.tap_point)
    if not pt:
        print(f"校准点不存在: {args.tap_point}，可选: {list(calib.keys())}")
        cap.close()
        return
    win_x, win_y = int(pt[0] * win_w), int(pt[1] * win_h)
    print(f"\n验证点: {args.tap_point} 窗口坐标 ({win_x}, {win_y})，"
          f"候选映射点击后对比 ROI 变化：\n")

    # 候选映射（窗口归一化 -> 设备像素）
    nx, ny = pt[0], pt[1]
    candidates = {
        "identity_scale": (nx * dev_w, ny * dev_h),
        "rot_cw_norm": ((1 - ny) * dev_w, nx * dev_h),     # 窗口逆时针=设备顺时针? 两种都试
        "rot_ccw_norm": (ny * dev_w, (1 - nx) * dev_h),
    }
    results = []
    for name, (dx, dy) in candidates.items():
        before, _ = cap.get_frame()
        t0 = time.perf_counter()
        ok = ex.tap(dx, dy, source="calib_check")
        lat = (time.perf_counter() - t0) * 1000
        time.sleep(0.25)  # 等按钮高亮消退前抓一帧
        after, _ = cap.get_frame()
        change = roi_change(before, after, (win_x, win_y)) if after is not None else 0.0
        results.append((change, name, (int(dx), int(dy)), ok, lat))
        print(f"  {name:<16} 设备坐标 ({int(dx):>4},{int(dy):>4}) "
              f"ROI变化 {change:6.2f}  延迟 {lat:5.0f} ms")

    best = max(results, key=lambda r: r[0])
    print(f"\n最佳映射: {best[1]}（ROI 变化 {best[0]:.2f}，坐标 {best[2]}）")
    print("注：ROI 变化大 = 点击确实落在按钮上产生了视觉反馈；"
          "若所有候选变化都小，可能校准点本身已偏移，需重新跑 calibrate_absolute.py。")

    ex.close()
    cap.close()


if __name__ == "__main__":
    main()
