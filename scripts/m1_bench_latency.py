# -*- coding: utf-8 -*-
"""M1 延迟基准：量化 采集/检测/小地图 各环节延迟（对局画面）。

用法：
    venv\\Scripts\\python.exe scripts\\m1_bench_latency.py [--seconds 20] [--model ...]

输出各环节平均/最大延迟（ms），并给出端到端预算结论。
"""
import argparse
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from wzry.calib import load_calibration  # noqa: E402
from wzry.capture.scrcpy_stream import ScrcpyStreamCapture  # noqa: E402
from wzry.state.match_state import MatchStateMachine  # noqa: E402
from wzry.vision.detector import YoloDetector  # noqa: E402
from wzry.vision.minimap_tracker import MinimapTracker  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, default=20.0)
    ap.add_argument("--model", default="runs/detect/zhongkui_detector_finetune/weights/best.pt")
    args = ap.parse_args()

    cap = ScrcpyStreamCapture(ROOT / "tools" / "scrcpy")
    print("启动 scrcpy 流...")
    cap.start()

    calib, _ = load_calibration()
    sm = MatchStateMachine(minimap_center_norm=calib.get("minimap_center", [0.086, 0.129]))
    det = YoloDetector(args.model, conf=0.35)

    h, w = None, None
    first = cap.wait_frame(timeout=8.0)
    if first[0] is not None:
        h, w = first[0].shape[:2]
    mc = calib.get("minimap_center", [0.086, 0.129])
    tracker = MinimapTracker(prior_center=[int(mc[0] * w), int(mc[1] * h)] if w else None)

    lat_cap, lat_det, lat_mm = [], [], []
    n_detect = n_mm = 0
    last_det = last_mm = 0.0
    t_end = time.time() + args.seconds

    print("采样中（建议在训练营对局内运行以获得有效数字）...")
    while time.time() < t_end:
        frame, _ = cap.wait_frame(timeout=2.0)
        if frame is None:
            continue
        phase = sm.update(frame)
        now = time.time()
        if phase.value == "in_match":
            if now - last_det > 0.5:
                last_det = now
                det.detect(frame)
                lat_det.append(det.last_infer_ms)
                n_detect += 1
            if now - last_mm > 0.5:
                last_mm = now
                tracker.update(frame)
                lat_mm.append(tracker.last_ms)
                n_mm += 1
        lat_cap.append(0)  # 采集延迟已在 wait_frame 内，帧到达即最新

    cap.stop()

    def stat(name, v):
        if v:
            print(f"  {name:<12} n={len(v):>3}  平均 {statistics.mean(v):6.1f} ms  "
                  f"最大 {max(v):6.1f} ms")
        else:
            print(f"  {name:<12} 无样本（未进入对局画面？）")
        return statistics.mean(v) if v else 0.0

    print("\n=== M1 延迟基准 ===")
    print(f"采集方式: scrcpy mkv 流 (~{cap.max_fps}fps 理论帧间隔 {1000/cap.max_fps:.0f}ms)")
    mean_det = stat("检测 YOLO", lat_det)
    mean_mm = stat("小地图", lat_mm)
    # 有效帧率估算：感知主循环 = 检测频率
    if n_detect:
        print(f"  检测频率: 每 0.5s 一次（可按需提高）")
    print("\n端到端预算（采集33 + 检测 + 小地图并行 + 决策预留50）: "
          f"{33 + max(mean_det, mean_mm) + 50:.0f} ms")


if __name__ == "__main__":
    main()
