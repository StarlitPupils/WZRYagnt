# -*- coding: utf-8 -*-
"""M0 延迟基准：量化 采集 / ADB 控制 两路延迟。

用法：
    venv\\Scripts\\python.exe scripts\\m0_bench_latency.py [--rounds 10] [--json]

说明：
  - 采集：mss 抓 MuMu 窗口 10 次，统计平均/最差耗时。
  - ADB：用 `input keyevent 0`（KEYCODE_UNKNOWN，无副作用）测命令往返延迟。
  - 模拟器未启动时输出引导信息，不报错。
"""
import argparse
import json
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from wzry.capture.window import WindowCapture  # noqa: E402
from wzry.control.executor import AdbExecutor, discover_mumu_adb_devices  # noqa: E402


def bench_capture(rounds=10):
    cap = WindowCapture()
    lat = []
    for _ in range(rounds):
        frame, ms = cap.get_frame()
        if frame is None:
            break
        lat.append(ms)
    cap.close()
    return lat


def bench_adb(rounds=10):
    """keyevent 0 是无副作用空指令，测真实命令往返延迟。"""
    try:
        ex = AdbExecutor()
    except RuntimeError as e:
        print(f"  [adb] {e}")
        return []
    lat = []
    for _ in range(rounds):
        ex.key(0, source="bench")
        lat.append(ex.last_cmd_ms)
    ex.close()
    return lat


def summarize(name, lat):
    if not lat:
        print(f"  {name:<12} N/A（无法测量，检查上方提示）")
        return None
    print(f"  {name:<12} 平均 {statistics.mean(lat):6.1f} ms | "
          f"中位 {statistics.median(lat):6.1f} ms | 最大 {max(lat):6.1f} ms | n={len(lat)}")
    return {"name": name, "mean_ms": round(statistics.mean(lat), 1),
            "median_ms": round(statistics.median(lat), 1),
            "max_ms": round(max(lat), 1), "n": len(lat)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=10)
    ap.add_argument("--json", action="store_true", help="结果同时写入 logs/bench_latency.json")
    args = ap.parse_args()

    print(f"=== M0 延迟基准（rounds={args.rounds}）===\n")

    print("[1] 窗口采集 (mss)")
    cap_lat = bench_capture(args.rounds)
    r1 = summarize("capture", cap_lat)

    print("\n[2] ADB 控制（keyevent 0 空指令往返）")
    adb_lat = bench_adb(args.rounds)
    r2 = summarize("adb", adb_lat)

    serials = discover_mumu_adb_devices()
    print(f"\n[3] 设备: {serials if serials else '无（请先启动 MuMu 模拟器并保持王者荣耀在前台）'}")

    if not cap_lat:
        print("\n提示: 未找到 MuMu 窗口。请先启动模拟器，再重试本基准。")
    if not adb_lat:
        print("提示: ADB 无设备。启动模拟器后确认 `adb devices` 能看到 127.0.0.1:*。")

    if args.json:
        out = {"t": time.time(), "rounds": args.rounds,
               "capture": r1, "adb": r2, "devices": serials}
        log_dir = ROOT / "logs"
        log_dir.mkdir(exist_ok=True)
        (log_dir / "bench_latency.json").write_text(
            json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n结果已写入 logs/bench_latency.json")


if __name__ == "__main__":
    main()
