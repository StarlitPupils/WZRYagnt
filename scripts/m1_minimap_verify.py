# -*- coding: utf-8 -*-
"""小地图模块最终验证（三组用例 + 性能）。

用例：
  1. 合成帧回归：720p 合成小地图（暗盘+蓝/红/黄点+塔点）应 found=True 且坐标接近真值；
  2. 录像 HUD 段（tmphjhl7fk3.mp4 ~11451-11820 帧，真实带小地图段）定位成功率；
  3. 录像无 HUD 段（~3000-9000 帧）应无假阳性；
  4. 单帧耗时统计（全扫描 / fast_prior）。

用法：
    venv\\Scripts\\python.exe scripts\\m1_minimap_verify.py
"""
import statistics
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from wzry.vision import minimap  # noqa: E402

VIDEO = ROOT / "temp" / "tmphjhl7fk3.mp4"


def make_synthetic():
    frame = np.full((720, 1280, 3), 60, dtype=np.uint8)
    cy, cx, r = 110, 110, 95
    yy, xx = np.ogrid[:720, :1280]
    frame[(xx - cx) ** 2 + (yy - cy) ** 2 <= r * r] = (40, 45, 50)
    for (dx, dy, col) in [(-40, -30, (255, 128, 0)), (0, -10, (255, 128, 0)),
                          (30, 20, (255, 128, 0)), (-20, 25, (0, 0, 255)),
                          (10, 35, (0, 0, 255)), (40, -20, (0, 255, 255))]:
        cv2.circle(frame, (cx + dx, cy + dy), 7, col, -1)
    for (dx, dy) in [(-50, 10), (50, 40), (-10, -50), (45, -35)]:
        cv2.circle(frame, (cx + dx, cy + dy), 4, (255, 128, 0), -1)
    return frame, (cx, cy, r)


def main():
    results = {}

    # 1) 合成帧
    frame, (cx, cy, r) = make_synthetic()
    res = minimap.analyze(frame)
    ok1 = res["found"] and res["center"] is not None
    err = np.hypot(res["center"][0] - cx, res["center"][1] - cy) if ok1 else None
    results["synthetic"] = {
        "found": res["found"], "center": res["center"], "radius": res["radius"],
        "gt": [cx, cy, r], "center_err_px": round(float(err), 1) if err is not None else None,
        "blue": len(res["dots"]["blue"]), "red": len(res["dots"]["red"]),
        "yellow": len(res["dots"]["yellow"]), "towers": len(res["towers"]),
    }
    print("[1] 合成帧:", results["synthetic"])

    # 2) 录像 HUD 段
    cap = cv2.VideoCapture(str(VIDEO))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    ok_cnt = 0
    times = []
    for pos in range(11451, 11820, 25):
        cap.set(cv2.CAP_PROP_POS_FRAMES, pos)
        ok, fr = cap.read()
        if not ok:
            continue
        t0 = time.perf_counter()
        r2 = minimap.find_minimap(fr)
        times.append((time.perf_counter() - t0) * 1000)
        if r2["found"]:
            ok_cnt += 1
    total2 = len(times)
    results["hud_segment"] = {"found": f"{ok_cnt}/{total2}",
                              "avg_ms": round(statistics.mean(times), 1) if times else None}
    print("[2] 录像HUD段:", results["hud_segment"])

    # 3) 无 HUD 段（假阳性检查）
    fp = 0
    for pos in range(3000, 9000, 400):
        cap.set(cv2.CAP_PROP_POS_FRAMES, pos)
        ok, fr = cap.read()
        if not ok:
            continue
        if minimap.find_minimap(fr)["found"]:
            fp += 1
    results["no_hud"] = {"false_positives": fp}
    print("[3] 录像无HUD段 假阳性:", fp)
    cap.release()

    # 4) fast_prior 性能（在 HUD 段帧上）
    cap = cv2.VideoCapture(str(VIDEO))
    cap.set(cv2.CAP_PROP_POS_FRAMES, 11451)
    ok, fr = cap.read()
    cap.release()
    ft = []
    if ok:
        seed = minimap.find_minimap(fr)
        if seed["found"]:
            for _ in range(5):
                t0 = time.perf_counter()
                minimap.find_minimap(fr, prior=seed["center"], fast_prior=True)
                ft.append((time.perf_counter() - t0) * 1000)
    results["fast_prior_ms"] = round(statistics.mean(ft), 1) if ft else None
    print("[4] fast_prior 平均:", results["fast_prior_ms"], "ms")

    out = ROOT / "temp" / "minimap_verify.json"
    out.write_text(__import__("json").dumps(results, ensure_ascii=False, indent=2),
                   encoding="utf-8")
    print(f"\n结果已写入 {out}")


if __name__ == "__main__":
    main()
