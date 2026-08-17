# -*- coding: utf-8 -*-
"""小地图模块最终验证（诚实口径）。

用例：
  1. 合成帧回归：720p 合成小地图（暗盘+蓝/红/黄点+塔点）应 found=True 且坐标精确；
  2. 录像对局段（tmphjhl7fk3.mp4，自由视角无 HUD）：定位应为 0（无小地图可找）；
  3. 录像场景段假阳性率：无 HUD 场景不应被误判为小地图（单帧门控）+ 连续帧圆心一致性
     （真实小地图圆心固定，场景误报会漂移——实时管线靠 tracker 固定圆心兜底）；
  4. 单帧耗时（全扫描 / fast_prior）。

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
        cv2.circle(frame, (cx + dx, cy + dy), 3, (255, 128, 0), -1)  # 塔点 r=3
    return frame, (cx, cy, r)


def main():
    results = {}

    # 1) 合成帧回归
    frame, (cx, cy, r) = make_synthetic()
    res = minimap.analyze(frame)
    err = (np.hypot(res["center"][0] - cx, res["center"][1] - cy)
           if res["found"] and res["center"] else None)
    results["synthetic"] = {
        "found": res["found"], "center": res["center"], "radius": res["radius"],
        "center_err_px": round(float(err), 1) if err is not None else None,
        "blue": len(res["dots"]["blue"]), "red": len(res["dots"]["red"]),
        "yellow": len(res["dots"]["yellow"]), "towers": len(res["towers"]),
    }
    print("[1] 合成帧:", results["synthetic"])

    cap = cv2.VideoCapture(str(VIDEO))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # 2) 录像对局段定位（自由视角，期望 0）+ 假阳性率
    ok_cnt = 0
    total = 0
    times = []
    for pos in range(4000, 14000, 400):
        cap.set(cv2.CAP_PROP_POS_FRAMES, pos)
        ok, fr = cap.read()
        if not ok:
            continue
        total += 1
        t0 = time.perf_counter()
        r2 = minimap.find_minimap(fr)
        times.append((time.perf_counter() - t0) * 1000)
        if r2["found"]:
            ok_cnt += 1
    results["replay_no_hud"] = {"false_positive": f"{ok_cnt}/{total}",
                                "avg_ms": round(statistics.mean(times), 1) if times else None}
    print("[2] 录像对局段(无HUD) 假阳性:", results["replay_no_hud"])

    # 3) 连续帧圆心一致性（真小地图圆心固定；场景误报会漂移）
    cap.set(cv2.CAP_PROP_POS_FRAMES, 6000)
    centers = []
    for _ in range(40):
        ok, fr = cap.read()
        if not ok:
            break
        r3 = minimap.find_minimap(fr)
        if r3["found"]:
            centers.append(r3["center"])
    if len(centers) >= 5:
        drift = float(np.mean([np.hypot(c[0] - centers[0][0], c[1] - centers[0][1])
                               for c in centers[1:]]))
        results["consistency"] = {"n_found": len(centers), "mean_drift_px": round(drift, 1)}
        print(f"[3] 连续帧一致性: 检出 {len(centers)} 帧, 平均圆心漂移 {drift:.1f}px "
              f"(真小地图应≈0；场景误报会大)")
    else:
        results["consistency"] = {"n_found": len(centers)}
        print(f"[3] 连续帧一致性: 检出 {len(centers)} 帧（不足）")
    cap.release()

    # 4) fast_prior 耗时（合成帧上）
    ft = []
    seed = minimap.find_minimap(frame)
    if seed["found"]:
        for _ in range(5):
            t0 = time.perf_counter()
            minimap.find_minimap(frame, prior=seed["center"], fast_prior=True)
            ft.append((time.perf_counter() - t0) * 1000)
    results["fast_prior_ms"] = round(statistics.mean(ft), 1) if ft else None
    print("[4] fast_prior 平均:", results["fast_prior_ms"], "ms")

    out = ROOT / "temp" / "minimap_verify.json"
    out.write_text(__import__("json").dumps(results, ensure_ascii=False, indent=2),
                   encoding="utf-8")
    print(f"\n结果已写入 {out}")


if __name__ == "__main__":
    main()
