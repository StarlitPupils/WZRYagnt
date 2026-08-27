# -*- coding: utf-8 -*-
"""深度学习 v2: 技能按钮亮灭检测(录像帧) -> 用户释放技能时刻 x 局面。"""
import sys
import json
import math
import collections
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
DEMO = ROOT / "data" / "demos" / "20260818_125747"
CAL = json.loads((ROOT / "configs" / "calibration_absolute.json").read_text(encoding="utf-8"))["points"]


def main():
    cap = cv2.VideoCapture(str(DEMO / "stream_0001.mkv"))
    points = {"skill1": CAL["skill1"], "skill2": CAL["skill2"],
              "attack": CAL["attack"], "skill3": CAL["skill3"]}
    step = 6   # 每6帧采样(0.1s)
    idx = 0
    series = {k: [] for k in points}   # 每按钮: [(vidx, vmean)]
    while True:
        ok, f = cap.read()
        if not ok:
            break
        if idx % step == 0:
            for k, (px, py) in points.items():
                roi = f[py - 14:py + 14, px - 14:px + 14]
                v = float(cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY).mean()) if roi.size else 0.0
                series[k].append((idx, v))
        idx += 1
    cap.release()
    print(f"采样点 {len(series['skill2'])} (源帧 {idx})")

    # 技能按钮按下时刻 = 亮度突降(冷却图标变暗/按下变亮) 检测: |ΔV| 突变峰
    # 推断: 技能按钮"按下"时图标短暂高亮; 冷却时变暗。用局部 z-score 峰
    presses = collections.defaultdict(list)
    for k, arr in series.items():
        vs = np.array([v for _, v in arr])
        med = np.median(vs)
        mad = np.median(np.abs(vs - med)) + 1e-6
        for i in range(1, len(vs)):
            if abs(vs[i] - vs[i - 1]) > 28 and vs[i - 1] != 0:
                presses[k].append(arr[i][0])
    for k in points:
        print(f"  {k}: 突变峰 {len(presses[k])} 次, 时间(帧) {presses[k][:20]}")

    # 对齐敌人序列: 用 user_events.json 的敌距(0.67s 间隔) -> 以帧为轴插值
    ev_path = ROOT / "temp" / "user_events.json"
    evs = json.loads(ev_path.read_text(encoding="utf-8")) if ev_path.exists() else []
    ev_by_frame = {e["f"]: e for e in evs}

    def dist_at(fidx):
        # 就近事件
        best = None
        for fidx2, e in ev_by_frame.items():
            d0 = abs(fidx2 - fidx)
            if best is None or d0 < best[0]:
                best = (d0, e)
        if best and best[0] <= step * 3:
            return best[1].get("edist"), best[1].get("enemy", 0)
        return None, 0

    print("\n=== 技能释放时局面(skill2 钩子) ===")
    tbl = collections.Counter()
    n = 0
    for fidx in presses["skill2"][:100]:
        ed, ecnt = dist_at(fidx)
        db = "敌<0.25" if ed is not None and ed < 0.25 else (
            "敌0.25-0.45" if ed is not None and ed < 0.45 else (
                "敌0.45-0.8" if ed is not None and ed < 0.8 else "敌远/无"))
        tbl[db] += 1
        n += 1
    for k, v in tbl.most_common():
        print(f"  {k}: {v}")

    print("\n=== attack 冲击时局面 ===")
    tbl2 = collections.Counter()
    for fidx in presses["attack"][:100]:
        ed, ecnt = dist_at(fidx)
        db = "敌<0.25" if ed is not None and ed < 0.25 else (
            "敌0.25-0.45" if ed is not None and ed < 0.45 else (
                "敌0.45-0.8" if ed is not None and ed < 0.8 else "敌远/无"))
        tbl2[db] += 1
    for k, v in tbl2.most_common():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
