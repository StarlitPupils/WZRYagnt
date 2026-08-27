# -*- coding: utf-8 -*-
"""深度学习用户手打局: 录像感知(技能/敌人) × states(局面) → 技能触发规则 + 全局叙事。

用法: venv\\Scripts\\python.exe -X utf8 -u scripts\\learn_deep.py
"""
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
from wzry.vision.detector import YoloDetector
from wzry.vision.self_bars import self_hp_mp


def main():
    det = YoloDetector(str(ROOT / "runs" / "detect" / "zhongkui_11cls" / "weights" / "best.pt"),
                       conf=0.25)
    cap = cv2.VideoCapture(str(DEMO / "stream_0001.mkv"))
    # 抽帧: 每10帧=~0.17s? 60fps -> 每 6 帧=0.1s 太重; 每 40 帧(0.67s) 记录
    step = 40
    idx = 0
    events = []  # (vidx, skill_fx_cnt, hook_cnt, enemy_cnt, enemy_dist)
    frames_used = 0
    while True:
        ok, f = cap.read()
        if not ok:
            break
        if idx % step == 0:
            try:
                dets = det.detect(f)
                skills = [d for d in dets if d.cls in ("skill_effect", "hook_aim")]
                enemies = [d for d in dets if d.cls == "enemy_hero"]
                edist = None
                if enemies:
                    e = enemies[0]
                    cx = (e.xyxy[0] + e.xyxy[2]) / 2 / 1280
                    cy = (e.xyxy[1] + e.xyxy[3]) / 2 / 720
                    edist = math.hypot(cx - 0.5, cy - 0.5)
                hp, mp, pos = self_hp_mp(f)
                events.append({"f": idx, "skill": len(skills), "enemy": len(enemies),
                               "edist": edist, "hp": hp})
            except Exception:
                pass
            frames_used += 1
        idx += 1
    cap.release()
    print(f"抽帧 {len(events)} (源帧 {idx})")

    # 技能时刻 = skill>0 且前 3 帧无 (新释放)
    skill_times = []
    for k in range(len(events)):
        e = events[k]
        if e["skill"] > 0 and (k == 0 or events[k - 1]["skill"] == 0):
            skill_times.append(k)
    print("技能释放次数(粗略):", len(skill_times))
    # 技能时局面
    print("\n=== 技能释放时的局面分布 ===")
    dists = collections.Counter()
    enemies_a = collections.Counter()
    hp_b = collections.Counter()
    for k in skill_times:
        e = events[k]
        d = e["edist"]
        db = "敌<0.25" if d is not None and d < 0.25 else ("敌0.25-0.45" if d is not None and d < 0.45 else ("敌0.45-0.8" if d is not None and d < 0.8 else "敌远/无"))
        dists[db] += 1
        enemies_a["有敌" if e["enemy"] > 0 else "无敌"] += 1
        h = e["hp"]
        hb = "HP<40%" if h is not None and h < 0.40 else ("HP40-80%" if h is not None and h < 0.80 else "HP>80%")
        hp_b[hb] += 1
    for k, v in dists.most_common():
        print(f"  距离: {k}: {v}")
    for k, v in enemies_a.most_common():
        print(f"  敌人: {k}: {v}")
    for k, v in hp_b.most_common():
        print(f"  HP: {k}: {v}")

    # 全局面: 接战(敌人出现)时间占比 / 技能频率 / 移动敌距分布
    n_enemy_frames = sum(1 for e in events if e["enemy"] > 0)
    print(f"\n=== 全局 ===")
    print(f"  敌人出现在屏幕: {n_enemy_frames}/{len(events)} = {n_enemy_frames/len(events)*100:.0f}% 帧")
    print(f"  技能帧占比: {sum(1 for e in events if e['skill']>0)/len(events)*100:.0f}%")
    print(f"  平均敌距(有敌帧): "
          f"{np.mean([e['edist'] for e in events if e['edist'] is not None]):.2f}")

    # 保存事件流供继续分析
    (ROOT / "temp" / "user_events.json").write_text(json.dumps(events), encoding="utf-8")
    print("事件流->temp/user_events.json")


if __name__ == "__main__":
    main()
