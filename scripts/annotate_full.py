# -*- coding: utf-8 -*-
"""全屏+小地图双 YOLO 标注：展示理解层完整输出。

- 全屏 11 类 YOLO：敌方英雄/我方英雄/小兵/塔/野怪等
- 小地图 9 类 YOLO：英雄标记(自己/队友/敌人)/塔/野怪/buff/小兵
- 自己 HP/MP 血条检测
输出: temp/annot_full_<name>.png

用法:
  venv\\Scripts\\python.exe scripts\\annotate_full.py <帧图>...
"""
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from wzry.vision.detector import YoloDetector  # noqa: E402
from wzry.vision.mm_hybrid import MMHybridDetector  # noqa: E402
from wzry.vision.self_bars import self_hp_mp  # noqa: E402

FULL_COLORS = {
    "enemy_hero": (0, 0, 255), "ally_hero": (255, 128, 0),
    "enemy_minion": (0, 100, 255), "ally_minion": (100, 200, 0),
    "enemy_turret": (0, 0, 200), "ally_turret": (200, 100, 0),
    "neutral_monster": (0, 255, 255), "enemy_crystal": (255, 0, 200),
    "ally_crystal": (200, 255, 0), "hook_aim": (255, 0, 255),
    "skill_effect": (255, 255, 0),
}
MM_COLORS = {"self": (0, 255, 0), "ally": (255, 128, 0), "enemy": (0, 0, 255),
             "monster": (0, 255, 255), "buff": (255, 0, 255)}
MM_LABELS = {"self": "自己", "ally": "队友", "enemy": "敌人",
             "monster": "野怪", "buff": "buff"}


def annotate(path, full_det, mm_det):
    img = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        print("无法读取", path)
        return
    vis = img.copy()
    h, w = img.shape[:2]

    # ---- 全屏 11 类（v4.1: 排除画面中央超大框=自己英雄/视角误检）----
    n_full = 0
    for d in full_det.detect(img):
        x1, y1, x2, y2 = (int(v) for v in d.xyxy)
        bw, bh = x2 - x1, y2 - y1
        # 中央大框过滤：覆盖画面中央且面积 >45% 的检测 = 自己英雄/大模型误检
        if (bw * bh) > 0.45 * h * w:
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            if 0.25 * w < cx < 0.75 * w and 0.25 * h < cy < 0.75 * h:
                continue
        color = FULL_COLORS.get(d.cls, (200, 200, 200))
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
        cv2.putText(vis, f"{d.cls} {d.conf:.2f}", (x1, max(0, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        n_full += 1

    # ---- 自己 HP/MP ----
    hp, mp, pos = self_hp_mp(img)
    if pos:
        cv2.circle(vis, pos, 10, (255, 0, 255), 3)
        cv2.putText(vis, f"自己 HP={hp} 蓝={mp}", (min(pos[0] + 15, w - 260),
                    max(30, pos[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (255, 255, 255), 2)

    # ---- 技能状态（校准坐标）----
    try:
        import json
        pts = json.loads((ROOT / "configs" / "calibration_absolute.json")
                         .read_text(encoding="utf-8"))["points"]
        from wzry.vision.ui_reader import skill_ready_state
        ss = skill_ready_state(img, {1: pts["skill1"], 2: pts["skill2"],
                                     3: pts["skill3"]})
        stxt = " | ".join(f"S{k}:{'READY' if ss[str(k)].get('ready') else 'CD'}"
                          for k in (1, 2, 3))
        cv2.putText(vis, f"SKILLS: {stxt}", (10, h - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
    except Exception:
        pass

    # ---- 小地图 v5 ----
    mm = mm_det.detect(img)
    mm_txt = ""
    if mm.get("found"):
        x0 = int(mm["center"][0] - mm["radius"])
        y0 = int(mm["center"][1] - mm["radius"])
        r2 = int(mm["radius"] * 2)
        # 小地图放大 2x 贴到画面右上角，便于看清
        mm_crop = img[y0:y0 + r2, x0:x0 + r2]
        mm_big = cv2.resize(mm_crop, (r2 * 2, r2 * 2), interpolation=cv2.INTER_NEAREST)
        ox, oy = w - r2 * 2 - 10, 10
        vis[oy:oy + r2 * 2, ox:ox + r2 * 2] = mm_big
        cnts = []
        for grp, color in MM_COLORS.items():
            for d in mm["dots"][grp]:
                px = int(ox + d["n"][0] * r2 * 2)
                py = int(oy + d["n"][1] * r2 * 2)
                cv2.circle(vis, (px, py), 7, color, -1)
                cv2.putText(vis, MM_LABELS[grp],
                            (min(px + 9, w - 60), min(max(12, py), oy + r2 * 2 - 4)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
                cnts.append(grp)
        for side, color in (("ally", (200, 100, 0)), ("enemy", (0, 100, 200))):
            for t in mm["towers"][side]:
                px = int(ox + t["n"][0] * r2 * 2)
                py = int(oy + t["n"][1] * r2 * 2)
                cv2.rectangle(vis, (px - 7, py - 7), (px + 7, py + 7), color, -1)
                cv2.putText(vis, f"{side.upper()}TW", (px + 9, py),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 2)
                cnts.append(side + "_tw")
        for side, color in (("ally", (150, 255, 150)), ("enemy", (150, 150, 255))):
            for m in mm["minions"][side]:
                px = int(ox + m["n"][0] * r2 * 2)
                py = int(oy + m["n"][1] * r2 * 2)
                cv2.circle(vis, (px, py), 4, color, -1)
                cnts.append(side + "_minion")
        mm_txt = f"MM: S{len(mm['dots']['self'])} A{len(mm['dots']['ally'])} " \
                 f"E{len(mm['dots']['enemy'])} M{len(mm['dots']['monster'])} " \
                 f"B{len(mm['dots']['buff'])} TW{len(mm['towers']['ally']) + len(mm['towers']['enemy'])}"
        cv2.putText(vis, "MINIMAP(x2) " + mm_txt, (ox, oy - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 200, 0), 2)
    else:
        cv2.putText(vis, "NO MINIMAP", (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 255), 2)

    # ---- 信息汇总（底部黑底白字，标注含义清晰）----
    summary = (f"全屏目标={n_full} | 小地图: 自己{len(mm['dots']['self']) if mm.get('found') else 0} "
               f"队友{len(mm['dots']['ally']) if mm.get('found') else 0} "
               f"敌人{len(mm['dots']['enemy']) if mm.get('found') else 0} "
               f"野怪{len(mm['dots']['monster']) if mm.get('found') else 0} "
               f"塔{len(mm['towers']['ally']) + len(mm['towers']['enemy']) if mm.get('found') else 0} | "
               f"自己HP={hp} 蓝={mp}")
    cv2.rectangle(vis, (0, h - 34), (w, h), (0, 0, 0), -1)
    cv2.putText(vis, summary, (10, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (0, 255, 255), 2)

    out = ROOT / "temp" / f"annot_full_{path.stem}.png"
    cv2.imwrite(str(out), vis)
    print("已保存:", out, "|", summary)


def main():
    full_weights = sys.argv[1] if len(sys.argv) > 1 and sys.argv[1].endswith(".pt") else None
    paths = sys.argv[2:] if full_weights else sys.argv[1:]
    full_det = YoloDetector(str(full_weights or ROOT / "runs" / "detect"
                                / "zhongkui_11cls" / "weights" / "best.pt"),
                            conf=0.25)
    mm_det = MMHybridDetector(str(ROOT / "runs" / "mm_detect" / "mm_v6"
                                  / "weights" / "best.pt"))
    for p in paths:
        annotate(Path(p), full_det, mm_det)


if __name__ == "__main__":
    main()
