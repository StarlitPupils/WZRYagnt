# -*- coding: utf-8 -*-
"""对局截图标注器：把理解层看到的信息画到截图上，供用户矫正。

用法：
    venv\\Scripts\\python.exe scripts\\annotate_frames.py frame1.png frame2.png ...
输出：同目录 annot_<原名>.png
"""
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from wzry.vision.minimap_tracker import MinimapTracker  # noqa: E402
from wzry.vision.terrain import DEFAULT_BOX  # noqa: E402
from wzry.vision.detector import YoloDetector  # noqa: E402
from wzry.vision.self_bars import self_hp_mp, detect_all_bars  # noqa: E402
from wzry.calib import load_calibration  # noqa: E402
import json  # noqa: E402

_COLORS = {
    "green": (0, 255, 0), "blue": (255, 128, 0), "red": (0, 0, 255),
    "yellow": (0, 255, 255),
}
_LABELS = {"green": "SELF", "blue": "ALLY", "red": "ENEMY", "yellow": "MONSTER"}


def annotate(path, det, calib, out_dir):
    img = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        print("无法读取", path)
        return
    vis = img.copy()
    h, w = img.shape[:2]
    mc = calib.get("minimap_center", [0.086, 0.129])

    # ---- 小地图（生产 tracker）----
    tracker = MinimapTracker(prior_center=(int(mc[0] * w), int(mc[1] * h)),
                             box_prior=DEFAULT_BOX)
    mm = tracker.update(img)
    if mm.get("found"):
        cx, cy = int(mm["center"][0]), int(mm["center"][1])
        r = int(mm.get("radius", 116))
        x0, y0 = cx - r, cy - r
        # 小地图框
        cv2.rectangle(vis, (x0, y0), (x0 + 2 * r, y0 + 2 * r), (255, 200, 0), 2)
        cv2.putText(vis, "MINIMAP", (x0, y0 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (255, 200, 0), 2)
        # 小地图内点（n 坐标 -> 小地图框内全屏坐标）
        for k, color in _COLORS.items():
            for p in (mm.get("dots") or {}).get(k, []):
                px = int(x0 + p[0] * 2 * r)
                py = int(y0 + p[1] * 2 * r)
                cv2.circle(vis, (px, py), 5, color, -1)
                cv2.putText(vis, _LABELS[k], (px + 7, py),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        # 塔
        for t in mm.get("towers", []):
            px = int(x0 + t[0] * 2 * r)
            py = int(y0 + t[1] * 2 * r)
            cv2.rectangle(vis, (px - 5, py - 5), (px + 5, py + 5), (0, 165, 255), -1)
            cv2.putText(vis, "TOWER", (px + 8, py), cv2.FONT_HERSHEY_SIMPLEX,
                        0.4, (0, 165, 255), 1)
    else:
        cv2.putText(vis, "NO MINIMAP", (30, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (255, 200, 0), 2)

    # ---- YOLO 检测框 ----
    for d in det.detect(img):
        x1, y1, x2, y2 = (int(v) for v in d.xyxy)
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 255), 2)
        cv2.putText(vis, f"{d.cls} {d.conf:.2f}", (x1, max(0, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)

    # ---- 自己 HP/MP/位置 ----
    hp, mp, pos = self_hp_mp(img)
    if pos:
        cv2.circle(vis, pos, 9, (255, 0, 255), 3)
        cv2.putText(vis, f"SELF HP={hp} MP={mp}", (pos[0] + 14, pos[1]),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 0, 255), 2)
    else:
        cv2.putText(vis, "NO SELF BAR", (30, 60), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (255, 0, 255), 2)

    # ---- 血条检测（全画面三色）----
    bars = detect_all_bars(img)
    for b in bars.get("enemies", []):
        cv2.rectangle(vis, (b["x"] - b["w"] // 2, b["y"] - 4),
                      (b["x"] + b["w"] // 2, b["y"] + 4), (0, 0, 255), 1)
        cv2.putText(vis, "ENEMY_BAR", (b["x"] - 30, b["y"] - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)
    for b in bars.get("allies", []):
        cv2.rectangle(vis, (b["x"] - b["w"] // 2, b["y"] - 4),
                      (b["x"] + b["w"] // 2, b["y"] + 4), (255, 128, 0), 1)
        cv2.putText(vis, "ALLY_BAR", (b["x"] - 30, b["y"] - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 128, 0), 1)

    # ---- 技能状态 ----
    try:
        pts = json.loads((ROOT / "configs" / "calibration_absolute.json")
                         .read_text(encoding="utf-8"))["points"]
        from wzry.vision.ui_reader import skill_ready_state
        ss = skill_ready_state(img, {1: pts["skill1"], 2: pts["skill2"],
                                     3: pts["skill3"]})
        txt = " | ".join(f"S{k}:{'ok' if ss[str(k)].get('ready') else 'CD'}"
                         for k in (1, 2, 3))
        cv2.putText(vis, txt, (30, 90), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (255, 255, 255), 2)
    except Exception as e:
        cv2.putText(vis, f"skills err: {e}", (30, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    out = out_dir / f"annot_{path.stem}.png"
    cv2.imwrite(str(out), vis)
    print(f"已保存: {out}")


def main(argv):
    det = YoloDetector(str(ROOT / "runs" / "detect" / "zhongkui_11cls"
                           / "weights" / "best.pt"), conf=0.35)
    calib, _ = load_calibration()
    out_dir = ROOT / "temp" / "annot"
    out_dir.mkdir(exist_ok=True)
    for p in argv[1:]:
        annotate(Path(p), det, calib, out_dir)


if __name__ == "__main__":
    main(sys.argv)
