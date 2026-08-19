# -*- coding: utf-8 -*-
"""实时标注模式：截屏 -> 全屏YOLO+小地图YOLO+HP/MP+技能 -> 标注图。

用法:
  venv\\Scripts\\python.exe scripts\\live_annotate.py [--seconds 600] [--interval 3]
输出:
  temp/live_annot.png      最新标注图（覆盖写）
  temp/live_annot.log      每次检测摘要（追加）
"""
import argparse
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from wzry.vision.detector import YoloDetector  # noqa: E402
from wzry.vision.mm_yolo import MMYoloDetector  # noqa: E402
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
MM_LABELS = {"self": "SELF", "ally": "ALLY", "enemy": "ENEMY",
             "monster": "MON", "buff": "BUFF"}


def grab(serial):
    """adb 截屏 -> BGR 帧。"""
    subprocess.run(["adb", "-s", serial, "shell", "screencap", "-p",
                    "/sdcard/_live.png"], capture_output=True, timeout=15)
    subprocess.run(["adb", "-s", serial, "pull", "/sdcard/_live.png",
                    "temp/_live.png"], capture_output=True, timeout=15)
    data = np.fromfile("temp/_live.png", dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def annotate(img, full_det, mm_det):
    vis = img.copy()
    h, w = img.shape[:2]
    n_full = 0
    for d in full_det.detect(img):
        x1, y1, x2, y2 = (int(v) for v in d.xyxy)
        color = FULL_COLORS.get(d.cls, (200, 200, 200))
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
        cv2.putText(vis, f"{d.cls} {d.conf:.2f}", (x1, max(0, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        n_full += 1
    hp, mp, pos = self_hp_mp(img)
    if pos:
        cv2.circle(vis, pos, 10, (255, 0, 255), 3)
        cv2.putText(vis, f"SELF HP={hp} MP={mp}", (pos[0] + 15, pos[1]),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
    mm = mm_det.detect(img)
    mm_txt = "NO MM"
    if mm.get("found"):
        x0 = int(mm["center"][0] - mm["radius"])
        y0 = int(mm["center"][1] - mm["radius"])
        r2 = int(mm["radius"] * 2)
        mm_crop = img[y0:y0 + r2, x0:x0 + r2]
        mm_big = cv2.resize(mm_crop, (r2 * 2, r2 * 2), interpolation=cv2.INTER_NEAREST)
        ox, oy = w - r2 * 2 - 10, 10
        vis[oy:oy + r2 * 2, ox:ox + r2 * 2] = mm_big
        cnts = {"S": len(mm["dots"]["self"]), "A": len(mm["dots"]["ally"]),
                "E": len(mm["dots"]["enemy"]), "M": len(mm["dots"]["monster"]),
                "B": len(mm["dots"]["buff"]),
                "TW": len(mm["towers"]["ally"]) + len(mm["towers"]["enemy"])}
        for grp, color in MM_COLORS.items():
            for d in mm["dots"][grp]:
                px = int(ox + d["n"][0] * r2 * 2)
                py = int(oy + d["n"][1] * r2 * 2)
                cv2.circle(vis, (px, py), 7, color, -1)
                cv2.putText(vis, f"{MM_LABELS[grp]} {d['conf']:.2f}", (px + 9, py),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
        for side, color in (("ally", (200, 100, 0)), ("enemy", (0, 100, 200))):
            for t in mm["towers"][side]:
                px = int(ox + t["n"][0] * r2 * 2)
                py = int(oy + t["n"][1] * r2 * 2)
                cv2.rectangle(vis, (px - 7, py - 7), (px + 7, py + 7), color, -1)
        mm_txt = "MM: " + " ".join(f"{k}{v}" for k, v in cnts.items())
        cv2.putText(vis, "MINIMAP x2", (ox, oy - 6), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (255, 200, 0), 2)
    summary = f"FULL={n_full} | {mm_txt} | HP={hp} MP={mp}"
    cv2.putText(vis, summary, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                (0, 255, 255), 2)
    return vis, summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, default=600)
    ap.add_argument("--interval", type=float, default=3.0)
    ap.add_argument("--serial", default="127.0.0.1:16384")
    ap.add_argument("--full", default="runs/detect/zhongkui_11cls_r13/weights/best.pt")
    ap.add_argument("--mm", default="runs/mm_detect/mm_v5/weights/best.pt")
    args = ap.parse_args()

    full_det = YoloDetector(str(ROOT / args.full), conf=0.25)
    mm_det = MMYoloDetector(str(ROOT / args.mm), conf=0.35)
    log = open(ROOT / "temp" / "live_annot.log", "a", encoding="utf-8")
    t_end = time.time() + args.seconds
    n = 0
    while time.time() < t_end:
        try:
            img = grab(args.serial)
            if img is None:
                log.write(f"[{time.strftime('%H:%M:%S')}] 截屏失败\n")
                log.flush()
                time.sleep(2)
                continue
            vis, summary = annotate(img, full_det, mm_det)
            cv2.imwrite(str(ROOT / "temp" / "live_annot.png"), vis)
            n += 1
            line = f"[{time.strftime('%H:%M:%S')}] #{n} {summary}"
            print(line)
            log.write(line + "\n")
            log.flush()
        except Exception as e:
            log.write(f"[{time.strftime('%H:%M:%S')}] 异常: {e}\n")
            log.flush()
        time.sleep(args.interval)
    log.close()


if __name__ == "__main__":
    main()
