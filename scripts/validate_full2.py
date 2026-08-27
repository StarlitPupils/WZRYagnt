# -*- coding: utf-8 -*-
"""新旧 11 类模型对比验证（vs 用户真值 s01-s12，全套不含中央自己）。

用法: python scripts/validate_full2.py <old.pt> <new.pt>
"""
import sys
from pathlib import Path
import numpy as np
import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from wzry.vision.detector import YoloDetector, camp_correct, Det, hero_dedup_ui

CLS_CN = ["敌英", "我英", "敌兵", "我兵", "敌塔", "我塔", "敌晶", "我晶", "野怪", "钩子", "技能"]
CLS_ID = {0: "enemy_hero", 1: "ally_hero", 2: "enemy_minion", 3: "ally_minion",
          4: "enemy_turret", 5: "ally_turret", 6: "enemy_crystal", 7: "ally_crystal",
          8: "neutral_monster", 9: "hook_aim", 10: "skill_effect"}


def iou(a, b):
    ax1, ay1, ax2, ay2 = a; bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    inter = (ix2 - ix1) * (iy2 - iy1)
    return inter / max(1e-6, (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter)


def eval_model(model_path, stats):
    det = YoloDetector(str(model_path), conf=0.25)
    names = det.model.names
    for i in range(1, 13):
        fp = ROOT / "temp" / "ann" / f"s{i:02d}.png"
        gtf = fp.with_suffix(".txt")
        if not fp.exists() or not gtf.exists():
            continue
        img = cv2.imdecode(np.fromfile(str(fp), dtype=np.uint8), cv2.IMREAD_COLOR)
        res = det.model.predict(img, conf=0.25, iou=0.5, device=det.device, verbose=False)[0]
        raw = []
        if res.boxes is not None:
            for box, c, cid in zip(res.boxes.xyxy.cpu().numpy(),
                                   res.boxes.conf.cpu().numpy(),
                                   res.boxes.cls.cpu().numpy().astype(int)):
                raw.append((int(cid), float(c), [float(v) for v in box]))
        corrected = hero_dedup_ui(camp_correct(
            img, [Det(names[c], cf, p, [(p[0] + p[2]) / 2, (p[1] + p[3]) / 2]) for c, cf, p in raw]))
        post_pairs = []
        for d in corrected:
            nid = next(k for k, v in CLS_ID.items() if v == d.cls)
            post_pairs.append((nid, d.xyxy))
        for ln in gtf.read_text(encoding="utf-8").splitlines():
            p = ln.split()
            if len(p) != 5:
                continue
            c = int(p[0])
            if c > 10:
                continue
            x, y, w, h = [float(v) for v in p[1:5]]
            gbox = ((x - w / 2) * 1280, (y - h / 2) * 720, (x + w / 2) * 1280, (y + h / 2) * 720)
            # 跳过中央自己
            cx, cy = (gbox[0] + gbox[2]) / 2 / 1280, (gbox[1] + gbox[3]) / 2 / 720
            area = (gbox[2] - gbox[0]) * (gbox[3] - gbox[1]) / (1280 * 720)
            if 0.3 < cx < 0.7 and 0.25 < cy < 0.75 and area > 0.08:
                continue
            best = None
            for cid_, pbox in post_pairs:
                v = iou(pbox, gbox)
                if v > 0.3 and (best is None or v > best[0]):
                    best = (v, cid_)
            stats[c][0] += 1
            stats[c][1] += int(best is not None and best[1] == c)


def main():
    old_m, new_m = sys.argv[1], sys.argv[2]
    so = {i: [0, 0] for i in range(11)}
    sn = {i: [0, 0] for i in range(11)}
    eval_model(old_m, so)
    eval_model(new_m, sn)
    print(f"{'类':<6}{'旧':>14}{'新':>14}")
    tot_o = [0, 0]; tot_n = [0, 0]
    for c in range(11):
        if so[c][0] or sn[c][0]:
            ot = f"{so[c][1]}/{so[c][0]}" if so[c][0] else "-"
            nt = f"{sn[c][1]}/{sn[c][0]}" if sn[c][0] else "-"
            print(f"{CLS_CN[c]:<6}{ot:>14}{nt:>14}")
        tot_o[0] += so[c][1]; tot_o[1] += so[c][0]
        tot_n[0] += sn[c][1]; tot_n[1] += sn[c][0]
    print(f"{'合计':<6}{f'{tot_o[0]}/{tot_o[1]}={tot_o[0]/max(1,tot_o[1])*100:.0f}%':>14}"
          f"{f'{tot_n[0]}/{tot_n[1]}={tot_n[0]/max(1,tot_n[1])*100:.0f}%':>14}")


if __name__ == "__main__":
    main()
