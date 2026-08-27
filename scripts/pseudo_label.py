# -*- coding: utf-8 -*-
"""全屏伪标注：英雄=血条(蓝→我英/红→敌英)，其余=模型+阵营修正。
输出质检图 + YOLO 标签到 temp/pseudo1/。
"""
import sys
from pathlib import Path
import numpy as np
import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from wzry.vision.detector import YoloDetector, camp_correct
from wzry.vision.self_bars import find_enemy_bars, find_ally_bars, self_hp_mp

CLS_ID = {"enemy_hero": 0, "ally_hero": 1, "enemy_minion": 2, "ally_minion": 3,
          "enemy_turret": 4, "ally_turret": 5, "enemy_crystal": 6, "ally_crystal": 7,
          "neutral_monster": 8, "hook_aim": 9, "skill_effect": 10}
BAR_H = 190   # 条→英雄框高度
BAR_PAD_L, BAR_PAD_R = 25, 45

det = YoloDetector(str(ROOT / "runs" / "detect" / "zhongkui_11cls" / "weights" / "best.pt"), conf=0.2)
src_dir = ROOT / "temp" / "frames2"
dst = ROOT / "temp" / "pseudo1"
(dst / "images").mkdir(parents=True, exist_ok=True)
(dst / "labels").mkdir(parents=True, exist_ok=True)


def self_bar_pos(img):
    """自己绿条位置 (bx, by)。"""
    from wzry.vision.self_bars import _find_bars
    def is_green(H, S, V):
        return (H >= 35) & (H <= 90) & (S > 50)
    greens = [b for b in _find_bars(img, is_green, "self")
              if b["y"] >= 250 and b["y"] <= 550 and not (b["x"] < 240 and b["y"] < 240)]
    if not greens:
        return None
    b = max(greens, key=lambda x: x["w"])
    return b["x"], b["y"]


def has_level_badge(img, x0, y):
    """英雄血条左侧等级徽章：黑底圆 + 白字。伤害数字/技能圈弧无徽章。"""
    xa, xb = max(0, x0 - 36), x0 - 6
    ya, yb = max(0, y - 8), y + 18
    patch = img[ya:yb, xa:xb]
    if patch.size == 0 or patch.shape[1] < 8:
        return False
    hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
    dark = float((hsv[..., 2].astype(int) < 70).mean())
    return dark >= 0.30


n_ok = n_skip = 0
for fp in sorted(src_dir.glob("*.png")):
    img = cv2.imdecode(np.fromfile(str(fp), dtype=np.uint8), cv2.IMREAD_COLOR)
    hp, mp, pos = self_hp_mp(img)
    if hp is None:
        n_skip += 1
        print(fp.name, "非对局帧,跳过")
        continue
    selfb = self_bar_pos(img)
    objs = []
    # 英雄: 蓝条→我英, 红条→敌英
    # 过滤: ① 自己绿条±70/±30 蓝条=自己蓝量条; ② 自己区域红条=自己伤害数字/技能圈;
    #       ③ 无等级徽章者=伤害数字/技能弧/UI碎片
    for b in find_ally_bars(img):
        if b["w"] >= 50 and has_level_badge(img, b["x0"], b["y"]):
            if selfb and abs(b["cx"] - selfb[0]) <= 70 and abs(b["y"] - selfb[1]) <= 30:
                continue
            objs.append((1, (b["x0"] - BAR_PAD_L, b["y"] - 5, b["x1"] + BAR_PAD_R, b["y"] + BAR_H)))
    for b in find_enemy_bars(img):
        if b["w"] >= 50 and has_level_badge(img, b["x0"], b["y"]):
            if selfb and abs(b["cx"] - selfb[0]) <= 100 and abs(b["y"] - selfb[1] - 55) <= 110:
                continue
            objs.append((0, (b["x0"] - BAR_PAD_L, b["y"] - 5, b["x1"] + BAR_PAD_R, b["y"] + BAR_H)))
    # 其余类: 模型 + 阵营修正
    for d in det.detect(img):
        if d.cls in ("enemy_hero", "ally_hero", "hook_aim", "skill_effect"):
            continue
        objs.append((CLS_ID[d.cls], d.xyxy))
    # 写标签
    lines = []
    for cid, (x1, y1, x2, y2) in objs:
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(1280, x2), min(720, y2)
        if x2 <= x1 or y2 <= y1:
            continue
        cx, cy = (x1 + x2) / 2 / 1280, (y1 + y2) / 2 / 720
        w, h = (x2 - x1) / 1280, (y2 - y1) / 720
        lines.append(f"{cid} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
    (dst / "labels" / f"{fp.stem}.txt").write_text("\n".join(lines), encoding="utf-8")
    cv2.imwrite(str(dst / "images" / f"{fp.stem}.png"), img)
    # 质检图
    vis = img.copy()
    for cid, (x1, y1, x2, y2) in objs:
        if cid == 1:
            cv2.rectangle(vis, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 3)
            cv2.putText(vis, "A", (int(x1) + 4, int(y1) + 26), 0, 0.9, (0, 255, 0), 3)
        elif cid == 0:
            cv2.rectangle(vis, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 3)
            cv2.putText(vis, "E", (int(x1) + 4, int(y1) + 26), 0, 0.9, (0, 0, 255), 3)
        else:
            cv2.rectangle(vis, (int(x1), int(y1)), (int(x2), int(y2)), (255, 255, 0), 2)
    (dst / f"qc_{fp.stem}.png").write_bytes(cv2.imencode(".png", vis)[1])
    n_ok += 1
print(f"完成: 标注 {n_ok} 帧, 跳过 {n_skip} 帧")
