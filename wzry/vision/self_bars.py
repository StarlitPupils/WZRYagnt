# -*- coding: utf-8 -*-
"""英雄头顶血条检测（HP/MP/阵营）。

王者荣耀 HUD 语义（用户指导）：
  - 自己英雄头顶血条 = 绿色
  - 队友英雄头顶血条 = 蓝色
  - 敌人英雄头顶血条 = 红色
  - 绿条下方有蓝色细条 = 蓝量（MP）

实现：
  - 全画面找高亮细横条（V>140, 宽25-150, 高<=10）
  - 按颜色分类：绿=自己、蓝=队友、红=敌人
  - 自己 HP = 绿条长度比例；MP = 绿条下方蓝条比例
  - 排除 HUD 边缘（x>1230 右侧 UI、y<60 顶部、y>660 底部）

用法：
    from wzry.vision.self_bars import detect_all_bars
    bars = detect_all_bars(frame)  # {"self": [...], "allies": [...], "enemies": [...]}
    hp, mp = self_hp_mp(frame)
"""
from __future__ import annotations

import cv2
import numpy as np

# 血条特征
BRIGHT_V = 140          # 血条高亮
BAR_W_MIN, BAR_W_MAX = 25, 150
BAR_H_MAX = 10
BAR_AREA_MIN = 40
# HUD 排除区（右侧 UI、顶部、底部）
EXCLUDE_X = 1230
EXCLUDE_Y_TOP = 60
EXCLUDE_Y_BOTTOM = 660

# 自己英雄搜索区（中央偏下）
HERO_REGION = (500, 300, 800, 550)
MIN_HERO_AREA = 300


def _find_bars(frame, hue_cond, label):
    """全画面找指定颜色的高亮细横条。"""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    H, S, V = hsv[..., 0].astype(int), hsv[..., 1].astype(int), hsv[..., 2].astype(int)
    mask = (hue_cond(H, S, V) & (V > BRIGHT_V)).astype(np.uint8) * 255
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (21, 1))
    bars = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
    n, lab, st, cent = cv2.connectedComponentsWithStats(bars, 8)
    found = []
    for i in range(1, n):
        w_, h_ = st[i, cv2.CC_STAT_WIDTH], st[i, cv2.CC_STAT_HEIGHT]
        area = st[i, cv2.CC_STAT_AREA]
        if not (BAR_W_MIN <= w_ <= BAR_W_MAX and h_ <= BAR_H_MAX
                and area >= BAR_AREA_MIN):
            continue
        lx, ly = int(cent[i][0]), int(cent[i][1])
        if lx > EXCLUDE_X or ly < EXCLUDE_Y_TOP or ly > EXCLUDE_Y_BOTTOM:
            continue
        found.append({"x": lx, "y": ly, "w": w_, "h": h_, "ratio": w_ / 150.0})
    found.sort(key=lambda b: (b["y"], b["x"]))
    return found


def detect_all_bars(frame):
    """检测所有英雄血条：绿=自己、蓝=队友、红=敌人。

    蓝条（队友）过滤：
      - 宽度 <= 80（场景/UI 宽条误检排除）
      - 远离自己绿条（自己蓝量条不算队友）
    """
    def is_green(H, S, V):
        return (H >= 35) & (H <= 90) & (S > 50)

    def is_blue(H, S, V):
        return (H >= 90) & (H <= 135) & (S > 60)

    def is_red(H, S, V):
        return ((H <= 15) | (H >= 165)) & (S > 60)

    result = {
        "self": _find_bars(frame, is_green, "self"),
        "allies": _find_bars(frame, is_blue, "ally"),
        "enemies": _find_bars(frame, is_red, "enemy"),
    }
    # 蓝条过滤：宽条（>80）排除；自己绿条旁 60px 内的蓝条=自己蓝量，排除
    if result["self"]:
        self_bars = result["self"]
        my_x = self_bars[0]["x"] if self_bars else None
        my_y = self_bars[0]["y"] if self_bars else None
        if my_x is not None:
            result["allies"] = [
                b for b in result["allies"]
                if b["w"] <= 80
                and not (abs(b["x"] - my_x) < 60 and abs(b["y"] - my_y) < 40)
            ]
    else:
        result["allies"] = [b for b in result["allies"] if b["w"] <= 80]
    return result


def _find_self_hero(frame):
    """中央区域找自己英雄（高饱和大块），返回 (cx, cy) 或 None。"""
    x0, y0, x1, y1 = HERO_REGION
    roi = frame[y0:y1, x0:x1]
    if roi.size == 0:
        return None
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    sat = (hsv[..., 1] > 80).astype(np.uint8)
    n, lab, st, cent = cv2.connectedComponentsWithStats(sat, 8)
    best = None
    for i in range(1, n):
        area = st[i, cv2.CC_STAT_AREA]
        if area >= MIN_HERO_AREA and (best is None or area > best[0]):
            best = (area, int(cent[i][0]) + x0, int(cent[i][1]) + y0)
    return (best[1], best[2]) if best else None


def _find_bar(head_roi, hue_cond):
    """在头顶区域找指定色相的横条，返回 (ratio, 条长, 区域宽)。"""
    if head_roi is None or head_roi.size == 0:
        return None, 0, 0
    hsv = cv2.cvtColor(head_roi, cv2.COLOR_BGR2HSV)
    H, S, V = hsv[..., 0].astype(int), hsv[..., 1].astype(int), hsv[..., 2].astype(int)
    mask = hue_cond(H, S, V)
    best_len = 0
    for ry in range(mask.shape[0]):
        row = mask[ry]
        run = 0
        for x in range(row.shape[0]):
            if row[x]:
                run += 1
                best_len = max(best_len, run)
            else:
                run = 0
    if best_len > 0:
        return best_len / max(1, mask.shape[1]), best_len, mask.shape[1]
    return None, 0, mask.shape[1]


def self_hp_mp(frame):
    """检测自己英雄 HP/MP（头顶绿条 + 蓝条）。

    返回 (hp_ratio, mp_ratio, hero_pos) 或 (None, None, None)。
    """
    hero = _find_self_hero(frame)
    if hero is None:
        return None, None, None
    cx, cy = hero
    h, w = frame.shape[:2]
    y0 = max(0, cy - 160)
    y1 = max(0, cy - 100)
    x0 = max(0, cx - 100)
    x1 = min(w, cx + 100)
    if y1 <= y0:
        return None, None, hero
    head = frame[y0:y1, x0:x1]

    def is_green(H, S, V):
        return (H >= 35) & (H <= 90) & (S > 60) & (V > 60)

    def is_blue(H, S, V):
        return (H >= 90) & (H <= 135) & (S > 60) & (V > 60)

    hp, hp_len, _ = _find_bar(head, is_green)
    mp, mp_len, _ = _find_bar(head, is_blue)
    if hp_len < 25:
        hp = None
    if mp_len < 25:
        mp = None
    return hp, mp, hero


# 兼容旧接口
def detect_self_bars(frame):
    return self_hp_mp(frame)


if __name__ == "__main__":
    import numpy as np
    for f in ["s01.png", "s05.png", "s08.png", "s11.png"]:
        data = np.fromfile(rf"E:\WZRYagent\temp\ann\{f}", dtype=np.uint8)
        img = cv2.imdecode(data, cv2.IMREAD_COLOR)
        if img is None:
            print(f"{f}: 无法读取")
            continue
        hp, mp, pos = self_hp_mp(img)
        bars = detect_all_bars(img)
        print(f"{f}: HP={hp} MP={mp} 自己绿条{len(bars['self'])} "
              f"队友蓝条{len(bars['allies'])} 敌人红条{len(bars['enemies'])}")
        if bars["self"]:
            print(f"  自己绿条: {[(b['x'], b['y'], b['w']) for b in bars['self'][:3]]}")
        if bars["enemies"]:
            print(f"  敌人红条: {[(b['x'], b['y'], b['w']) for b in bars['enemies'][:3]]}")
