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
# HUD 排除区（右侧 UI、顶部 HUD、底部技能栏）
EXCLUDE_X = 1230
EXCLUDE_Y_TOP = 240    # v2.13: 顶部 240px 是小地图(0-232)+头像条，无真实血条（旧 60 误检多）
EXCLUDE_Y_BOTTOM = 660

# 自己英雄搜索区（中央偏下）
HERO_REGION = (500, 300, 800, 550)
MIN_HERO_AREA = 300

# v2.13 血条标定（1280x720）：头顶血条满条宽 ~115px（标注帧 90-123 中位）
FULL_BAR_W = 115.0
# 血条中心到英雄位置垂直距离（血条在头顶上方）
HERO_BAR_GAP = 55


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


def _find_bar(head_roi, hue_cond):
    """在头顶区域找指定色相的横条，返回 (ratio, 条长, 区域宽)。

    v2.13：改用连通域找最长水平条（逐行 run 对血条渐变/缺口不鲁棒）。
    """
    if head_roi is None or head_roi.size == 0:
        return None, 0, 0
    hsv = cv2.cvtColor(head_roi, cv2.COLOR_BGR2HSV)
    H, S, V = hsv[..., 0].astype(int), hsv[..., 1].astype(int), hsv[..., 2].astype(int)
    mask = (hue_cond(H, S, V)).astype(np.uint8) * 255
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (11, 1))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
    n, lab, st, cent = cv2.connectedComponentsWithStats(mask, 8)
    best_w, best_x, best_y = 0, 0, 0
    for i in range(1, n):
        w_, h_ = st[i, cv2.CC_STAT_WIDTH], st[i, cv2.CC_STAT_HEIGHT]
        if h_ > 14 or w_ < 12:  # 横条：高 <=14，宽 >=12
            continue
        if w_ > best_w:
            best_w = int(w_)
            best_x, best_y = int(cent[i][0]), int(cent[i][1])
    if best_w > 0:
        return best_w / max(1, mask.shape[1]), best_w, mask.shape[1], (best_x, best_y)
    return None, 0, mask.shape[1], None


def self_hp_mp(frame):
    """检测自己英雄 HP/MP（头顶绿条 + 蓝条）。

    返回 (hp_ratio, mp_ratio, hero_pos) 或 (None, None, None)。

    v2.13 重构（血条驱动）：
      - 旧版用"中央高饱和大块"定位英雄，场景（水域/草丛/金币提示）常误判
        → 血条位置全错，HP/MP 交替缺失
      - 新版：全画面绿条中**最宽的** = 自己血条（队友蓝/敌人红/场景绿短条）
        → HP = 条宽 / 满条宽(115px)；MP = 绿条正下方蓝条
      - 修复旧 ratio 用区域宽(200px)作分母导致满血只显示 0.55 的低估 bug
    """
    h, w = frame.shape[:2]

    def is_green(H, S, V):
        return (H >= 35) & (H <= 90) & (S > 50)

    def is_blue(H, S, V):
        return (H >= 90) & (H <= 135) & (S > 60) & (V > 60)

    greens = _find_bars(frame, is_green, "self")
    if not greens:
        return None, None, None
    # 排除小地图方形区域 (x<240, y<240) 里的绿点/绿圈
    greens = [b for b in greens if not (b["x"] < 240 and b["y"] < 240)]
    # 排除 HUD 区误检：自己英雄镜头跟随一般在画面中部，血条 y>=250
    # （右上角头像/技能图标/金币动画的绿色元素 y<250）
    greens = [b for b in greens if b["y"] >= 250]
    # v3.0: 血条 y<=550（自己英雄在画面中央，血条不会低于 550；
    # 底部按钮区(战队/结算界面)绿色按钮 y 600+ 误检排除）
    greens = [b for b in greens if b["y"] <= 550]
    if not greens:
        return None, None, None
    # 自己血条 = 最宽的绿条（血条 90-123px，场景绿 <60px）
    best = max(greens, key=lambda b: b["w"])
    bx, by, bw = best["x"], best["y"], best["w"]
    hp = min(1.0, bw / FULL_BAR_W)
    hero_pos = (bx, by + HERO_BAR_GAP)

    # MP：绿条正下方 3-25px，同 x ±45px
    mp = None
    y0 = max(0, by + 3)
    y1 = min(h, by + 25)
    if y1 > y0:
        sub = frame[y0:y1, max(0, bx - 45):min(w, bx + 45)]
        mp, mp_len, _w, _c = _find_bar(sub, is_blue)
        if mp is not None and mp_len < 20:
            mp = None
    if hp is not None and bw < 25:
        hp = None
    return hp, mp, hero_pos


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
