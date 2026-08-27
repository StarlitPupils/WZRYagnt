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
BAR_H_MIN = 5           # v2.36 野怪/小怪血条很细(1-4px), 英雄血条粗(5-9px) -> 下限过滤
BAR_AREA_MIN = 40
# HUD 排除区（右侧 UI、顶部 HUD、底部技能栏）
EXCLUDE_X = 1230
EXCLUDE_Y_TOP = 240    # v2.13: 顶部 240px 是小地图(0-232)+头像条，无真实血条（旧 60 误检多）
EXCLUDE_Y_BOTTOM = 660

# 自己英雄搜索区（中央偏下）
HERO_REGION = (500, 300, 800, 550)
MIN_HERO_AREA = 300


def hero_bar_check(frame, cx, cy, half_w=60, y_span=40):
    """v2.75 敌英血条厚度校验: (cx,cy)附近找红条, 返回最高条高(px)。
    >=BAR_H_MIN(5)=英雄; 1-4px=野怪/小兵 -> 由调用方剔除。无条返回 None。"""
    try:
        x0 = max(0, int(cx) - half_w)
        x1 = min(frame.shape[1], int(cx) + half_w)
        y0 = max(0, int(cy))
        y1 = min(frame.shape[0], int(cy) + y_span)
        roi = frame[y0:y1, x0:x1]
        if roi.size == 0:
            return None
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        S, V = hsv[..., 1].astype(int), hsv[..., 2].astype(int)
        m = ((S > 60) & (V > 100)).astype(np.uint8)
        n, lab, st, _ = cv2.connectedComponentsWithStats(m, 8)
        best = None
        for i in range(1, n):
            x, y, w2, h2, area = st[i]
            # 红条: 宽 10-200, 高 1-12, 长宽比>=3(横条), 红色(R>G*1.4)
            if not (10 <= w2 <= 200 and 1 <= h2 <= 12 and w2 >= 3 * h2):
                continue
            seg = roi[y:y + h2, x:x + w2]
            rmean = float(seg[..., 2].mean())
            gmean = float(seg[..., 1].mean())
            if rmean < gmean * 1.3:
                continue
            if best is None or h2 > best:
                best = h2
        return best
    except Exception:
        return None

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
        if not (BAR_W_MIN <= w_ <= BAR_W_MAX and BAR_H_MIN <= h_ <= BAR_H_MAX
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


def find_color_bars(frame, is_col, y_top=70, y_bot=635, x_max=1230,
                    wmin=25, wmax=150, gap=9, exclude_mm=True, thick_min=4, thick_max=14):
    """找指定颜色的分段血条（刻度将条切成多段 → 行内 gap 合并 + 行间堆叠成条）。

    返回 [{x0, x1, y, w, cx}]，y 为条顶部行，w 为最大行宽。
    v2.36: thick 过滤——野怪/小怪血条很细(2-3px), 英雄血条较粗(5-9px)
           条厚 < thick_min 或 > thick_max 的条剔除(野怪白给, 英雄误杀).
    排除：小地图区(x<245,y<245)、右上/右下 UI(x>=x_max)、底部(y>y_bot)。
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    H, S, V = hsv[..., 0].astype(int), hsv[..., 1].astype(int), hsv[..., 2].astype(int)
    mask = is_col(H, S, V).astype(np.uint8)
    rows = []  # (y, x0, x1)
    for y in range(y_top, y_bot):
        if exclude_mm:
            row = mask[y, :x_max].copy()
            if y < 245:
                row[:245] = 0
        else:
            row = mask[y, :x_max]
        idx = np.nonzero(row)[0]
        if len(idx) < wmin // 2:
            continue
        groups = []
        start = prev = idx[0]
        for i in idx[1:]:
            if i - prev > gap:
                if prev - start + 1 >= wmin:
                    groups.append((start, prev))
                start = i
            prev = i
        if prev - start + 1 >= wmin:
            groups.append((start, prev))
        for g0, g1 in groups:
            if g1 - g0 + 1 <= wmax:
                rows.append((y, int(g0), int(g1)))
    if not rows:
        return []
    # 行间堆叠：y 差<=2 且 x 区间重叠 -> 同一条
    rows.sort()
    bars = []
    for y, x0, x1 in rows:
        placed = False
        for b in bars:
            if y - b["yb"] <= 2 and not (x1 < b["x0"] - 10 or x0 > b["x1"] + 10):
                b["yb"] = y
                b["x0"] = min(b["x0"], x0)
                b["x1"] = max(b["x1"], x1)
                placed = True
                break
        if not placed:
            bars.append({"y": y, "yb": y, "x0": x0, "x1": x1})
    out = []
    for b in bars:
        w = b["x1"] - b["x0"] + 1
        thick = b["yb"] - b["y"] + 1
        if w < wmin or thick < thick_min or thick > thick_max:
            continue
        out.append({"x0": b["x0"], "x1": b["x1"], "y": b["y"],
                    "w": w, "cx": (b["x0"] + b["x1"]) // 2, "thick": thick})
    return out


def find_ally_bars(frame):
    """队友蓝条（分段合并版）。"""
    return find_color_bars(frame, lambda H, S, V: (H >= 90) & (H <= 135) & (S > 55) & (V > 60))


def find_enemy_bars(frame):
    """敌人红条（分段合并版）。"""
    return find_color_bars(frame, lambda H, S, V: ((H <= 15) | (H >= 165)) & (S > 60) & (V > 60))
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

    # MP：绿条正下方 3-25px。蓝条左端与 HP 条左端对齐、填充连续；
    # 用"最右蓝色列位置"换算抗遮挡（金币飘字/血条被截断时左段法会低估）
    mp = None
    y0 = max(0, by + 3)
    y1 = min(h, by + 27)
    if y1 > y0:
        bar_left = bx - bw // 2
        x_lo = max(0, bar_left - 2)
        x_hi = min(w, bar_left + int(FULL_BAR_W) + 4)
        if x_hi > x_lo:
            sub = frame[y0:y1, x_lo:x_hi]
            hsv2 = cv2.cvtColor(sub, cv2.COLOR_BGR2HSV)
            H2, S2, V2 = hsv2[..., 0].astype(int), hsv2[..., 1].astype(int), hsv2[..., 2].astype(int)
            m2 = ((H2 >= 90) & (H2 <= 135) & (S2 > 60) & (V2 > 60))
            cols = m2.any(axis=0)
            if cols.any():
                last = int(np.nonzero(cols)[0].max())
                if last >= 12:
                    mp = min(1.0, (last + 1) / FULL_BAR_W)
                else:
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
