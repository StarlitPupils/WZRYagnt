# -*- coding: utf-8 -*-
"""顶部状态条检测 v2（理解层）：队友头像+血条 / 右上角敌方。

真机标定（示范局 1280x720）：
  - 队友头像区：x230-580, y0-160，4 个头像横向排列
    （实测 y≈30-35，x≈259/309/363/421，间距~55，头像约 38x35）
  - 每个头像下方 y≈53 有血条（42x3 横条）
  - 检测用高亮（V>150）连通域 + 位置规律
"""
from __future__ import annotations

import cv2
import numpy as np

# 队友头像搜索区（真机标定）
MATE_REGION = (230, 0, 580, 160)
# 头像尺寸范围
MATE_AREA_MIN = 150
MATE_AREA_MAX = 2000


def detect_teammates(frame, region=MATE_REGION):
    """定位顶部队友头像条（高亮圆形头像 + 下方血条）。

    返回 [{"x","y","w","h","avatar_img","hp_ratio","hp_bar"}, ...]
    """
    x0, y0, x1, y1 = region
    roi = frame[y0:y1, x0:x1]
    if roi.size == 0:
        return []
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    V = hsv[..., 2].astype(np.uint8)
    # 高亮（头像亮区）
    bright = (V > 150).astype(np.uint8)
    bright = cv2.morphologyEx(bright, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    n, lab, st, cent = cv2.connectedComponentsWithStats(bright, 8)
    cands = []
    for i in range(1, n):
        area = st[i, cv2.CC_STAT_AREA]
        w_, h_ = st[i, cv2.CC_STAT_WIDTH], st[i, cv2.CC_STAT_HEIGHT]
        if not (MATE_AREA_MIN <= area <= MATE_AREA_MAX):
            continue
        comp = area / max(1.0, w_ * h_)
        if comp < 0.5:
            continue
        lx, ly = int(cent[i][0]) + x0, int(cent[i][1]) + y0
        # v2：头像在 y<80（排除"全队"按钮 y≈120）
        if ly > 80:
            continue
        cands.append({"x": lx, "y": ly, "w": w_, "h": h_, "area": area})

    # 聚类：头像横向排列（y 相近），按 x 排序，最多 5 个
    cands.sort(key=lambda c: c["x"])
    mates = []
    for c in cands:
        if mates and abs(c["y"] - mates[-1]["y"]) < 25 and \
                c["x"] - mates[-1]["x"] < 25:
            continue  # 同一头像的重复块
        mates.append(c)
        if len(mates) >= 5:
            break

    # v2：头像固定间距 ~55px（真机标定），补齐漏检
    if mates:
        # 用第一个头像的 y 作为基准，x 按间距补全
        base_y = mates[0]["y"]
        x_positions = []
        for m in mates:
            if abs(m["y"] - base_y) < 25:
                x_positions.append(m["x"])
        if x_positions:
            x_positions.sort()
            # 从最小 x 开始按 55px 间距生成 4 个位置
            full = []
            x_min = x_positions[0]
            for k in range(4):
                full.append(x_min + k * 55)
            # 用检测到的 x 修正基准（取最接近检测的）
            mates = []
            for k, fx in enumerate(full):
                near = min(x_positions, key=lambda x: abs(x - fx))
                mates.append({
                    "x": fx, "y": base_y,
                    "w": 45, "h": 45,
                    "area": 0, "_near": near,
                })

    out = []
    for m in mates:
        pad = 6
        x0c, y0c = max(0, m["x"] - m["w"] // 2 - pad), max(0, m["y"] - m["h"] // 2 - pad)
        x1c, y1c = m["x"] + m["w"] // 2 + pad, m["y"] + m["h"] // 2 + pad
        avatar = frame[y0c:y1c, x0c:x1c].copy() if (x1c > x0c and y1c > y0c) else None
        hp_bar, hp_ratio = _detect_hp_below(frame, m["x"], m["y"] + m["h"] // 2 + pad)
        out.append({
            "x": m["x"], "y": m["y"], "w": m["w"], "h": m["h"],
            "avatar_img": avatar, "hp_bar": hp_bar, "hp_ratio": hp_ratio,
        })
    return out


def _detect_hp_below(frame, cx, start_y, max_h=20, max_w=80):
    """头像下方找血条（亮色横条），返回 (bar_bbox, ratio)。"""
    h, w = frame.shape[:2]
    if start_y >= h or cx < 5 or cx >= w:
        return None, None
    x0 = max(0, cx - 30)
    x1 = min(w, cx + 50)
    roi = frame[start_y:min(h, start_y + max_h), x0:x1]
    if roi.size == 0:
        return None, None
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    # 找最亮的横条行
    best = None
    for ry in range(roi.shape[0]):
        row = gray[ry]
        run = 0
        best_run = 0
        for x in range(row.shape[0]):
            if row[x] > 120:
                run += 1
                best_run = max(best_run, run)
            else:
                run = 0
        if best_run >= 20 and (best is None or best_run > best[0]):
            best = (best_run, ry)
    if best is None:
        return None, None
    bar_len, ry = best
    return (x0, start_y + ry, bar_len, 2), bar_len / max(1.0, x1 - x0)


def detect_enemies(frame, region=(900, 0, 1280, 160)):
    """右上角敌方信息（占位：红色圆形头像检测）。"""
    x0, y0, x1, y1 = region
    roi = frame[y0:y1, x0:x1]
    if roi.size == 0:
        return []
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    red = ((hsv[..., 0] <= 15) | (hsv[..., 0] >= 165)) & (hsv[..., 1] > 60)
    red = cv2.morphologyEx(red.astype(np.uint8), cv2.MORPH_CLOSE,
                           np.ones((7, 7), np.uint8))
    n, lab, st, cent = cv2.connectedComponentsWithStats(red, 8)
    enemies = []
    for i in range(1, n):
        area = st[i, cv2.CC_STAT_AREA]
        w_, h_ = st[i, cv2.CC_STAT_WIDTH], st[i, cv2.CC_STAT_HEIGHT]
        if not (100 <= area <= 3000):
            continue
        comp = area / max(1.0, w_ * h_)
        if comp < 0.4:
            continue
        lx, ly = int(cent[i][0]) + x0, int(cent[i][1]) + y0
        enemies.append({"x": lx, "y": ly, "w": w_, "h": h_})
    enemies.sort(key=lambda e: e["x"])
    return enemies


if __name__ == "__main__":
    import numpy as np
    for f in ["s01.png", "s02.png", "s03.png", "s05.png"]:
        data = np.fromfile(rf"E:\WZRYagent\temp\ann\{f}", dtype=np.uint8)
        img = cv2.imdecode(data, cv2.IMREAD_COLOR)
        if img is None:
            print(f"{f}: 无法读取")
            continue
        mates = detect_teammates(img)
        print(f"{f}: {len(mates)} 个队友 "
              f"{[(m['x'], m['y'], m['w'], m['h'], round(m['hp_ratio'], 2) if m['hp_ratio'] else None) for m in mates]}")
