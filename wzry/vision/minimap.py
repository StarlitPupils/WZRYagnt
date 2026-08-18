# -*- coding: utf-8 -*-
"""王者荣耀小地图感知模块（纯 OpenCV 视觉，不依赖 YOLO / ultralytics）。

功能
----
1. ``find_minimap(frame)``    自适应定位左上角小地图圆盘（深色圆盘 + 蓝/红/黄圆点）。
   不依赖固定坐标：在画面左上 40% 区域做多方法扫描并投票：
     - 方法 A：深色圆盘 + 圆环对比度 + 盘内"稀疏彩色圆点"打分（filter2D 圆核，快）；
     - 方法 B：霍夫圆变换候选，再用同一打分函数复评；
     - 方法 C：亮环/暗心（部分版本小地图有亮色描边）。
2. ``detect_dots(frame, minimap)`` 在圆盘内按 HSV 阈值提取蓝/红/黄圆点：
   形态学开运算去噪 -> 连通域聚类取质心 -> 按点大小分类（英雄 / 塔点 / 噪声），
   输出以圆盘为参考系的归一化坐标（圆心 = 地图中心，(0,0)-(1,1) 为圆盘外接正方形），
   并标记是否落在圆内（圆形裁切正方形地图，圆外为无效区）。
3. ``analyze(frame)``           组合入口。
4. 模块级 ``if __name__ == "__main__"`` 演示：单张图片或视频抽样。

输出结构（analyze / detect_dots）
---------------------------------
{
  "found": bool,
  "center": [cx, cy],          # 窗口像素
  "radius": r,                 # 窗口像素
  "method": str,               # 定位命中的方法名
  "score": float,              # 定位置信分（仅参考）
  "dots": {
     "blue":   [[nx, ny], ...],  # 蓝方英雄圆点（归一化）
     "red":    [[nx, ny], ...],  # 红方英雄圆点（归一化）
     "yellow": [[nx, ny], ...],  # 中立单位圆点（归一化）
  },
  "towers": [[nx, ny], ...],   # 塔点（略小于英雄点，颜色见 detail）
  "detail": {                  # 调试/后处理用明细
     "blue":   [{"n": [nx, ny], "px": [x, y], "area": a, "valid": bool}, ...],
     "red":    [...],
     "yellow": [...],
     "towers": [{"n": [nx, ny], "px": [x, y], "area": a, "color": "b|r|y", "valid": bool}, ...],
  },
}

数值均为窗口像素坐标（center/radius/px）或相对圆盘外接正方形的归一化坐标（n）。
归一化：nx = (x - (cx - r)) / (2r)，ny = (y - (cy - r)) / (2r)；
圆盘圆心即地图中心（nx=0.5, ny=0.5）；圆形区域外的无效标记：valid=False
（即 (nx-0.5)^2 + (ny-0.5)^2 > 0.25，圆盘边缘的圆点可能略微越界，仍保留但标记无效）。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# 常量 / 默认参数
# ---------------------------------------------------------------------------

# HSV 颜色阈值（OpenCV H: 0-180）
# v2.8 语义（用户实测指导）：
#   green = 自己（绿色圈/箭头）
#   blue  = 队友英雄（蓝色圈）+ 我方小兵（移动小蓝点）
#   red   = 敌方英雄（红色圈）+ 敌方小兵（移动小红点）
#   yellow= 野怪（黄色点）
#   塔：蓝/红方块（我方/敌方防御塔，在 towers 中带 color）
COLOR_THRESH = {
    # 队友/我方单位（蓝圈/蓝点）
    "blue": {"h_lo": 95, "h_hi": 135, "s_min": 60, "v_min": 50},
    # 敌方单位（红圈/红点）：H 红 = 0 附近与 170-180 两段
    "red": {"h_lo": 0, "h_hi": 10, "h2_lo": 170, "h2_hi": 180, "s_min": 70, "v_min": 50},
    # 野怪（黄色点）
    "yellow": {"h_lo": 15, "h_hi": 38, "s_min": 80, "v_min": 60},
    # 自己（绿色圈/箭头）：真机标定 HSV=(55,182,196)（鲜艳绿，H 35-90）
    "green": {"h_lo": 35, "h_hi": 90, "s_min": 100, "v_min": 100},
}

# 定位搜索范围：默认左上 40% 区域（任务要求；可通过 search_region 覆盖）
DEFAULT_SEARCH = (0.0, 0.0, 0.40, 0.40)  # (x0, y0, x1, y1) 相对比例

# 圆盘半径搜索范围
R_MIN_FRAC = 0.05           # 相对短边
R_MIN_W_FRAC = 0.045        # 相对画面宽度（王者荣耀小地图半径约 0.07-0.10 宽；下限放宽到 0.045 以兼容小窗口）
R_MAX_FRAC = 0.32           # 相对短边

# 圆盘判据（minimap 的特征）
DISK_GRAY_MAX = 80.0         # 盘内平均灰度上限（深色圆盘；实测场景补丁 65-96，真实圆盘 45-65）
RING_CONTRAST_POS_MIN = 12.0 # 圆环(盘外)-盘内平均灰度差下限（小地图有亮描边；场景补丁实测 -21~+10）
COLOR_FRAC_MIN = 0.0004      # 盘内彩色像素占比下限（必须同时有蓝和红才可能是小地图）
COLOR_FRAC_MAX = 0.35        # 盘内彩色占比上限（超过则更像是场景而非小地图）
BLUE_FRAC_SCENE = 0.22       # 盘内蓝色占比超过该值判为"蓝方基地场景"，扣分
FOUND_SCORE_MIN = 45.0       # 判定"找到"的最低综合分

# 圆点尺寸（以圆盘半径 r 为基准的半径比例）。
# 英雄点（小圆点）> 塔点（略小）> 噪声（极小）；三个区间不重叠：
#   tower: [0.014, 0.030) * r，hero: [0.032, 0.090] * r
FRAC_NOISE_MAX = 0.014       # 半径 < 1.4% r 视为噪声
FRAC_TOWER_LO = 0.014        # 塔点半径下界
FRAC_HERO_LO = 0.032         # 英雄点半径下界
FRAC_HERO_HI = 0.090         # 英雄点半径上界（再大视为误检）
MIN_HERO_DOTS = 1            # 验证时每阵营最少英雄级圆点数（阵亡/1v1 时可能只剩 1 个）

# 形态学核
MORPH_K = 3


# ---------------------------------------------------------------------------
# 内部工具
# ---------------------------------------------------------------------------

def _disk_kernel(r: int, pad: int = 12) -> np.ndarray:
    size = 2 * (r + pad) + 1
    k = np.zeros((size, size), np.float32)
    c = r + pad
    yy, xx = np.mgrid[0:size, 0:size]
    k[np.hypot(xx - c, yy - c) <= r] = 1.0
    return k


def _ring_kernel(r: int, pad: int = 12, thick: int = 6) -> np.ndarray:
    size = 2 * (r + pad) + 1
    k = np.zeros((size, size), np.float32)
    c = r + pad
    yy, xx = np.mgrid[0:size, 0:size]
    d = np.hypot(xx - c, yy - c)
    k[(d > r) & (d <= r + thick)] = 1.0
    return k


def _color_masks(frame: np.ndarray):
    """返回 {"blue": np.uint8 mask, "red": ..., "yellow": ...}（与 frame 同尺寸）。"""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    H, S, V = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    masks = {}
    for name, t in COLOR_THRESH.items():
        m = (H >= t["h_lo"]) & (H <= t["h_hi"]) & (S >= t["s_min"]) & (V >= t["v_min"])
        if "h2_lo" in t:
            m = m | ((H >= t["h2_lo"]) & (H <= t["h2_hi"]))
        masks[name] = m.astype(np.uint8) * 255
    return masks


def _disk_metrics(gray_f, masks_f, cx, cy, r, n_sectors=8):
    """计算候选圆盘的一组度量（基于圆形 mask，像素均值；仅在本地区域内计算）。

    返回 dict（越界返回 None）：
      g_in / g_ring / b_in / r_in / y_in : 盘内灰度与颜色占比、圆环灰度
      sector_contrast : 把圆环按角度均分 n_sectors 个扇区，返回每个扇区
                        |扇区灰度 - 盘内灰度| 的列表（用于圆边界一致性判别）
      ring_consistent : 扇区对比度 >= 5.0 的扇区数 >= 5 时为 True
    """
    h, w = gray_f.shape
    if cx - r < 2 or cy - r < 2 or cx + r >= w - 2 or cy + r >= h - 2:
        return None
    # 只在圆盘外接框内计算，避免整帧大数组操作
    pad = 8
    x0 = max(0, int(cx - r - pad)); y0 = max(0, int(cy - r - pad))
    x1 = min(w, int(cx + r + pad)); y1 = min(h, int(cy + r + pad))
    rr = int(round(r))
    lcx, lcy = int(cx) - x0, int(cy) - y0

    g = gray_f[y0:y1, x0:x1]
    disk = np.zeros((y1 - y0, x1 - x0), np.uint8)
    cv2.circle(disk, (lcx, lcy), rr, 255, -1)
    ring = np.zeros((y1 - y0, x1 - x0), np.uint8)
    cv2.circle(ring, (lcx, lcy), rr + 6, 255, 2)
    ring[disk == 255] = 0

    ins = disk == 255
    g_in = float(g[ins].mean())
    ring_pts = ring == 255
    g_ring = float(g[ring_pts].mean()) if ring_pts.any() else g_in
    # masks_f 为 0-255 尺度，这里归一化到 0-1（与 COLOR_FRAC_* 阈值一致）
    b_in = float(masks_f["blue"][y0:y1, x0:x1][ins].mean()) / 255.0
    r_in = float(masks_f["red"][y0:y1, x0:x1][ins].mean()) / 255.0
    y_in = float(masks_f["yellow"][y0:y1, x0:x1][ins].mean()) / 255.0

    # 扇区对比度（圆边界一致性）
    sector_contrast = []
    if ring_pts.any():
        ys, xs = np.nonzero(ring_pts)
        ang = np.arctan2(ys - lcy, xs - lcx)
        sec = ((ang / (2 * np.pi)) * n_sectors).astype(int) % n_sectors
        for k in range(n_sectors):
            sel = sec == k
            if sel.any():
                sector_contrast.append(abs(float(g[ys[sel], xs[sel]].mean()) - g_in))
    ring_consistent = (sum(1 for v in sector_contrast if v >= 5.0) >= max(4, n_sectors - 3)
                       if sector_contrast else False)
    return {"g_in": g_in, "g_ring": g_ring, "b_in": b_in, "r_in": r_in, "y_in": y_in,
            "r": float(r), "sector_contrast": sector_contrast, "ring_consistent": ring_consistent}


def _score_disk(m, short=None):
    """把 _disk_metrics 的输出转成分数。分数高 => 更像小地图圆盘。

    short : 画面短边（像素），提供时加入"半径/短边比"尺寸偏好惩罚。
    """
    g_in, g_ring = m["g_in"], m["g_ring"]
    b_in, r_in, y_in = m["b_in"], m["r_in"], m["y_in"]
    col = b_in + r_in + y_in
    score = 0.0
    # 1) 盘内应偏暗
    if g_in < DISK_GRAY_MAX:
        score += (DISK_GRAY_MAX - g_in) * 0.8
    # 2) 盘外与盘内应有可辨识边界（亮环或暗环）
    score += min(abs(g_ring - g_in), 40.0) * 0.5
    # 3) 必须同时含蓝、红圆点（这是小地图最强的判别信号）
    if b_in >= COLOR_FRAC_MIN and r_in >= COLOR_FRAC_MIN:
        score += 60.0
    else:
        score -= 40.0
    # 4) 黄色中立点加分
    if y_in >= COLOR_FRAC_MIN * 0.75:
        score += 12.0
    # 5) 彩色占比过高的区域更像场景
    if col > COLOR_FRAC_MAX:
        score -= 50.0
    # 6) 蓝色占绝对主导 => 蓝方基地场景
    if b_in > BLUE_FRAC_SCENE:
        score -= 35.0
    # 7) 圆边界一致性（强判别信号；粗扫阶段无扇区信息时为中性）
    if m.get("ring_consistent") is not None:
        score += 30.0 if m["ring_consistent"] else -30.0
    # 8) 尺寸偏好：半径/短边比应在 [0.09, 0.22]（1280x720 下约 0.146），区间外扣分
    if short and m.get("r"):
        frac = m["r"] / float(short)
        if not (0.09 <= frac <= 0.22):
            score -= 25.0
            if frac < 0.05 or frac > 0.30:
                score -= 20.0
    return float(score)


def _hough_candidates(gray, x1, y1, r_lo, r_hi):
    """在搜索区域内用霍夫圆变换找候选圆，返回 [(cx, cy, r), ...]。"""
    reg = gray[:y1, :x1]
    blur = cv2.GaussianBlur(reg, (5, 5), 0)
    out = []
    try:
        circles = cv2.HoughCircles(
            blur, cv2.HOUGH_GRADIENT, dp=1.2, minDist=max(30, r_hi // 2),
            param1=100, param2=28, minRadius=r_lo, maxRadius=r_hi,
        )
    except cv2.error:
        circles = None
    if circles is not None:
        for (x, y, r) in np.round(circles[0]).astype(int):
            out.append((x, y, r))
    return out


def _dot_structure(masks_uint8, cx, cy, r):
    """统计盘内各颜色"紧凑圆点"（连通域面积落在塔点~英雄点区间）。

    返回 {"blue": [(area, x, y), ...], "red": [...], "yellow": [...]}，
    (x, y) 为圆点质心（与输入 mask 同坐标系）。
    这是排除"场景色块"的关键：蓝方基地里虽然蓝/红像素很多，但都是弥散的
    纹理/噪点，经形态学开运算后不会留下紧凑圆点；小地图上的英雄/塔点是
    小而圆的连通域，会留下 1~N 个圆点。
    """
    h, w = masks_uint8["blue"].shape
    pad = 6
    x0 = max(0, int(cx - r - pad)); y0 = max(0, int(cy - r - pad))
    x1 = min(w, int(cx + r + pad)); y1 = min(h, int(cy + r + pad))
    if x1 - x0 < 6 or y1 - y0 < 6:
        return {"blue": [], "red": [], "yellow": []}
    disk = np.zeros((y1 - y0, x1 - x0), np.uint8)
    cv2.circle(disk, (int(cx) - x0, int(cy) - y0), int(r), 255, -1)
    disk = cv2.dilate(disk, np.ones((3, 3), np.uint8))
    area_lo = np.pi * (FRAC_TOWER_LO * r) ** 2
    area_hi = np.pi * (FRAC_HERO_HI * r) ** 2
    out = {}
    for color in ("blue", "red", "yellow"):
        m = masks_uint8[color][y0:y1, x0:x1] & disk
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((MORPH_K, MORPH_K), np.uint8))
        n, lab, st, cent = cv2.connectedComponentsWithStats(m, 8)
        out[color] = [(int(st[i, cv2.CC_STAT_AREA]),
                       float(cent[i][0]) + x0, float(cent[i][1]) + y0)
                      for i in range(1, n)
                      if area_lo <= st[i, cv2.CC_STAT_AREA] <= area_hi]
    return out


def _verify_dots(dots, cx, cy, r):
    """圆点结构最终验证，返回 (ok, nb, nr, ny, mean_dist_frac)。

    ok 需要满足：
      1) 蓝方、红方各有 MIN_HERO_DOTS(默认2)~5 个"英雄级"圆点
         （英雄点略大于塔点；5v5 对局双方各 5 名英雄始终可见，贴脸重叠
         合并后一般仍 >=2）；
      2) 英雄级圆点的平均距心距离 <= 0.80*r —— 排除"候选圆贴住屏幕顶部
         UI 图标条/盘缘图标"的场景圆（图标全挤在圆盘上缘）；
      3) 距心距离 > 0.85*r 的盘缘圆点占比 <= 0.6 —— 排除"少量圆点几乎
         全部挂在圆盘边缘"的假阳性（真实小地图的英雄点散布在盘内各处）。
    """
    hero_lo = np.pi * (FRAC_HERO_LO * r) ** 2
    hero_hi = np.pi * (FRAC_HERO_HI * r) ** 2
    heros = {"blue": [], "red": [], "yellow": []}
    for color in ("blue", "red", "yellow"):
        heros[color] = [(a, x, y) for (a, x, y) in dots[color]
                        if hero_lo <= a <= hero_hi]
    nb, nr, ny = len(heros["blue"]), len(heros["red"]), len(heros["yellow"])

    pts = heros["blue"] + heros["red"] + heros["yellow"]
    mean_dist_frac = None
    rim_frac = None
    if pts:
        dists = [np.hypot(x - cx, y - cy) for (_, x, y) in pts]
        mean_dist_frac = float(np.mean(dists) / max(1.0, r))
        rim_frac = float(np.mean([d > 0.85 * r for d in dists]))

    ok = (MIN_HERO_DOTS <= nb <= 5 and MIN_HERO_DOTS <= nr <= 5
          and mean_dist_frac is not None and mean_dist_frac <= 0.90
          and rim_frac is not None and rim_frac <= 0.60)
    return ok, nb, nr, ny, mean_dist_frac


def _dot_bonus(nb, nr, ny):
    """圆点结构加分：蓝/红各至少 1 个紧凑圆点才算小地图；各 >=2 个更可信。"""
    if nb >= 2 and nr >= 2:
        b = 45.0
    elif nb >= 1 and nr >= 1:
        b = 20.0
    else:
        b = -40.0
    return b + min(ny, 3) * 5.0


# ---------------------------------------------------------------------------
# 公开 API
# ---------------------------------------------------------------------------

def find_minimap(frame, search_region=None, prior=None, prior_r=None,
                 debug=False, fast_prior=False):
    """自适应定位左上角小地图圆盘。

    Parameters
    ----------
    frame : np.ndarray
        BGR 帧（任意分辨率）。
    search_region : tuple | None
        搜索区域 (x0, y0, x1, y1) 相对比例；默认 (0, 0, 0.40, 0.40)。
    prior : tuple | None
        可选的先验圆心 (px, py)；提供时作为精修种子之一（不提供则完全自适应）。
    prior_r : int | None
        先验半径（跟踪模式使用）；与 fast_prior 搭配可只扫 r±10。
    fast_prior : bool
        True 且提供 prior 时：跳过全图扫描，固定先验圆心做"半径扫描(步长2) +
        圆心微调 ±4"，单帧约 30-80ms（小地图在 UI 上位置固定，先验圆心可信时适用；
        首次定位请用 fast_prior=False）。
    debug : bool
        返回额外调试信息。

    Returns
    -------
    dict
        {"found": bool, "center": [cx, cy], "radius": r, "method": str,
         "score": float, "search_region": [...]}（未找到时 found=False）。
    """
    h, w = frame.shape[:2]
    # ---- 粗定位在降采样图上进行（性能 ~4x），结果映射回原尺寸 ----
    s = 2 if min(h, w) >= 320 else 1
    work = cv2.resize(frame, (w // s, h // s), interpolation=cv2.INTER_AREA) if s > 1 else frame
    wh, ww = work.shape[:2]

    if search_region is None:
        search_region = DEFAULT_SEARCH
    x0 = int(search_region[0] * ww)
    y0 = int(search_region[1] * wh)
    x1 = int(search_region[2] * ww)
    y1 = int(search_region[3] * wh)
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(ww, x1), min(wh, y1)

    gray = cv2.cvtColor(work, cv2.COLOR_BGR2GRAY)
    gray_f = cv2.GaussianBlur(gray, (5, 5), 0).astype(np.float32)
    masks = _color_masks(work)
    masks_f = {k: v.astype(np.float32) for k, v in masks.items()}

    short = min(wh, ww)
    r_lo = max(24, int(short * R_MIN_FRAC), int(ww * R_MIN_W_FRAC))
    r_hi = max(40, int(short * R_MAX_FRAC))
    r_hi = min(r_hi, min(y1 - y0, x1 - x0) // 2 - 2)
    if r_hi < r_lo:
        return {"found": False, "center": None, "radius": None,
                "method": None, "score": None,
                "search_region": [x0 * s, y0 * s, x1 * s, y1 * s], "debug": None}

    cands = []  # (score, cx, cy, r, method)，坐标均为降采样空间

    prior_s = None
    if prior is not None:
        prior_s = (int(prior[0] / s), int(prior[1] / s))

    if fast_prior and prior_s is not None:
        # ---- 跟踪模式：先验圆心可信，两段式（半径粗扫 -> 圆心微调 -> 半径精调）----
        px, py = prior_s
        if prior_r is not None:
            r_center = int(prior_r / s)
            r_band = range(max(r_lo, r_center - 12), min(r_hi, r_center + 13), 4)
        else:
            r_band = range(max(r_lo, int(short * 0.07)), min(r_hi, int(short * 0.16) + 1), 4)
        best_r = None
        for r in r_band:
            m = _disk_metrics(gray_f, masks_f, px, py, r)
            if m is None:
                continue
            cands.append((_score_disk(m, short=wh), px, py, r, "prior"))
            if best_r is None or cands[-1][0] > best_r[0]:
                best_r = cands[-1]
        if best_r is not None:
            # 圆心微调
            for ddy in (-4, 0, 4):
                for ddx in (-4, 0, 4):
                    m = _disk_metrics(gray_f, masks_f, px + ddx, py + ddy, best_r[3])
                    if m is None:
                        continue
                    cands.append((_score_disk(m, short=wh), px + ddx, py + ddy,
                                  best_r[3], "prior"))
            # 半径精调
            for r in range(max(r_lo, best_r[3] - 3), min(r_hi, best_r[3] + 4), 2):
                m = _disk_metrics(gray_f, masks_f, px, py, r)
                if m is None:
                    continue
                cands.append((_score_disk(m, short=wh), px, py, r, "prior"))
    else:
        # ---- 方法 A：filter2D 圆核粗扫（numpy 向量化筛选）----
        radii = sorted(set(range(r_lo, r_hi + 1, max(6, (r_hi - r_lo) // 6))))
        if not radii:
            radii = [r_lo]
        step = max(4, min(8, (r_hi - r_lo) // 4))
        for r in radii:
            kd = _disk_kernel(r)
            kr = _ring_kernel(r)
            sd, sr = kd.sum(), kr.sum()
            g_in_map = cv2.filter2D(gray_f, -1, kd, borderType=cv2.BORDER_REPLICATE) / sd
            g_ring_map = cv2.filter2D(gray_f, -1, kr, borderType=cv2.BORDER_REPLICATE) / sr
            # 颜色占比图归一化到 0-1，与 COLOR_FRAC_* 阈值一致
            b_map = cv2.filter2D(masks_f["blue"], -1, kd, borderType=cv2.BORDER_REPLICATE) / sd / 255.0
            r_map = cv2.filter2D(masks_f["red"], -1, kd, borderType=cv2.BORDER_REPLICATE) / sd / 255.0
            y_map = cv2.filter2D(masks_f["yellow"], -1, kd, borderType=cv2.BORDER_REPLICATE) / sd / 255.0
            y0c, y1c = y0 + r + 2, y1 - r - 2
            x0c, x1c = x0 + r + 2, x1 - r - 2
            if y1c <= y0c or x1c <= x0c:
                continue
            ok = ((g_in_map <= DISK_GRAY_MAX + 30) &
                  (b_map >= COLOR_FRAC_MIN) & (r_map >= COLOR_FRAC_MIN) &
                  ((b_map + r_map + y_map) <= COLOR_FRAC_MAX + 0.2))
            sub = ok[y0c:y1c, x0c:x1c]
            ys, xs = np.nonzero(sub)
            for dy, dx in zip(ys[::step], xs[::step]):
                cy, cx = y0c + int(dy), x0c + int(dx)
                m = {"g_in": float(g_in_map[cy, cx]), "g_ring": float(g_ring_map[cy, cx]),
                     "b_in": float(b_map[cy, cx]), "r_in": float(r_map[cy, cx]),
                     "y_in": float(y_map[cy, cx]), "r": float(r)}
                sc = _score_disk(m, short=wh)
                if sc > 15:
                    cands.append((sc, cx, cy, r, "disk"))

        # ---- 方法 B：霍夫圆候选 ----
        for (cx, cy, r) in _hough_candidates(gray, x1, y1, r_lo, r_hi):
            if not (x0 + r < cx < x1 - r and y0 + r < cy < y1 - r):
                continue
            m = _disk_metrics(gray_f, masks_f, cx, cy, r)
            if m is None:
                continue
            cands.append((_score_disk(m, short=wh), cx, cy, r, "hough"))

        # ---- 先验种子（可选）----
        if prior_s is not None:
            px, py = prior_s
            for r in radii:
                if not (x0 + r < px < x1 - r and y0 + r < py < y1 - r):
                    continue
                m = _disk_metrics(gray_f, masks_f, px, py, r)
                if m is None:
                    continue
                cands.append((_score_disk(m, short=wh), px, py, r, "prior"))

    if not cands:
        return {"found": False, "center": None, "radius": None,
                "method": None, "score": None,
                "search_region": [x0 * s, y0 * s, x1 * s, y1 * s], "debug": None}

    # ---- 局部精修：取前 N 个候选，微调圆心与半径（精确圆形 mask + 扇区一致性）----
    cands.sort(key=lambda t: -t[0])
    refined = []
    for (s0, cx, cy, r, meth) in cands[:6]:
        b_local = None
        for rr in range(max(r_lo, r - 4), min(r_hi, r + 5), 2):
            for ddy in range(-4, 5, 2):
                for ddx in range(-4, 5, 2):
                    m = _disk_metrics(gray_f, masks_f, cx + ddx, cy + ddy, rr)
                    if m is None:
                        continue
                    s2 = _score_disk(m, short=wh)
                    if b_local is None or s2 > b_local[0]:
                        b_local = (s2, cx + ddx, cy + ddy, rr, meth,
                                   m["g_in"], m["g_ring"])
        if b_local is not None:
            refined.append(b_local)

    if not refined:
        return {"found": False, "center": None, "radius": None,
                "method": None, "score": None,
                "search_region": [x0 * s, y0 * s, x1 * s, y1 * s], "debug": None}

    # ---- 圆点结构验证：候选圆内必须有紧凑的蓝/红圆点（排除场景色块）----
    refined.sort(key=lambda t: -t[0])
    # 圆点结构验证在原分辨率上进行（降采样会合并/丢失小圆点，导致漏判）
    masks_full = _color_masks(frame)
    best = None
    for (s0, cx, cy, r, meth, g_in, g_ring) in refined[:10]:
        fx, fy, frr = cx * s, cy * s, r * s
        dots = _dot_structure(masks_full, fx, fy, frr)
        ok, nb, nr, ny, mdf = _verify_dots(dots, fx, fy, frr)
        comb = s0 + _dot_bonus(nb, nr, ny)
        # 优先"结构验证通过"的候选；同为 ok 时取 comb 高者
        if best is None or (ok and not best[8]) or (ok == best[8] and comb > best[0]):
            best = (comb, s0, fx, fy, frr, meth, g_in, g_ring, ok, nb, nr, ny, mdf)

    comb, s0, fx, fy, frr, meth, g_in, g_ring, ok_dots, nb, nr, ny, mdf = best
    ring_pos = g_ring - g_in
    found = bool(s0 >= FOUND_SCORE_MIN and comb >= FOUND_SCORE_MIN and ok_dots
                 and g_in <= DISK_GRAY_MAX and ring_pos >= RING_CONTRAST_POS_MIN)
    result = {
        "found": found,
        "center": [int(fx), int(fy)],
        "radius": int(frr),
        "method": meth,
        "score": round(s0, 2),
        "dot_score": round(comb, 2),
        "disk_stats": {"g_in": round(g_in, 1), "g_ring": round(g_ring, 1),
                       "ring_contrast": round(ring_pos, 1)},
        "dots_in_disk": {"blue": nb, "red": nr, "yellow": ny,
                         "mean_dist_frac": round(mdf, 3) if mdf is not None else None},
        "search_region": [x0 * s, y0 * s, x1 * s, y1 * s],
    }
    if debug:
        result["debug"] = {
            "candidates": [[round(float(t[0]), 2), t[1] * s, t[2] * s, t[3] * s, t[4]]
                           for t in cands[:10]],
            "refined": [[round(float(t[0]), 2), t[1] * s, t[2] * s, t[3] * s, t[4]]
                        for t in refined[:5]],
            "metrics": _disk_metrics(gray_f, masks_f, fx // s, fy // s, frr // s),
        }
    return result


def detect_dots(frame, minimap, debug=False):
    """在小地图圆盘内检测并分类蓝/红/黄圆点与塔点。

    Parameters
    ----------
    frame : np.ndarray  BGR 帧。
    minimap : dict      find_minimap 的返回。
    debug : bool        是否在 detail 中附带更多信息。

    Returns
    -------
    dict  见模块 docstring 中的输出结构；未找到小地图时 dots 为空。
    """
    empty = {"dots": {"blue": [], "red": [], "yellow": [], "green": []},
             "minions": {"blue": [], "red": [], "green": []},
             "towers": [], "detail": {}}
    if not minimap.get("found"):
        return empty

    h, w = frame.shape[:2]
    cx, cy, r = minimap["center"][0], minimap["center"][1], minimap["radius"]
    if r <= 0:
        return empty

    masks = _color_masks(frame)
    # v2.8：小地图为方形（用户实测：最大范围 2r x 2r），用方形 mask 而非圆形
    disk = np.zeros((h, w), np.uint8)
    x0m, y0m = max(0, cx - r), max(0, cy - r)
    x1m, y1m = min(w, cx + r), min(h, cy + r)
    disk[y0m:y1m, x0m:x1m] = 255

    # 尺寸阈值（相对圆盘半径 r）v2.8 真机标定：
    #   英雄圈（头像）：面积 ~90-300（30x30）
    #   小兵/野怪点：~20-40（7x7）
    #   塔（方块）：~40-90（矩形）
    #   河道/大块：>500（误检过滤）
    area_noise_max = np.pi * (FRAC_NOISE_MAX * r) ** 2
    area_hero_lo = np.pi * (FRAC_HERO_LO * r) ** 2
    area_hero_hi = np.pi * (FRAC_HERO_HI * r) ** 2

    def _classify(area, comp=None, bright=False, color=None):
        """返回 "noise" | "tower" | "hero" | "minion"。

        v2.8 真机标定（方形小地图 232x232，r=116 时）：
          hero  : 60 <= area <= 450（英雄圈：自己绿圈 32x33=372、队友蓝圈 12x17≈140）
          minion: 10 <= area < 60（小兵/野怪小点 7x7≈30）
          tower : 矩形且面积 40-90（塔方块，长宽比>1.5）
          noise : area < 10 或 > 500（河道/大块）
        """
        if area < 10 or area > 700:
            return "noise"
        if comp is not None and comp < 0.45:
            return "noise"
        if area >= 60:
            # v2.9：英雄圈需近圆形（comp>=0.55），排除红色地形碎片
            if comp is not None and comp < 0.55:
                return "noise"
            return "hero"
        # v2.9：只有蓝/红小点才算塔（塔有阵营色）；黄色/绿色小点=野怪/小兵
        if color in ("blue", "red") and comp is not None and comp < 0.7:
            return "tower"     # 矩形（长宽比大）= 塔
        return "minion"

    def _norm(x, y):
        nx = (x - (cx - r)) / (2.0 * r)
        ny = (y - (cy - r)) / (2.0 * r)
        # v2.8：方形小地图，所有框内点均有效
        valid = 0.0 <= nx <= 1.0 and 0.0 <= ny <= 1.0
        return round(float(nx), 4), round(float(ny), 4), bool(valid)

    dots = {"blue": [], "red": [], "yellow": [], "green": []}
    # v2.8：minions 单独输出（小绿点=我方小兵、小红点=敌方小兵、小蓝点=我方小兵）
    minions = {"blue": [], "red": [], "green": []}
    towers = []
    detail = {"blue": [], "red": [], "yellow": [], "green": [], "towers": [],
              "minions": {"blue": [], "red": [], "green": []}}

    # 全图灰度（供亮度验证）
    gray_full = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    for color in ("blue", "red", "yellow", "green"):
        m = masks[color] & disk
        # v2.9：蓝色/红色/绿色先开运算去噪再闭运算合并（英雄圈+箭头连成一体）；
        # 黄色（野怪小点 6x7）保持原始，不做形态学（避免小点消失）
        if color == "yellow":
            pass
        else:
            m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
            m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
        n, lab, st, cent = cv2.connectedComponentsWithStats(m, 8)
        for i in range(1, n):
            area = int(st[i, cv2.CC_STAT_AREA])
            w_ = int(st[i, cv2.CC_STAT_WIDTH])
            h_ = int(st[i, cv2.CC_STAT_HEIGHT])
            comp = area / max(1.0, w_ * h_)          # 紧凑度
            x0c = int(st[i, cv2.CC_STAT_LEFT])
            y0c = int(st[i, cv2.CC_STAT_TOP])
            # 亮度验证：连通域内灰度均值（英雄点较亮）
            bbox = gray_full[y0c:y0c + h_, x0c:x0c + w_]
            gmean = float(bbox.mean()) if bbox.size else 0.0
            bright = gmean > 90
            cls = _classify(area, comp=comp, bright=bright, color=color)
            if cls == "noise":
                continue
            px, py = float(cent[i][0]), float(cent[i][1])
            nx, ny, valid = _norm(px, py)
            rec = {"n": [nx, ny], "px": [round(px, 1), round(py, 1)],
                   "area": area, "comp": round(comp, 2), "gmean": round(gmean, 0),
                   "valid": valid}
            if cls == "hero":
                # v2.9：小地图顶部边缘（归一化 y<0.10）的红色是基地/UI，非敌人
                if color == "red" and ny < 0.10:
                    continue
                dots[color].append([nx, ny])
                detail[color].append(rec)
            elif cls == "minion":
                if color == "yellow":
                    # 黄点 = 野怪（无论大小都算野怪标记）
                    dots["yellow"].append([nx, ny])
                    detail["yellow"].append(rec)
                else:
                    minions[color].append([nx, ny])
                    detail["minions"][color].append(rec)
            else:  # tower
                towers.append([nx, ny])
                rec["color"] = color
                detail["towers"].append(rec)

    return {"dots": dots, "minions": minions, "towers": towers,
            "detail": detail}


def analyze(frame, find_kw=None, detect_kw=None):
    """组合入口：定位小地图 + 检测圆点。

    Returns
    -------
    dict  {"found", "center", "radius", "method", "score", "dots", "towers", "detail"}
    """
    find_kw = find_kw or {}
    detect_kw = detect_kw or {}
    minimap = find_minimap(frame, **find_kw)
    det = detect_dots(frame, minimap, **detect_kw)
    return {
        "found": minimap["found"],
        "center": minimap["center"],
        "radius": minimap["radius"],
        "method": minimap["method"],
        "score": minimap["score"],
        "dot_score": minimap.get("dot_score"),
        "disk_stats": minimap.get("disk_stats"),
        "dots_in_disk": minimap.get("dots_in_disk"),
        "search_region": minimap.get("search_region"),
        "dots": det["dots"],
        "towers": det["towers"],
        "detail": det["detail"],
    }


# ---------------------------------------------------------------------------
# 可视化辅助（供测试脚本 / 演示使用）
# ---------------------------------------------------------------------------

def draw_overlay(frame, result, show_labels=True):
    """在帧上叠加小地图圆盘与圆点，返回新图像（BGR）。"""
    out = frame.copy()
    if result.get("found") and result["center"]:
        cx, cy, r = result["center"][0], result["center"][1], result["radius"]
        cv2.circle(out, (cx, cy), r, (0, 255, 255), 2)
        cv2.circle(out, (cx, cy), 3, (0, 255, 255), -1)
        colors = {"blue": (255, 128, 0), "red": (0, 0, 255), "yellow": (0, 255, 255)}
        for color, pts in result["dots"].items():
            for (nx, ny) in pts:
                px = int((nx - 0.5) * 2 * r) + cx
                py = int((ny - 0.5) * 2 * r) + cy
                cv2.circle(out, (px, py), 4, colors[color], -1)
                if show_labels:
                    cv2.putText(out, color[0], (px + 5, py - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, colors[color], 1, cv2.LINE_AA)
        for (nx, ny) in result["towers"]:
            px = int((nx - 0.5) * 2 * r) + cx
            py = int((ny - 0.5) * 2 * r) + cy
            cv2.circle(out, (px, py), 2, (255, 255, 255), -1)
        if show_labels:
            cv2.putText(out, f"minimap r={r}", (cx + r + 6, cy),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
    else:
        cv2.putText(out, "minimap NOT found", (12, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
    return out


# ---------------------------------------------------------------------------
# 模块级演示
# ---------------------------------------------------------------------------

def _demo_image(path, out_dir, show):
    img = cv2.imread(str(path))
    if img is None:
        print(f"[demo] 无法读取图片: {path}")
        return
    res = analyze(img)
    print(json.dumps({k: res[k] for k in ("found", "center", "radius", "method", "score")},
                     ensure_ascii=False))
    print(json.dumps({"dots": res["dots"], "towers": res["towers"]}, ensure_ascii=False))
    if out_dir:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        p = out_dir / (path.stem + "_overlay.png")
        cv2.imwrite(str(p), draw_overlay(img, res))
        print(f"[demo] 叠加图已保存: {p}")
    if show:
        cv2.imshow("minimap demo", draw_overlay(img, res))
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def _demo_video(path, out_dir, max_frames=30, show=False, start_frame=0):
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        print(f"[demo] 无法打开视频: {path}")
        return
    n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"[demo] 视频 {path}: {n_total} 帧, {fps:.2f} fps")
    start = start_frame
    if start <= 0 and n_total > max_frames:
        start = int(n_total * 0.3)  # 跳过开头（可能是加载/选英雄）
    stride = max(1, (n_total - start) // max_frames)
    frames = [start + i * stride for i in range(min(max_frames, n_total))]
    if out_dir:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
    found = 0
    for i, n in enumerate(frames):
        cap.set(cv2.CAP_PROP_POS_FRAMES, n)
        ok, fr = cap.read()
        if not ok:
            continue
        res = analyze(fr)
        if res["found"]:
            found += 1
        print(f"  frame {n}: found={res['found']} center={res['center']} r={res['radius']} "
              f"blue={len(res['dots']['blue'])} red={len(res['dots']['red'])} "
              f"yellow={len(res['dots']['yellow'])} towers={len(res['towers'])}")
        if out_dir and i < 5:
            cv2.imwrite(str(out_dir / f"demo_frame{n}.png"), draw_overlay(fr, res))
        if show:
            cv2.imshow("minimap demo", draw_overlay(fr, res))
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    cap.release()
    print(f"[demo] 定位成功 {found}/{len(frames)}")
    if show:
        cv2.destroyAllWindows()


def main(argv=None):
    ap = argparse.ArgumentParser(description="小地图感知演示")
    ap.add_argument("--image", type=str, default=None, help="单张图片路径")
    ap.add_argument("--video", type=str, default=None, help="视频路径")
    ap.add_argument("--frame", type=int, default=0, help="视频起始帧/跳帧帧号")
    ap.add_argument("--out", type=str, default=None, help="输出目录（保存叠加图）")
    ap.add_argument("--show", action="store_true", help="弹窗显示")
    args = ap.parse_args(argv)

    if args.image:
        _demo_image(Path(args.image), args.out, args.show)
    elif args.video:
        _demo_video(args.video, args.out, show=args.show, start_frame=args.frame)
    else:
        ap.print_help()
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
