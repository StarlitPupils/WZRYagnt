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
COLOR_THRESH = {
    # 蓝方（我方）圆点
    "blue": {"h_lo": 95, "h_hi": 135, "s_min": 60, "v_min": 50},
    # 红方（敌方）圆点：H 红 = 0 附近与 170-180 两段
    "red": {"h_lo": 0, "h_hi": 10, "h2_lo": 170, "h2_hi": 180, "s_min": 70, "v_min": 50},
    # 中立单位（野怪/龙等）圆点
    "yellow": {"h_lo": 15, "h_hi": 38, "s_min": 80, "v_min": 60},
}

# 定位搜索范围：默认左上 40% 区域（任务要求；可通过 search_region 覆盖）
DEFAULT_SEARCH = (0.0, 0.0, 0.40, 0.40)  # (x0, y0, x1, y1) 相对比例

# 圆盘半径搜索范围（相对短边）
R_MIN_FRAC = 0.05
R_MAX_FRAC = 0.32

# 圆盘判据（minimap 的特征）
DISK_GRAY_MAX = 112.0        # 盘内平均灰度上限（深色圆盘）
RING_CONTRAST_MIN = 6.0      # 圆环(盘外)与盘内的最小平均灰度差（亮环或暗环均可，取绝对值）
COLOR_FRAC_MIN = 0.0004      # 盘内彩色像素占比下限（必须同时有蓝和红才可能是小地图）
COLOR_FRAC_MAX = 0.35        # 盘内彩色占比上限（超过则更像是场景而非小地图）
BLUE_FRAC_SCENE = 0.22       # 盘内蓝色占比超过该值判为"蓝方基地场景"，扣分

# 圆点尺寸（以圆盘半径 r 为基准的半径比例）。
# 英雄点（小圆点）> 塔点（略小）> 噪声（极小）；三个区间不重叠：
#   tower: [0.014, 0.030) * r，hero: [0.032, 0.090] * r
FRAC_NOISE_MAX = 0.014       # 半径 < 1.4% r 视为噪声
FRAC_TOWER_LO = 0.014        # 塔点半径下界
FRAC_HERO_LO = 0.032         # 英雄点半径下界
FRAC_HERO_HI = 0.090         # 英雄点半径上界（再大视为误检）

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


def _disk_metrics(gray_f, masks_f, cx, cy, r):
    """计算候选圆盘的一组度量（都基于圆形 mask，像素均值）。返回 dict 或 None（越界）。"""
    h, w = gray_f.shape
    if cx - r < 2 or cy - r < 2 or cx + r >= w - 2 or cy + r >= h - 2:
        return None
    disk = np.zeros((h, w), np.uint8)
    cv2.circle(disk, (int(cx), int(cy)), int(r), 255, -1)
    ring = np.zeros((h, w), np.uint8)
    cv2.circle(ring, (int(cx), int(cy)), int(r + 6), 255, 2)
    ring[disk == 255] = 0
    ins = disk == 255
    g_in = float(gray_f[ins].mean())
    g_ring = float(gray_f[ring == 255].mean()) if ring.any() else g_in
    b_in = float(masks_f["blue"][ins].mean())
    r_in = float(masks_f["red"][ins].mean())
    y_in = float(masks_f["yellow"][ins].mean())
    return {"g_in": g_in, "g_ring": g_ring, "b_in": b_in, "r_in": r_in, "y_in": y_in}


def _score_disk(m):
    """把 _disk_metrics 的输出转成分数。分数高 => 更像小地图圆盘。"""
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
        score += 70.0
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


# ---------------------------------------------------------------------------
# 公开 API
# ---------------------------------------------------------------------------

def find_minimap(frame, search_region=None, prior=None, debug=False):
    """自适应定位左上角小地图圆盘。

    Parameters
    ----------
    frame : np.ndarray
        BGR 帧（任意分辨率）。
    search_region : tuple | None
        搜索区域 (x0, y0, x1, y1) 相对比例；默认 (0, 0, 0.40, 0.40)。
    prior : tuple | None
        可选的先验圆心 (px, py)；提供时作为精修种子之一（不提供则完全自适应）。
    debug : bool
        返回额外调试信息。

    Returns
    -------
    dict
        {"found": bool, "center": [cx, cy], "radius": r, "method": str,
         "score": float, "search_region": [...]}（未找到时 found=False）。
    """
    h, w = frame.shape[:2]
    if search_region is None:
        search_region = DEFAULT_SEARCH
    x0 = int(search_region[0] * w)
    y0 = int(search_region[1] * h)
    x1 = int(search_region[2] * w)
    y1 = int(search_region[3] * h)
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(w, x1), min(h, y1)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray_f = cv2.GaussianBlur(gray, (5, 5), 0).astype(np.float32)
    masks = _color_masks(frame)
    masks_f = {k: v.astype(np.float32) for k, v in masks.items()}

    short = min(h, w)
    r_lo = max(16, int(short * R_MIN_FRAC))
    r_hi = max(40, int(short * R_MAX_FRAC))
    r_hi = min(r_hi, min(y1 - y0, x1 - x0) // 2 - 2)

    cands = []  # (score, cx, cy, r, method)

    # ---- 方法 A：filter2D 圆核粗扫 + 局部精修 ----
    radii = sorted(set(range(r_lo, r_hi + 1, max(6, (r_hi - r_lo) // 8))))
    if not radii:
        radii = [r_lo]
    step = max(3, min(8, (r_hi - r_lo) // 6))
    seen = set()
    for r in radii:
        kd = _disk_kernel(r)
        kr = _ring_kernel(r)
        sd, sr = kd.sum(), kr.sum()
        g_in_map = cv2.filter2D(gray_f, -1, kd, borderType=cv2.BORDER_REPLICATE) / sd
        g_ring_map = cv2.filter2D(gray_f, -1, kr, borderType=cv2.BORDER_REPLICATE) / sr
        b_map = cv2.filter2D(masks_f["blue"], -1, kd, borderType=cv2.BORDER_REPLICATE) / sd
        r_map = cv2.filter2D(masks_f["red"], -1, kd, borderType=cv2.BORDER_REPLICATE) / sd
        y_map = cv2.filter2D(masks_f["yellow"], -1, kd, borderType=cv2.BORDER_REPLICATE) / sd
        for cy in range(y0 + r + 2, y1 - r - 2, step):
            for cx in range(x0 + r + 2, x1 - r - 2, step):
                gi = float(g_in_map[cy, cx])
                if gi > DISK_GRAY_MAX + 30:
                    continue
                bi, ri, yi = float(b_map[cy, cx]), float(r_map[cy, cx]), float(y_map[cy, cx])
                if not (bi >= COLOR_FRAC_MIN and ri >= COLOR_FRAC_MIN):
                    continue
                if bi + ri + yi > COLOR_FRAC_MAX + 0.2:
                    continue
                m = {"g_in": gi, "g_ring": float(g_ring_map[cy, cx]),
                     "b_in": bi, "r_in": ri, "y_in": yi}
                s = _score_disk(m)
                if s > 20:
                    cands.append((s, cx, cy, r, "disk"))

    # ---- 方法 B：霍夫圆候选 ----
    for (cx, cy, r) in _hough_candidates(gray, x1, y1, r_lo, r_hi):
        if not (x0 + r < cx < x1 - r and y0 + r < cy < y1 - r):
            continue
        m = _disk_metrics(gray_f, masks_f, cx, cy, r)
        if m is None:
            continue
        s = _score_disk(m)
        if s > 20:
            cands.append((s, cx, cy, r, "hough"))

    # ---- 先验种子（可选）----
    if prior is not None:
        px, py = int(prior[0]), int(prior[1])
        for r in radii:
            if not (x0 + r < px < x1 - r and y0 + r < py < y1 - r):
                continue
            m = _disk_metrics(gray_f, masks_f, px, py, r)
            if m is None:
                continue
            s = _score_disk(m)
            cands.append((s, px, py, r, "prior"))

    if not cands:
        return {"found": False, "center": None, "radius": None,
                "method": None, "score": None,
                "search_region": [x0, y0, x1, y1], "debug": None}

    # ---- 局部精修：取前 N 个候选，微调圆心与半径 ----
    cands.sort(key=lambda t: -t[0])
    best = None
    for (s0, cx, cy, r, meth) in cands[:8]:
        for rr in range(max(r_lo, r - 5), min(r_hi, r + 6)):
            for ddy in range(-4, 5, 2):
                for ddx in range(-4, 5, 2):
                    m = _disk_metrics(gray_f, masks_f, cx + ddx, cy + ddy, rr)
                    if m is None:
                        continue
                    s = _score_disk(m)
                    if best is None or s > best[0]:
                        best = (s, cx + ddx, cy + ddy, rr, meth)

    s, cx, cy, r, meth = best
    result = {
        "found": s >= 30.0,
        "center": [int(cx), int(cy)],
        "radius": int(r),
        "method": meth,
        "score": round(s, 2),
        "search_region": [x0, y0, x1, y1],
    }
    if debug:
        result["debug"] = {
            "candidates": [[round(float(t[0]), 2), t[1], t[2], t[3], t[4]] for t in cands[:10]],
            "metrics": _disk_metrics(gray_f, masks_f, cx, cy, r),
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
    empty = {"dots": {"blue": [], "red": [], "yellow": []},
             "towers": [], "detail": {}}
    if not minimap.get("found"):
        return empty

    h, w = frame.shape[:2]
    cx, cy, r = minimap["center"][0], minimap["center"][1], minimap["radius"]
    if r <= 0:
        return empty

    masks = _color_masks(frame)
    disk = np.zeros((h, w), np.uint8)
    cv2.circle(disk, (cx, cy), r, 255, -1)
    # 稍微膨胀圆盘 mask，避免贴边圆点被裁掉一部分面积而掉档
    disk = cv2.dilate(disk, np.ones((3, 3), np.uint8))

    # 尺寸阈值（相对圆盘半径 r）
    area_noise_max = np.pi * (FRAC_NOISE_MAX * r) ** 2
    area_hero_lo = np.pi * (FRAC_HERO_LO * r) ** 2
    area_hero_hi = np.pi * (FRAC_HERO_HI * r) ** 2

    def _classify(area):
        """返回 "noise" | "tower" | "hero"。塔点略小于英雄点。"""
        if area < max(2.0, area_noise_max):
            return "noise"
        if area <= area_hero_hi and area >= area_hero_lo:
            return "hero"
        if area < area_hero_lo:
            return "tower"
        return "noise"  # 过大，视为误检

    def _norm(x, y):
        nx = (x - (cx - r)) / (2.0 * r)
        ny = (y - (cy - r)) / (2.0 * r)
        valid = (nx - 0.5) ** 2 + (ny - 0.5) ** 2 <= 0.25 + 1e-6
        return round(float(nx), 4), round(float(ny), 4), bool(valid)

    dots = {"blue": [], "red": [], "yellow": []}
    towers = []
    detail = {"blue": [], "red": [], "yellow": [], "towers": []}

    for color in ("blue", "red", "yellow"):
        m = masks[color] & disk
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((MORPH_K, MORPH_K), np.uint8))
        n, lab, st, cent = cv2.connectedComponentsWithStats(m, 8)
        for i in range(1, n):
            area = int(st[i, cv2.CC_STAT_AREA])
            cls = _classify(area)
            if cls == "noise":
                continue
            px, py = float(cent[i][0]), float(cent[i][1])
            nx, ny, valid = _norm(px, py)
            rec = {"n": [nx, ny], "px": [round(px, 1), round(py, 1)],
                   "area": area, "valid": valid}
            if cls == "hero":
                dots[color].append([nx, ny])
                detail[color].append(rec)
            else:  # tower
                towers.append([nx, ny])
                rec["color"] = color
                detail["towers"].append(rec)

    return {"dots": dots, "towers": towers, "detail": detail}


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


def _demo_video(path, out_dir, max_frames=30, show=False):
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        print(f"[demo] 无法打开视频: {path}")
        return
    n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"[demo] 视频 {path}: {n_total} 帧, {fps:.2f} fps")
    start = 0
    if max_frames > 0 and n_total > max_frames:
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
        _demo_video(args.video, args.out, show=args.show)
    else:
        ap.print_help()
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
