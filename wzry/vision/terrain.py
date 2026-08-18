# -*- coding: utf-8 -*-
"""小地图地形感知：方形小地图 -> 可走性栅格（墙体识别 + A* 寻路基础）。

背景（真机实测标定）：
  - 王者荣耀小地图在左上角，**方形**（本局实测 (40,80)-(160,200)，120x120px；
    不同分辨率/局次位置可能不同，用 find_minimap_box 自适应定位）。
  - 小地图底色是蓝灰色系半透明，灰度中位 ~68，亮度 V 主峰 64-80。
  - 可走性语义（亮度 V + 饱和度 S 联合）：
      可走（道路/兵线/平地）：V 高（> 中位+阈值）且 S 低
      可走（河道/水域）：S 高 + H 蓝青（走速慢，但可通行）
      不可走（墙体/高地边缘/深色地形）：V 低
  - 单帧噪声大（英雄/塔图标），用多帧平均底图（去图标）做静态地形，
    实时运行时用"最近 N 帧平均"滚动更新。

用法：
    from wzry.vision.terrain import extract_walk_grid, astar_path, find_minimap_box
"""
from __future__ import annotations

import heapq
import time
from pathlib import Path

import cv2
import numpy as np

# 小地图默认搜索区（左上角，比例坐标）
SEARCH_REGION = (0.0, 0.0, 0.25, 0.30)

# 默认小地图边界框（1280x720，真机实测标定：方形小地图圆心(116,116) 半径116）
DEFAULT_BOX = (0, 0, 232, 232)

# 方形小地图尺寸范围（相对画面宽）
BOX_W_MIN_FRAC = 0.06
BOX_W_MAX_FRAC = 0.18

# 可走性阈值（亮度 V 与饱和度 S，0-255）
WALK_V_MIN = 88        # V >= 88 视为偏亮（道路/平地）
RIVER_S_MIN = 110      # 饱和度 >= 110 且 H 蓝青 -> 河道（可走）
RIVER_H_LO, RIVER_H_HI = 85, 135
WALL_V_MAX = 66        # V <= 66 且非河道 -> 墙体/深色地形（不可走）


def find_minimap_box(frame):
    """自适应定位方形小地图边界框。

    优先返回标定默认框（王者 UI 固定，DEFAULT_BOX 按 1280x720 标定，
    按当前帧尺寸等比缩放）；若默认框内蓝青特征不足，再扫描左上角。
    返回 (x0, y0, x1, y1) 或 None。
    """
    h, w = frame.shape[:2]
    # 默认框按分辨率等比缩放
    sx = w / 1280.0
    sy = h / 720.0
    def_box = (int(DEFAULT_BOX[0] * sx), int(DEFAULT_BOX[1] * sy),
               int(DEFAULT_BOX[2] * sx), int(DEFAULT_BOX[3] * sy))
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    H, S, V = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    # 蓝青 mask（小地图底色特征）
    blue = ((H >= RIVER_H_LO) & (H <= RIVER_H_HI) & (S >= 40) & (V >= 40)).astype(np.uint8)
    # 彩色图标 mask（英雄/塔/野怪点）
    color = (S >= 90).astype(np.uint8)

    # 1) 标定框检查
    x0, y0, x1, y1 = def_box
    if x1 <= w and y1 <= h:
        b = float(blue[y0:y1, x0:x1].mean())
        cc = float(color[y0:y1, x0:x1].mean())
        if 0.30 < b < 0.95 and cc > 0.006:
            return def_box
    # 2) 回退：窄范围扫描（只在极左区域，避免头像区误检）
    sx0 = int(SEARCH_REGION[0] * w)
    sy0 = int(SEARCH_REGION[1] * h)
    sx1 = int(0.14 * w)          # 只在 x < 14% 宽度内找（排除头像区）
    sy1 = int(SEARCH_REGION[3] * h)
    box_lo = int(BOX_W_MIN_FRAC * w)
    box_hi = int(BOX_W_MAX_FRAC * w)
    if sx1 <= sx0 or sy1 <= sy0:
        return def_box
    best = None
    for y0 in range(sy0, sy1 - box_lo, 6):
        for x0 in range(sx0, sx1 - box_lo, 6):
            for size in range(box_lo, box_hi + 1, 10):
                x1, y1 = x0 + size, y0 + size
                if x1 > sx1 or y1 > sy1:
                    continue
                b = float(blue[y0:y1, x0:x1].mean())
                cc = float(color[y0:y1, x0:x1].mean())
                if 0.30 < b < 0.95 and cc > 0.006:
                    score = b + cc * 12.0
                    if best is None or score > best[0]:
                        best = (score, x0, y0, x1, y1)
    if best is None:
        return def_box
    return best[1:]


def extract_walk_grid(frame, box=None, grid_n=40):
    """从单帧提取可走性栅格（0=不可走, 1=可走, 2=河道可走）。

    box: (x0, y0, x1, y1)；缺省用 find_minimap_box 定位。
    返回 (grid, box)：grid (grid_n, grid_n) float32。
    """
    if box is None:
        box = find_minimap_box(frame)
        if box is None:
            return None, None
    x0, y0, x1, y1 = box
    roi = frame[y0:y1, x0:x1]
    if roi.size == 0:
        return None, box
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    H, S, V = hsv[..., 0].astype(np.int16), hsv[..., 1].astype(np.int16), hsv[..., 2].astype(np.int16)

    walk = np.full((grid_n, grid_n), 0.0, np.float32)
    for gy in range(grid_n):
        for gx in range(grid_n):
            # 网格中心 -> ROI 像素
            px = int((gx + 0.5) / grid_n * roi.shape[1])
            py = int((gy + 0.5) / grid_n * roi.shape[0])
            px = min(roi.shape[1] - 1, max(0, px))
            py = min(roi.shape[0] - 1, max(0, py))
            h_, s_, v_ = H[py, px], S[py, px], V[py, px]
            if (s_ >= RIVER_S_MIN and RIVER_H_LO <= h_ <= RIVER_H_HI):
                walk[gy, gx] = 2.0            # 河道：可走但标记
            elif v_ >= WALK_V_MIN:
                walk[gy, gx] = 1.0            # 偏亮：道路/平地
            else:
                walk[gy, gx] = 0.0            # 暗：墙体/深色地形
    return walk, box


class RollingTerrain:
    """滚动平均小地图底图 -> 稳定可走栅格（去图标噪声）。

    用法：
        t = RollingTerrain()
        grid, box = t.update(frame)   # 每帧调用，内部多帧平均后提取栅格
    """

    def __init__(self, n_frames=8, grid_n=40, settle_s=3.0):
        self.n_frames = n_frames
        self.grid_n = grid_n
        self.settle_s = settle_s          # 滚动窗口时长（秒）
        self.box = None
        self._acc = None
        self._t0 = None
        self._n = 0

    def update(self, frame):
        box = self.box or find_minimap_box(frame)
        if box is None:
            return None, None
        x0, y0, x1, y1 = box
        roi = frame[y0:y1, x0:x1]
        if roi.size == 0:
            return None, box
        now = time.time()
        if self._acc is None:
            self._acc = np.zeros(roi.shape, np.float32)
            self._t0 = now
        # 滚动：超过窗口时间则重置（避免旧帧残留）
        if now - self._t0 > self.settle_s:
            self._acc = np.zeros(roi.shape, np.float32)
            self._t0 = now
            self._n = 0
        self._acc += roi.astype(np.float32)
        self._n += 1
        avg = (self._acc / self._n).astype(np.uint8)
        # 用平均底图提取栅格
        hsv = cv2.cvtColor(avg, cv2.COLOR_BGR2HSV)
        H = hsv[..., 0].astype(np.int16)
        S = hsv[..., 1].astype(np.int16)
        V = hsv[..., 2].astype(np.int16)
        n = self.grid_n
        walk = np.full((n, n), 0.0, np.float32)
        for gy in range(n):
            for gx in range(n):
                px = min(roi.shape[1] - 1, max(0, int((gx + 0.5) / n * roi.shape[1])))
                py = min(roi.shape[0] - 1, max(0, int((gy + 0.5) / n * roi.shape[0])))
                h_, s_, v_ = H[py, px], S[py, px], V[py, px]
                if s_ >= RIVER_S_MIN and RIVER_H_LO <= h_ <= RIVER_H_HI:
                    walk[gy, gx] = 2.0
                elif v_ >= WALK_V_MIN:
                    walk[gy, gx] = 1.0
                else:
                    walk[gy, gx] = 0.0
        self.box = box
        return walk, box


def astar_path(grid, start, goal, allow_river=True):
    """A* 寻路：grid 可走性栅格 -> 路径（网格坐标列表）。

    河道（值为2）默认可走但代价高（*1.5），墙体（0）不可走。
    返回 [(gy, gx), ...] 或 None（无路径）。
    """
    h, w = grid.shape

    def cost(v):
        if v <= 0:
            return None
        return 1.5 if (v >= 2 and not allow_river) else (1.5 if v >= 2 else 1.0)

    def heur(a, b):
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def neighbors(p):
        for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1),
                       (-1, -1), (-1, 1), (1, -1), (1, 1)):
            ny, nx = p[0] + dy, p[1] + dx
            if 0 <= ny < h and 0 <= nx < w:
                yield ny, nx, (1.414 if dy and dx else 1.0)

    if cost(grid[start]) is None or cost(grid[goal]) is None:
        return None
    open_h = [(heur(start, goal), 0.0, start)]
    gcost = {start: 0.0}
    came = {}
    closed = set()
    while open_h:
        _, g, cur = heapq.heappop(open_h)
        if cur == goal:
            path = [cur]
            while cur in came:
                cur = came[cur]
                path.append(cur)
            return path[::-1]
        if cur in closed:
            continue
        closed.add(cur)
        for ny, nx, step in neighbors(cur):
            c = cost(grid[ny, nx])
            if c is None:
                continue
            ng = g + step * c
            nxt = (ny, nx)
            if nxt not in gcost or ng < gcost[nxt]:
                gcost[nxt] = ng
                came[nxt] = cur
                heapq.heappush(open_h, (ng + heur(nxt, goal), ng, nxt))
    return None


def path_to_moves(path, grid_n=40):
    """路径（网格坐标）-> 每步移动方向（屏幕角度，0=右 逆时针）。

    返回 [(theta, r, duration_ms), ...]：逐网格移动指令（每格 400ms）。
    """
    moves = []
    for i in range(1, len(path)):
        dy = path[i][0] - path[i - 1][0]
        dx = path[i][1] - path[i - 1][1]
        # 屏幕坐标：网格 y 向下，屏幕 y 向下；theta 0=右
        theta = np.arctan2(dy, dx)
        moves.append((float(theta), 1.0, 400))
    return moves


if __name__ == "__main__":
    import sys
    img_path = sys.argv[1] if len(sys.argv) > 1 else None
    if img_path:
        img = cv2.imread(img_path)
        box = find_minimap_box(img)
        print("box:", box)
        if box:
            grid, _ = extract_walk_grid(img, box=box)
            if grid is not None:
                print(f"可走率 {grid[grid > 0].size / grid.size * 100:.0f}%")
                for gy in range(grid.shape[0]):
                    row = "".join("#" if grid[gy, gx] else "." for gx in range(grid.shape[1]))
                    print(row)
