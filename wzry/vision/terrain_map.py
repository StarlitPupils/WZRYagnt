# -*- coding: utf-8 -*-
"""王者峡谷固定地形模板（小地图归一化坐标 0-1，40x40 栅格）。

王者峡谷地图是固定布局（公开知识 + 小地图实测校准）：
  - 三条兵线（对抗路/中路/发育路）在边缘，形成"工"字形骨架；
  - 河道斜穿地图（左上-右下），在中路附近拐弯；
  - 野区在河道两侧；基地在左下（蓝）与右上（红）。

用途：Agent 全局寻路时用模板栅格做 A*（墙体=0 不可走，兵线/河道=1 可走），
运行时小地图定位后可直接查询。注意：这是**粗模板**，真实墙体以撞墙感知兜底。
"""
import numpy as np

GRID_N = 40


def lane_band(gx, gy, x0, y0, x1, y1, half=2):
    """点到线段 (x0,y0)-(x1,y1) 的距离 < half 格 -> 属于兵线。"""
    # 参数 t 投影
    dx, dy = x1 - x0, y1 - y0
    seg2 = dx * dx + dy * dy
    if seg2 < 1e-6:
        return False
    t = max(0.0, min(1.0, ((gx - x0) * dx + (gy - y0) * dy) / seg2))
    px, py = x0 + t * dx, y0 + t * dy
    return (gx - px) ** 2 + (gy - py) ** 2 <= half * half


def build_king_canyon_template(n=GRID_N):
    """构建王者峡谷 40x40 可走模板：1=可走(兵线/河道/基地区)，0=墙体/野区。"""
    g = np.zeros((n, n), np.float32)
    c = n - 1
    # 三条兵线（端点按小地图归一化坐标 * n）
    #   对抗路：左上(0.08,0.08) -> 右上(0.92,0.08)   [上边线]
    #   中路：  左下(0.08,0.92) -> 右上(0.92,0.08)   [对角线]
    #   发育路：左下(0.08,0.92) -> 右下(0.92,0.92)   [下边线]
    #   左边线：对抗路左上 -> 中路左下  [连接上左下]
    #   右边线：发育路右下 -> 中路右上  [连接下右上]
    lanes = [
        # 三条兵线（蓝方视角：泉水左下）
        (0.10, 0.90, 0.90, 0.90),   # 发育路：下边线（泉水->右下）
        (0.90, 0.90, 0.90, 0.10),   # 发育路转对抗路：右边线（右下->右上）
        (0.90, 0.10, 0.10, 0.10),   # 对抗路：上边线（右上->左上）
        (0.10, 0.10, 0.10, 0.90),   # 对抗路转泉水：左边线（左上->左下）
        (0.10, 0.90, 0.90, 0.10),   # 中路：对角线（左下->右上）
    ]
    # 中路与边线交汇处加宽（野区入口开阔）
    for (x0, y0, x1, y1) in lanes:
        p0 = (int(x0 * c), int(y0 * c))
        p1 = (int(x1 * c), int(y1 * c))
        for gy in range(n):
            for gx in range(n):
                if lane_band(gx, gy, *p0, *p1, half=3):
                    g[gy, gx] = 1.0
    # 基地区（蓝左下 / 红右上）：2x2 可走
    g[n - 3:n - 1, 1:3] = 1.0
    g[1:3, n - 3:n - 1] = 1.0
    # 河道（斜穿，与中路大致平行但略偏）：标记为 2（可走但代价高）
    # 河道约从左上(0.25,0.25)到右下(0.75,0.75) 弧形 -> 简化为对角带
    for gy in range(n):
        for gx in range(n):
            if lane_band(gx, gy, int(0.25 * c), int(0.25 * c),
                         int(0.75 * c), int(0.75 * c), half=2):
                if g[gy, gx] == 0.0:
                    g[gy, gx] = 2.0
    return g


def load_terrain(path="configs/terrain_grid.npy", fallback_template=True):
    """加载地形栅格；文件缺失时用王者峡谷模板。"""
    import os
    p = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), path)
    if os.path.exists(p):
        return np.load(p)
    return build_king_canyon_template()


if __name__ == "__main__":
    g = build_king_canyon_template()
    print(f"模板可走率 {g[g > 0].size / g.size * 100:.0f}% 河道 {g[g == 2].size / g.size * 100:.0f}%")
    for gy in range(0, 40, 2):
        row = "".join("R" if g[gy, gx] == 2 else "#" if g[gy, gx] == 1 else "."
                      for gx in range(40))
        print(f"{gy:2d} {row}")
