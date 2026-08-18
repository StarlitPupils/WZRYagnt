# -*- coding: utf-8 -*-
"""墙体识别与寻路模块单元测试：terrain（栅格/A*/定位）+ wall_sensor（撞墙感知）。

运行：
    venv\\Scripts\\python.exe scripts\\m2_agent_v2_test.py  （已含）
    单独：venv\\Scripts\\python.exe -m unittest wzry.tests.test_terrain
"""
import math
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from wzry.vision.terrain import (  # noqa: E402
    astar_path,
    extract_walk_grid,
    find_minimap_box,
    path_to_moves,
    DEFAULT_BOX,
)
from wzry.vision.wall_sensor import WallSensor  # noqa: E402


def make_frame(box=DEFAULT_BOX):
    """生成一张模拟小地图的帧：方形框内亮色道路+蓝色河道+暗色墙体。"""
    f = np.full((720, 1280, 3), 200, np.uint8)   # 亮背景（道路）
    x0, y0, x1, y1 = box
    # 框内：大部分暗色（墙体）
    f[y0:y1, x0:x1] = (40, 40, 40)
    # 一条水平道路（亮）
    f[y0 + y1 // 2 - 5:y0 + y1 // 2 + 5, x0:x1] = (180, 180, 180)
    # 一条垂直蓝色河道
    f[y0:y1, x0 + x1 // 2 - 5:x0 + x1 // 2 + 5] = (180, 100, 60)  # BGR 蓝
    return f


class TestFindBox(unittest.TestCase):
    def test_find_minimap_box_returns_default(self):
        f = make_frame()
        box = find_minimap_box(f)
        self.assertIsNotNone(box)
        self.assertEqual(box, DEFAULT_BOX)


class TestExtractGrid(unittest.TestCase):
    def test_grid_shape_and_values(self):
        f = make_frame()
        grid, box = extract_walk_grid(f)
        self.assertIsNotNone(grid)
        self.assertEqual(grid.shape, (40, 40))
        self.assertIn(0.0, grid)
        self.assertIn(1.0, grid)

    def test_road_is_walkable(self):
        f = make_frame()
        grid, _ = extract_walk_grid(f)
        # 水平道路行（网格 20 附近）应有可走格
        row = grid[19:22, :]
        self.assertGreater(row.sum(), 10)

    def test_wall_not_walkable(self):
        f = make_frame()
        grid, _ = extract_walk_grid(f)
        # 暗色墙体角区域（如网格 5,5）应不可走
        self.assertEqual(grid[5, 5], 0.0)


class TestAStar(unittest.TestCase):
    def test_path_found_on_road(self):
        grid = np.zeros((20, 20), np.float32)
        grid[10, 2:18] = 1.0   # 水平道路
        path = astar_path(grid, (10, 2), (10, 17))
        self.assertIsNotNone(path)
        self.assertEqual(path[0], (10, 2))
        self.assertEqual(path[-1], (10, 17))

    def test_no_path_through_wall(self):
        grid = np.zeros((20, 20), np.float32)
        grid[10, 2:18] = 1.0
        grid[10, 9] = 0.0      # 断点
        grid[9, 9] = 1.0       # 上方可绕行
        path = astar_path(grid, (10, 2), (10, 17))
        # 有绕行路径（上方）
        self.assertIsNotNone(path)
        self.assertIn((9, 9), path)

    def test_no_path_blocked(self):
        grid = np.zeros((10, 10), np.float32)
        grid[0, 0] = 1.0
        grid[9, 9] = 1.0       # 全墙分隔
        path = astar_path(grid, (0, 0), (9, 9))
        self.assertIsNone(path)

    def test_river_has_cost(self):
        grid = np.zeros((10, 10), np.float32)
        grid[5, 1:9] = 1.0
        grid[5, 5] = 2.0       # 河道
        # 直穿河道 vs 绕行：走河道更短但贵，仍应找到路径
        path = astar_path(grid, (5, 1), (5, 8))
        self.assertIsNotNone(path)


class TestPathToMoves(unittest.TestCase):
    def test_moves_generated(self):
        path = [(5, 5), (5, 6), (5, 7)]   # 向右
        moves = path_to_moves(path)
        self.assertEqual(len(moves), 2)
        theta = moves[0][0]
        self.assertAlmostEqual(math.cos(theta), 1.0, places=4)


class TestWallSensor(unittest.TestCase):
    def test_no_wall_when_moving(self):
        ws = WallSensor(still_px=0.02, frames=3)
        # 每次移动超过阈值
        hit = False
        for i in range(10):
            hit = ws.update(float(i), True, (0.1 + i * 0.05, 0.5))
        self.assertFalse(hit)

    def test_wall_detected_when_stuck(self):
        ws = WallSensor(still_px=0.02, frames=3)
        hits = []
        for i in range(10):
            hits.append(ws.update(float(i), True, (0.1, 0.5)))   # 位置不动
        self.assertTrue(any(hits))

    def test_avoid_action_theta(self):
        ws = WallSensor()
        act = ws.avoid_action(0.0)
        self.assertEqual(act["type"], "move")
        self.assertEqual(act["reason"], "wall_avoid")
        self.assertAlmostEqual(abs(act["theta"]), math.pi / 2, places=4)

    def test_reset_when_stop_moving(self):
        ws = WallSensor(still_px=0.02, frames=3)
        for i in range(5):
            ws.update(float(i), True, (0.1, 0.5))   # 卡住
        # 停止移动后重置
        for i in range(5, 10):
            ws.update(float(i), False, None)
        # 再动起来应正常（首帧仅记录位置）
        hit = ws.update(10.0, True, (0.2, 0.5))
        self.assertFalse(hit)
        hit2 = ws.update(11.0, True, (0.3, 0.5))
        self.assertFalse(hit2)


if __name__ == "__main__":
    unittest.main()
