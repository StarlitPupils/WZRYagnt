# -*- coding: utf-8 -*-
"""策略网络 v0（行为克隆骨架）：结构化 GameState -> 动作分布。

输入（与 wzry/train/encoding.py 对齐）：
  units     (B, N=20, 16)  单位特征（one-hot 11 + 屏幕坐标 4 + conf 1）
  unit_mask (B, N)
  grid      (B, 40, 40, 3) 小地图栅格（蓝/红/黄密度）
  ui        (B, 6)         UI 数值（gold/level/hp/skill_cd/kills/deaths，缺失 -1）

输出：
  move_theta  (B, 8)   8 向移动离散分布（行为克隆常用离散化）
  move_r      (B, 1)   移动幅度（sigmoid）
  act         (B, 8)   离散动作分布：none/skill1/skill2/skill3/attack/buy/recall/summoner
  target      (B, 2)   技能目标点（归一化 0-1，sigmoid）

训练阶段仅需前向；M5 强化学习时复用 encoder 与 head。
"""
import torch
import torch.nn as nn

N_UNITS = 20
UNIT_DIM = 16
GRID = 40
UI_DIM = 6
N_MOVE_DIR = 8
N_ACT = 10   # move/skill1/skill2/skill3/attack/buy/recall/summoner/restore/none（与 encode_action 对齐）


class UnitEncoder(nn.Module):
    """单位列表编码：per-unit MLP -> 平均池化（mask 感知）。"""

    def __init__(self, hidden=64):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(UNIT_DIM, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )

    def forward(self, units, mask):
        # units: (B,N,16), mask: (B,N)
        x = self.mlp(units)                       # (B,N,64)
        m = mask.unsqueeze(-1)                    # (B,N,1)
        s = (x * m).sum(dim=1)                    # (B,64)
        n = m.sum(dim=1).clamp(min=1.0)
        return s / n


class GridEncoder(nn.Module):
    """小地图栅格编码：小 CNN。"""

    def __init__(self, out_dim=64):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),   # 20x20
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),  # 10x10
            nn.Conv2d(32, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),  # 5x5
            nn.Flatten(),
            nn.Linear(32 * 5 * 5, out_dim), nn.ReLU(),
        )

    def forward(self, grid):
        # grid: (B,40,40,3) -> (B,3,40,40)
        x = grid.permute(0, 3, 1, 2)
        return self.cnn(x)


class BCNet(nn.Module):
    def __init__(self, hidden=128):
        super().__init__()
        self.unit_enc = UnitEncoder(hidden)
        self.grid_enc = GridEncoder(hidden)
        self.fc = nn.Sequential(
            nn.Linear(hidden * 2 + UI_DIM, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        self.move_theta = nn.Linear(hidden, N_MOVE_DIR)
        self.move_r = nn.Linear(hidden, 1)
        self.act = nn.Linear(hidden, N_ACT)
        self.target = nn.Linear(hidden, 2)

    def forward(self, units, unit_mask, grid, ui):
        u = self.unit_enc(units, unit_mask)        # (B,hidden)
        g = self.grid_enc(grid)                    # (B,hidden)
        x = self.fc(torch.cat([u, g, ui], dim=-1))
        return {
            "move_theta": torch.log_softmax(self.move_theta(x), dim=-1),
            "move_r": torch.sigmoid(self.move_r(x)),
            "act": torch.log_softmax(self.act(x), dim=-1),
            "target": torch.sigmoid(self.target(x)),
        }


if __name__ == "__main__":
    # 冒烟测试：随机输入跑通前向
    torch.manual_seed(0)
    net = BCNet()
    B = 4
    units = torch.randn(B, N_UNITS, UNIT_DIM)
    units[:, :, :11] = torch.softmax(units[:, :, :11], dim=-1)  # one-hot-ish
    mask = (torch.rand(B, N_UNITS) > 0.5).float()
    grid = torch.rand(B, GRID, GRID, 3)
    ui = torch.randn(B, UI_DIM)
    out = net(units, mask, grid, ui)
    for k, v in out.items():
        print(f"{k:<12} {tuple(v.shape)}")
    assert out["move_theta"].shape == (B, N_MOVE_DIR)
    assert out["act"].shape == (B, N_ACT)
    assert out["target"].shape == (B, 2)
    # 概率和应为 1
    assert torch.allclose(out["move_theta"].exp().sum(-1), torch.ones(B), atol=1e-4)
    assert torch.allclose(out["act"].exp().sum(-1), torch.ones(B), atol=1e-4)
    print("BCNet 冒烟测试通过")
