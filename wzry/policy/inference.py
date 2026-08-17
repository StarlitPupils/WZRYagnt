# -*- coding: utf-8 -*-
"""BCNet 推理封装：GameState -> 动作原语 dict（Agent v3 的决策核心）。

解码规则：
  - move_theta: 8 向 log_softmax -> argmax -> bin 中心弧度（0=右，逆时针）
  - move_r:     sigmoid -> 摇杆幅度
  - act:        8 类 ['none','skill1','skill2','skill3','attack','buy','recall','summoner']
                -> argmax；none 时不产生动作
  - target:     sigmoid 目标点（保留，当前执行器用智能施法 tap）
"""
import math
from pathlib import Path

import numpy as np
import torch

from wzry.policy.model import BCNet
from wzry.train.encoding import encode_state

ACT_NAMES = ["none", "skill1", "skill2", "skill3", "attack", "buy", "recall", "summoner"]
SKILL_MAP = {"skill1": 1, "skill2": 2, "skill3": 3}
BIN_ANGLE = 2 * math.pi / 8


class BCNetInference:
    def __init__(self, checkpoint, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.net = BCNet()
        state = torch.load(str(checkpoint), map_location=self.device)
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        self.net.load_state_dict(state)
        self.net.eval().to(self.device)
        self.last_infer_ms = 0.0

    def decide(self, state_dict: dict) -> dict:
        """GameState dict -> 动作 dict（与 Agent v2 decide 输出同构）。"""
        import time
        enc = encode_state(state_dict)
        units = torch.from_numpy(enc["units"]).unsqueeze(0).to(self.device)
        mask = torch.from_numpy(enc["unit_mask"]).unsqueeze(0).to(self.device)
        grid = torch.from_numpy(enc["grid"]).unsqueeze(0).to(self.device)
        ui = torch.from_numpy(enc["ui"]).unsqueeze(0).to(self.device)
        t0 = time.perf_counter()
        with torch.no_grad():
            out = self.net(units, mask, grid, ui)
        self.last_infer_ms = (time.perf_counter() - t0) * 1000.0

        move_bin = int(out["move_theta"].argmax(-1).item())
        theta = (move_bin + 0.5) * BIN_ANGLE  # bin 中心
        r = float(out["move_r"].squeeze().item())
        act_i = int(out["act"].argmax(-1).item())
        act_name = ACT_NAMES[act_i]

        action = {"theta": theta, "r": max(0.05, min(1.0, r))}
        if act_name in SKILL_MAP:
            action["skill_id"] = SKILL_MAP[act_name]
            action["reason"] = f"bc:skill{action['skill_id']}"
        elif act_name == "attack":
            action["attack"] = True
            action["reason"] = "bc:attack"
        else:
            action["reason"] = f"bc:{act_name}"
        action["move_bin"] = move_bin
        return action
