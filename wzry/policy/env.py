# -*- coding: utf-8 -*-
"""强化学习环境接口骨架（M5 前置）。

GameEnv 把"感知管线输出（GameState）"与"动作执行器（原语）"包成 RL 接口：
  - observation(state)      GameState dict -> 观测张量（复用 train/encoding 编码）
  - step(action_dict)       动作原语 dict -> 执行 -> 返回 (next_state, reward, done)
  - reward(state, prev)     稠密 shaping：击杀/推塔/经济/死亡/时间惩罚（v0）

v0 定位：接口与奖励定义先行；在线执行由 Agent 循环调用（无需 gym 依赖）。
"""
import time

from wzry.train.encoding import encode_state

# 奖励权重（v0 默认，后续 RL 阶段消融调优）
REWARD_W = {
    "win": 10.0, "death": -2.0, "kill": 1.5, "assist": 0.8,
    "turret": 2.0, "gold_delta": 0.001, "time_penalty": -0.01,
}


def compute_reward(prev: dict, curr: dict, w: dict = None) -> float:
    """从两帧 GameState 计算 shaping 奖励。

    prev/curr 为 GameState dict（含 ui.kills/deaths/assists、minimap.towers、ui.gold、t）。
    """
    w = w or REWARD_W
    r = w["time_penalty"]
    pu, cu = prev.get("ui") or {}, curr.get("ui") or {}
    pk, pd, pa = pu.get("kills", 0), pu.get("deaths", 0), pu.get("assists", 0)
    ck, cd, ca = cu.get("kills", 0), cu.get("deaths", 0), cu.get("assists", 0)
    r += (ck - pk) * w["kill"] + (cd - pd) * w["death"] + (ca - pa) * w["assist"]

    pg, cg = pu.get("gold", 0), cu.get("gold", 0)
    r += (cg - pg) * w["gold_delta"]

    # 塔数变化（小地图 towers 的敌/我塔状态，v0 简化为计数差）
    pt, ct = prev.get("minimap") or {}, curr.get("minimap") or {}
    r += (len(ct.get("towers", [])) - len(pt.get("towers", []))) * w["turret"] * 0  # 占位
    return round(float(r), 4)


class GameEnv:
    """RL 环境骨架：观测编码 + 奖励计算 + 动作接口。

    execute_action 由 Agent 循环注入（在线环境），保持无 gym 依赖。
    """

    def __init__(self, executor=None, reward_w: dict = None):
        self.executor = executor          # ActionExecutor（M2）
        self.reward_w = reward_w or REWARD_W
        self.prev_state = None
        self._t = time.time()

    def observation(self, state: dict):
        """GameState dict -> 观测张量（BCNet 输入布局）。"""
        return encode_state(state)

    def reward(self, state: dict) -> float:
        """相对上一帧的 shaping 奖励；首帧返回 0。"""
        if self.prev_state is None:
            self.prev_state = state
            return 0.0
        r = compute_reward(self.prev_state, state, self.reward_w)
        self.prev_state = state
        return r

    def execute_action(self, action: dict):
        """执行动作原语 dict（见 ActionExecutor.sequence 格式）。"""
        if self.executor is None:
            raise RuntimeError("未注入 executor（在线环境需要 ActionExecutor）")
        self.executor._exec_dict(action)

    def reset(self):
        self.prev_state = None
        self._t = time.time()
