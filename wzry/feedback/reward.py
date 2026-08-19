# -*- coding: utf-8 -*-
"""反馈层：奖励/惩罚机制（用户规则 v3.0）。

三层架构第三层：理解层输出行为标签 -> 决策层决策 -> 反馈层按事件奖惩，
累计分数用于评估/调优决策。

用户给定系数（事件触发计分，基础 100 分制）：
  - 胜利            +80
  - 失败            -80
  - 勾到人并击杀    +30
  - 击杀            +20
  - 勾到人          +10
  - 阵亡            -50
其余事件系数默认 0，用户后续给出后在此表补充。

用法：
    rw = RewardSystem()
    rw.on_event("hook")          # 勾到人 +10
    rw.on_event("died")          # 阵亡 -50
    print(rw.total, rw.history)
"""
import json
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

# 事件系数表（用户给定；基础 100 分制）
EVENT_SCORES = {
    "victory": 80,       # 我胜利了
    "defeat": -80,       # 我失败了
    "hook_kill": 30,     # 勾到人并击杀
    "kill": 20,          # 击杀敌方英雄
    "hook": 10,          # 勾到人
    "died": -50,         # 我阵亡了
    # ---- 待用户补充系数（暂 0）----
    "tower_kill": 0,     # 推掉防御塔
    "be_attacked": 0,    # 我被攻击了
    "be_tower_attacked": 0,  # 我被防御塔攻击了
    "assist": 0,         # 助攻
}

# 事件去重窗口（秒）：同一事件在窗口内不重复计分
EVENT_DEDUP_S = {
    "victory": 60, "defeat": 60, "hook_kill": 3.0, "kill": 3.0,
    "hook": 2.0, "died": 5.0, "tower_kill": 3.0,
    "be_attacked": 1.0, "be_tower_attacked": 1.0, "assist": 3.0,
}


class RewardSystem:
    """事件触发式奖惩（用户规则：事件触发计分）。

    记录：
      - self.total        累计总分
      - self.history      事件历史 [(t, event, score)]
      - data/rewards.jsonl 持久化（追加）
    """

    def __init__(self, log_path=None):
        self.total = 0.0
        self.history = []
        self._last_event_t = {}
        self.log_path = Path(log_path) if log_path else ROOT / "data" / "rewards.jsonl"
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

    def on_event(self, event: str, meta: dict = None, now: float = None) -> float:
        """触发事件计分。窗口内重复事件去重，返回本次得分（0=未计分）。"""
        score = float(EVENT_SCORES.get(event, 0.0))
        now = now if now is not None else time.time()
        dedup = EVENT_DEDUP_S.get(event, 1.0)
        if now - self._last_event_t.get(event, -1e9) < dedup:
            return 0.0
        self._last_event_t[event] = now
        self.total += score
        rec = {"t": now, "event": event, "score": score, "total": round(self.total, 1)}
        if meta:
            rec["meta"] = meta
        self.history.append(rec)
        try:
            with open(self.log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        except Exception:
            pass
        return score

    def __repr__(self):
        return f"<RewardSystem total={self.total:.1f} events={len(self.history)}>"
