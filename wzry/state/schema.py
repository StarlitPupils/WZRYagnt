# -*- coding: utf-8 -*-
"""GameState v0 schema（与 docs/PLAN.md §4.2 对齐的落地子集）。

M0 阶段先用最小字段；后续按里程碑逐项扩充（小地图、UI 数值、追踪等）。
"""
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class Unit:
    """视野内单位（含自己）。"""
    id: Optional[int] = None          # 追踪 ID（M1 后可用）
    cls: str = ""                     # 类别：enemy_hero / ally_hero / minion / turret ...
    camp: Optional[str] = None        # blue / red
    screen: Optional[List[float]] = None   # [cx, cy, w, h] 归一化像素坐标
    hp: Optional[float] = None        # 0-1
    dead: bool = False


@dataclass
class GameState:
    """一帧完整游戏状态。"""
    t: float = 0.0                          # 全局时间戳（秒）
    phase: str = "unknown"                  # waiting / in_match / post_match
    frame_id: int = 0
    source: str = "live"                    # live / replay
    screen_size: List[int] = field(default_factory=lambda: [0, 0])
    units: List[Unit] = field(default_factory=list)
    minimap: Dict[str, Any] = field(default_factory=dict)   # M1 后填充
    ui: Dict[str, Any] = field(default_factory=dict)        # M1 后填充
    last_action: Optional[Dict[str, Any]] = None
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self) -> str:
        import json
        return json.dumps(self.to_dict(), ensure_ascii=False)
