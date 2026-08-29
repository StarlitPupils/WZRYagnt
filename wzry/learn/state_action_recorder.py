# -*- coding: utf-8 -*-
"""v10.14 操作-状态记录器 (自我对弈/离线学习基础)。

记录每步: (状态向量, 操作, 时间戳) 到 data/selfplay/{session_id}.jsonl
状态向量: hp, mp, 我方血量状态, 技能1/2/3 CD(ready/unlocked), 召唤师CD,
          敌英在屏内数+最近敌距离, 我方范围内敌数, 敌红点数, 队友数, 行为标签,
          敌/友血条状态, 死亡/低血标记, 位置(绿点)
操作: type/id/reason
用法: from wzry.learn.state_action_recorder import record
"""
import json
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
_lock = threading.Lock()
_session = None
_fh = None
_cnt = 0


def _ensure_session():
    global _session, _fh
    if _session is None:
        _session = time.strftime("%Y%m%d_%H%M%S")
        d = ROOT / "data" / "selfplay"
        d.mkdir(parents=True, exist_ok=True)
        _fh = open(d / f"{_session}.jsonl", "a", encoding="utf-8")
    return _fh


def record(state_dict, cooldowns, action, now=None):
    """记录一步 (状态, 操作)。"""
    global _cnt
    try:
        fh = _ensure_session()
        ui = state_dict.get("ui") or {}
        mm = state_dict.get("minimap") or {}
        dots = (mm.get("dots") or {}) if mm.get("found") else {}
        units = state_dict.get("units") or []
        enemies_scr = [u for u in units if str(u.get("cls", "")) == "enemy_hero"]
        friendly_scr = [u for u in units if str(u.get("cls", "")) == "ally_hero"]
        skill_st = ui.get("skill_states") or {}
        # 状态向量 (宽泛, 供后续学习)
        rec = {
            "t": round(now or time.time(), 2),
            "hp": (float(ui.get("hp")) if ui.get("hp") is not None else None),
            "mp": (float(ui.get("mp")) if ui.get("mp") is not None else None),
            "sk1_cd": (skill_st.get("1") or {}).get("ready"),
            "sk2_cd": (skill_st.get("2") or {}).get("ready"),
            "sk3_cd": (skill_st.get("3") or {}).get("ready"),
            "n_enemy_scr": len(enemies_scr),
            "n_ally_scr": len(friendly_scr),
            "n_red_mm": len(dots.get("red") or []),
            "n_blue_mm": len(dots.get("blue") or []),
            "n_green_mm": len(dots.get("green") or []),
            "hook_pending": bool(cooldowns.get("hook_pending")),
            "dead": bool((state_dict.get("behavior") or {}).get("dead")),
            "label": (state_dict.get("behavior") or {}).get("label", ""),
            "camp": cooldowns.get("camp") or "?",
            "action_type": action.get("type", "") if isinstance(action, dict) else "",
            "action_id": action.get("id", None) if isinstance(action, dict) else None,
            "reason": action.get("reason", "") if isinstance(action, dict) else "",
        }
        with _lock:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            _cnt += 1
    except Exception:
        pass
