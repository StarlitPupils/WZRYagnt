# -*- coding: utf-8 -*-
"""v10.16 操作-状态-得分记录器 (自我对弈/离线学习基础)。

每步记录:
  状态向量: hp, mp, 技能1/2/3 CD(ready), 召唤师, 屏内敌/友数, 红蓝绿点,
            hook_pending, 死亡, 行为标签, 阵营
  操作: type/id/reason
  得分: 当前总分 total, 本步得分变化 delta, 触发得分的事件名(如 kill/died/assist)
        -> 归因"哪个操作得了/扣了分"

自对弈学习: 状态 -> 操作 -> 得分变化, 学习"哪个状态下哪操作得分高"。
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
_last_total = None      # 上一步总分
_last_evt = None        # 最近得分事件 (由外部 set_event 写入)


def _ensure_session():
    global _session, _fh
    if _session is None:
        _session = time.strftime("%Y%m%d_%H%M%S")
        d = ROOT / "data" / "selfplay"
        d.mkdir(parents=True, exist_ok=True)
        _fh = open(d / f"{_session}.jsonl", "a", encoding="utf-8")
    return _fh


def set_event(event, score_delta, now=None):
    """外部: 触发得分事件时告知 (kill+20 等), 归因到下一步操作。"""
    global _last_evt
    with _lock:
        _last_evt = {"event": event, "delta": round(float(score_delta), 1),
                     "t": round(now or time.time(), 2)}


def record(state_dict, cooldowns, action, reward_total, now=None):
    """记录一步 (状态, 操作, 当前总分, 本步得分变化+事件)。"""
    global _cnt, _last_total, _last_evt
    try:
        fh = _ensure_session()
        ui = state_dict.get("ui") or {}
        mm = state_dict.get("minimap") or {}
        dots = (mm.get("dots") or {}) if mm.get("found") else {}
        units = state_dict.get("units") or []
        enemies_scr = [u for u in units if str(u.get("cls", "")) == "enemy_hero"]
        friendly_scr = [u for u in units if str(u.get("cls", "")) == "ally_hero"]
        skill_st = ui.get("skill_states") or {}
        total = float(reward_total) if reward_total is not None else 0.0
        # 本步得分变化 (相对上一步), 与得分事件归属
        delta = round(total - _last_total, 2) if _last_total is not None else 0.0
        evt = _last_evt
        _last_evt = None
        _last_total = total
        rec = {
            "t": round(now or time.time(), 2),
            "hp": (float(ui.get("hp")) if ui.get("hp") is not None else None),
            "mp": (float(ui.get("mp")) if ui.get("mp") is not None else None),
            "sk1_ready": (skill_st.get("1") or {}).get("ready"),
            "sk2_ready": (skill_st.get("2") or {}).get("ready"),
            "sk3_ready": (skill_st.get("3") or {}).get("ready"),
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
            # v10.16 得分: 当前总分 + 本步变化 + 归因事件
            "total": round(total, 2),
            "delta": delta,
            "event": evt,
        }
        with _lock:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            _cnt += 1
    except Exception:
        pass
