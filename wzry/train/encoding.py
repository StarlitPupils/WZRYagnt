# -*- coding: utf-8 -*-
"""GameState -> 训练特征编码器（行为克隆数据工厂 v0）。

设计：
  - 单位编码：固定长度矩阵（n_units_max x UNIT_DIM），含类别 one-hot(11)、
    屏幕归一化坐标、尺寸、置信度；不足补零 + mask 向量。
  - 小地图栅格：40x40x3（蓝/红/黄圆点密度），对应全局视野（含视野外）。
  - UI 向量：金币/等级/血条/技能冷却等标量（缺失补 -1）。
  - 动作编码：one-hot 类型 + 连续参数（theta/r/目标点）。

输出：单帧特征 dict；states_to_dataset 把 data/matches/*/states.jsonl
批量转为 npz（供 torch Dataset 使用）。
"""
import json
from pathlib import Path

import numpy as np

CLASSES = ["enemy_hero", "ally_hero", "enemy_minion", "ally_minion",
           "enemy_turret", "ally_turret", "enemy_crystal", "ally_crystal",
           "neutral_monster", "hook_aim", "skill_effect"]
CLS2ID = {c: i for i, c in enumerate(CLASSES)}

UNIT_DIM = 11 + 4 + 1          # one-hot + (cx,cy,w,h) + conf
GRID = 40
UI_KEYS = ["gold", "level", "hp", "skill_cd", "kills", "deaths"]


def encode_state(st: dict, n_units_max: int = 20) -> dict:
    """GameState dict -> 特征。返回 {"units": (n,UNIT_DIM) float32, "unit_mask": (n,),
    "grid": (GRID,GRID,3) float32, "ui": (len(UI_KEYS),) float32, "meta": dict}。"""
    units = np.zeros((n_units_max, UNIT_DIM), dtype=np.float32)
    mask = np.zeros((n_units_max,), dtype=np.float32)
    for i, u in enumerate(st.get("units", [])[:n_units_max]):
        cls = u.get("cls", "")
        cid = CLS2ID.get(cls, -1)
        if cid >= 0:
            units[i, cid] = 1.0
        scr = u.get("screen") or [0, 0, 0, 0]
        units[i, 11:15] = scr[:4]
        units[i, 15] = 1.0   # conf 占位（检测器暂未输出到 state）
        mask[i] = 1.0

    grid = np.zeros((GRID, GRID, 3), dtype=np.float32)
    mm = st.get("minimap") or {}
    if mm.get("found"):
        for c, ch in (("blue", 0), ("red", 1), ("yellow", 2)):
            for (nx, ny) in mm.get("dots", {}).get(c, []):
                gx = min(GRID - 1, max(0, int(nx * GRID)))
                gy = min(GRID - 1, max(0, int(ny * GRID)))
                grid[gy, gx, ch] += 1.0

    ui = np.full((len(UI_KEYS),), -1.0, dtype=np.float32)
    ui_map = st.get("ui") or {}
    for i, k in enumerate(UI_KEYS):
        v = ui_map.get(k)
        if isinstance(v, (int, float)):
            ui[i] = float(v)

    return {"units": units, "unit_mask": mask, "grid": grid, "ui": ui,
            "meta": {"t": st.get("t"), "frame_id": st.get("frame_id"),
                     "phase": st.get("phase")}}


def encode_action(act: dict, n_actions: int = 6) -> np.ndarray:
    """动作 dict -> 向量。act: {"type": "move|skill|attack|buy|recall|none", ...}
    输出长度 n_actions + 4（theta, r, target_x, target_y 归一化）。"""
    types = ["move", "skill", "attack", "buy", "recall", "none"]
    vec = np.zeros((max(n_actions, len(types)) + 4,), dtype=np.float32)
    t = act.get("type", "none")
    if t in types:
        vec[types.index(t)] = 1.0
    vec[n_actions] = float(act.get("theta", 0.0) or 0.0) / (2 * np.pi + 1e-6)
    vec[n_actions + 1] = float(act.get("r", 0.0) or 0.0)
    vec[n_actions + 2] = float(act.get("target_x", 0.0) or 0.0)
    vec[n_actions + 3] = float(act.get("target_y", 0.0) or 0.0)
    return vec


def states_to_dataset(matches_dir="data/matches", out="data/datasets/bc_v0.npz",
                      max_samples=None, n_units_max=20):
    """扫描 data/matches/*/states.jsonl，编码全部状态为 npz 特征库。

    动作标签来自同会话 actions.jsonl 按时间戳最近匹配（v0 简单对齐：
    动作事件取该状态之后最近一条；无动作样本标 none）。
    """
    matches_dir = Path(matches_dir)
    if not matches_dir.exists():
        raise FileNotFoundError(f"无对局数据: {matches_dir}（先跑 m1_live_pipeline 采集）")

    feats, acts, metas = [], [], []
    for sess in sorted(matches_dir.iterdir()):
        sf = sess / "states.jsonl"
        af = sess / "actions.jsonl"
        if not sf.exists():
            continue
        actions = []
        if af.exists():
            actions = [json.loads(l) for l in af.read_text(encoding="utf-8").splitlines()]
        a_idx = 0
        for line in sf.read_text(encoding="utf-8").splitlines():
            st = json.loads(line)
            enc = encode_state(st, n_units_max=n_units_max)
            t = st.get("t", 0.0)
            while a_idx < len(actions) and actions[a_idx].get("t", 0.0) < t:
                a_idx += 1
            act = actions[a_idx] if a_idx < len(actions) else {"type": "none"}
            feats.append(enc)
            acts.append(encode_action(act))
            metas.append(enc["meta"])
            if max_samples and len(feats) >= max_samples:
                break
        if max_samples and len(feats) >= max_samples:
            break

    if not feats:
        raise RuntimeError("没有可编码的状态样本")

    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
        units=np.stack([f["units"] for f in feats]),
        unit_mask=np.stack([f["unit_mask"] for f in feats]),
        grid=np.stack([f["grid"] for f in feats]),
        ui=np.stack([f["ui"] for f in feats]),
        actions=np.stack(acts),
        metas=np.array([json.dumps(m, ensure_ascii=False) for m in metas]),
    )
    print(f"数据集已写出: {out}  ({len(feats)} 样本, 单位矩阵 {feats[0]['units'].shape}, "
          f"栅格 {feats[0]['grid'].shape})")
    return out
