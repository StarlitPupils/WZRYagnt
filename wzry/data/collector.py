# -*- coding: utf-8 -*-
"""对局采集器：按"对局会话"组织存档，供模仿学习数据工厂直接消费。

会话结构（data/matches/<session_id>/）：
  states.jsonl   逐条 GameState JSON（含时间戳、检测、小地图、UI）
  frames/        关键帧 PNG（默认每 5 秒一帧，可配）
  actions.jsonl  动作事件流（由执行器写入，按时间戳与 states 对齐）
  manifest.json  会话元信息（起止时间、时长、帧数、状态条数、视频源等）

用法（集成到感知管线）：
    rec = MatchRecorder()
    rec.on_state(st)            # 对局中每帧/每周期调用
    rec.on_frame(frame)         # 可选：保存关键帧
    rec.on_action(act_dict)     # 可选：动作事件
    rec.close()                 # 对局结束（或进程退出）
"""
import json
import time
from datetime import datetime
from pathlib import Path


class MatchRecorder:
    def __init__(self, base_dir="data/matches", frame_every_s=5.0, max_frames=2000):
        self.base_dir = Path(base_dir)
        self.frame_every_s = frame_every_s
        self.max_frames = max_frames
        self.session_dir = None
        self._states_fp = None
        self._actions_fp = None
        self._frame_dir = None
        self._n_states = 0
        self._n_frames = 0
        self._start_t = None
        self._last_frame_t = 0.0

    @property
    def active(self):
        return self._states_fp is not None

    def start(self, meta=None, session_name=None):
        """开启一个新对局会话。session_name 可指定会话目录名（默认时间戳）。"""
        if self.active:
            return
        ts = session_name or datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        self.session_dir = self.base_dir / ts
        self.session_dir.mkdir(parents=True, exist_ok=True)
        self._frame_dir = self.session_dir / "frames"
        self._frame_dir.mkdir(exist_ok=True)
        self._states_fp = open(self.session_dir / "states.jsonl", "a", encoding="utf-8")
        self._actions_fp = open(self.session_dir / "actions.jsonl", "a", encoding="utf-8")
        self._n_states = 0
        self._n_frames = 0
        self._start_t = time.time()
        self._last_frame_t = 0.0
        if meta:
            self._write_manifest({"meta": meta})

    def on_state(self, state_dict):
        """写入一条 GameState（dict）。"""
        if not self.active:
            return
        self._states_fp.write(json.dumps(state_dict, ensure_ascii=False,
                                         default=self._json_default) + "\n")
        self._n_states += 1

    @staticmethod
    def _json_default(o):
        """numpy 标量/数组 -> 原生类型（检测/血条等字段常带 numpy intc/float32）。"""
        import numpy as np
        if isinstance(o, np.generic):
            return o.item()
        if isinstance(o, np.ndarray):
            return o.tolist()
        raise TypeError(f'Object of type {o.__class__.__name__} is not JSON serializable')

    def on_frame(self, frame):
        """按 frame_every_s 节流保存关键帧 PNG。"""
        if not self.active or self._n_frames >= self.max_frames:
            return
        now = time.time()
        if now - self._last_frame_t < self.frame_every_s:
            return
        self._last_frame_t = now
        import cv2
        fn = self._frame_dir / f"frame_{self._n_frames:06d}.png"
        cv2.imwrite(str(fn), frame)
        self._n_frames += 1

    def on_action(self, act_dict):
        """写入一条动作事件（与执行器事件流同构）。"""
        if not self.active:
            return
        self._actions_fp.write(json.dumps(act_dict, ensure_ascii=False,
                                          default=self._json_default) + "\n")

    def _write_manifest(self, extra=None):
        if not self.session_dir:
            return
        manifest = {
            "session": self.session_dir.name,
            "started_at": datetime.fromtimestamp(self._start_t).isoformat()
            if self._start_t else None,
            "n_states": self._n_states,
            "n_frames": self._n_frames,
        }
        if extra:
            manifest.update(extra)
        (self.session_dir / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    def close(self):
        """结束会话并写清单。"""
        if not self.active:
            return
        self._write_manifest()
        self._states_fp.close()
        self._actions_fp.close()
        self._states_fp = None
        self._actions_fp = None
        self.session_dir = None
