# -*- coding: utf-8 -*-
"""局内行为习惯学习器 v1 (自己/对面/队友行为模式自动挖掘)。

局内每 2s 采样小地图蓝/红/绿点 -> 时间序列 -> 每 60s 聚合:
  - 对面: 红点活跃区(常驻路/蹲草点) -> 支援权重线索
  - 队友: 蓝点活跃区(主力在哪路) -> 跟队线索
  - 自己: 位置占用/技能发放节奏 -> 自评线索
写入 configs/behavior_model.json (局内滚动更新, 决策可读)。
"""
import json
import threading
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "configs" / "behavior_model.json"


class BehaviorMiner:
    def __init__(self, sample_s=2.0, agg_s=60.0):
        self.sample_s = sample_s
        self.agg_s = agg_s
        self.lock = threading.Lock()
        self._stop = False
        self.samples = {"red": [], "blue": [], "green": []}
        self._last_sample = 0.0
        self._last_agg = 0.0
        self.model = {"red_active": [], "blue_active": [], "self_zone": [],
                      "updated": 0.0}
        self.thread = threading.Thread(target=self._loop, daemon=True)

    def start(self):
        self.thread.start()

    def stop(self):
        self._stop = True

    def feed(self, mm_dots):
        with self.lock:
            self.last_dots = mm_dots

    def _loop(self):
        while not self._stop:
            try:
                time.sleep(0.5)
                with self.lock:
                    dots = getattr(self, "last_dots", None)
                if not dots:
                    continue
                now = time.time()
                if now - self._last_sample >= self.sample_s:
                    self._last_sample = now
                    with self.lock:
                        for k, key in (("red", "red"), ("blue", "blue"),
                                       ("green", "green")):
                            pts = dots.get(key) or []
                            for p in pts:
                                self.samples[k].append((p[0], p[1], now))
                        # 只保留最近 300s
                        for k in self.samples:
                            self.samples[k] = [s for s in self.samples[k]
                                               if now - s[2] < 300]
                if now - self._last_agg >= self.agg_s:
                    self._last_agg = now
                    self._aggregate(now)
            except Exception:
                pass

    def _aggregate(self, now):
        """聚类最近 300s 轨迹 -> 活跃区(常驻位置)。"""
        with self.lock:
            red = [(p[0], p[1]) for p in self.samples["red"]]
            blue = [(p[0], p[1]) for p in self.samples["blue"]]
            green = [(p[0], p[1]) for p in self.samples["green"]]
            skill_rate = None
        model = {
            "red_active": self._cluster(red, min_pts=4),
            "blue_active": self._cluster(blue, min_pts=4),
            "self_zone": self._cluster(green, min_pts=3),
            "updated": now,
        }
        with self.lock:
            self.model = model
        try:
            OUT.write_text(json.dumps(model, ensure_ascii=False), encoding="utf-8",
                           )
        except Exception:
            pass

    @staticmethod
    def _cluster(pts, d=0.06, min_pts=3):
        out = []
        for x, y in pts:
            for o in out:
                if (o[0] - x) ** 2 + (o[1] - y) ** 2 < d * d:
                    o[0] = (o[0] + x) / 2
                    o[1] = (o[1] + y) / 2
                    o[2] += 1
                    break
            else:
                out.append([x, y, 1])
        return [[round(x, 3), round(y, 3), c] for x, y, c in out if c >= min_pts]
