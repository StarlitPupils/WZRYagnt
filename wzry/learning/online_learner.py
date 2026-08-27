# -*- coding: utf-8 -*-
"""在线学习器 v1 (局中实时进化, 零等待)。

事件驱动更新 optimizer 价值权重:
  kill   -> w_win 增强(持续接战有效)  也把"死亡前3s我在场"判为正样本
  assist -> w_win 微增
  died   -> w_risk 增强(死亡=风险判断不足), 朝敌群权重降
  tower  -> w_risk 微增
队友/对方:
  敌方死亡(event kill/assist)时, 若我有队友在场 -> w_team 增(学跟队接团)
观察窗口: 死亡前3s行为摘要(深入/低血/敌多)计入风险修正。
EMA 平滑 + 上下限夹紧, 防单事件震荡; 线程安全。
"""
import threading
from pathlib import Path

_W_MIN, _W_MAX = 0.3, 3.0


class OnlineLearner:
    def __init__(self, alpha=0.06):
        self._lock = threading.Lock()
        self._w = {
            "w_win": 1.0, "w_safe": 1.2, "w_map": 0.6, "w_risk": 2.0,
            "w_hold": 0.4, "w_skill": 1.4, "w_micro": 0.3, "w_team": 0.6,
            "k_win_enemy": 1.5, "k_win_support": 0.6,
        }
        self.alpha = alpha
        self.episodes = {"kill": 0, "assist": 0, "died": 0, "tower": 0}
        self.ema = 0.92   # 权重 EMA 平滑
        self.entries = []          # v2.67 学习笔记(供落盘文档)
        self.session_start = None

    def _bump(self, key, delta):
        with self._lock:
            cur = self._w.get(key, 0.0)
            nxt = cur * self.ema + (cur + delta) * (1 - self.ema)
            self._w[key] = max(_W_MIN, min(_W_MAX, nxt))

    def _log(self, tag, detail):
        import datetime
        self.entries.append(f"- [{datetime.datetime.now():%H:%M:%S}] {tag}: {detail}")

    def observe(self, event, near_enemies=0, hp_low=False, ally_near=False):
        """局中事件即时学习。"""
        if not event:
            return
        if event == "kill":
            self._bump("w_win", self.alpha * 1.0)
            self._bump("k_win_enemy", self.alpha * 0.8)
            self.episodes["kill"] += 1
            self._log("学对方/自己", "击杀有效 → 接战价值 w_win↑(下次更敢打)")
        elif event == "assist":
            self._bump("w_win", self.alpha * 0.5)
            self.episodes["assist"] += 1
            if ally_near:
                self._bump("w_team", self.alpha * 0.4)
                self._log("学队友", "助攻且队友在场 → 跟队价值 w_team↑(学抱团位置)")
            else:
                self._log("学自己", "助攻有效 → w_win 微增")
        elif event == "died":
            self._bump("w_risk", self.alpha * 0.8)
            reason = "风险判断不足"
            if near_enemies >= 2:
                self._bump("w_risk", self.alpha * 0.5)
                reason += " + 敌多人堆(接战过深)"
            if hp_low:
                self._bump("w_hold", self.alpha * 0.3)
                reason += " + 残血硬拼"
            self.episodes["died"] += 1
            self._log("学自己(教训)", f"死亡 → 风险权重 w_risk↑ ({reason})")
        elif event == "be_tower_attacked":
            self._bump("w_risk", self.alpha * 0.3)
            self._log("学自己(教训)", "被塔打 → 塔圈风险 w_risk↑")
        elif event == "tower_kill":
            self._bump("w_map", self.alpha * 0.3)
            self._log("学我方节奏", "推塔有效 → 兵线价值 w_map↑")
        elif event == "minion_clear":
            self._bump("w_map", self.alpha * 0.05)

    def weights(self):
        with self._lock:
            return dict(self._w)

    def stats(self):
        with self._lock:
            return dict(self.episodes)

    def dump_log(self, path, extra=""):
        """v2.67: 本局学习内容 -> markdown 文档(用户查阅)。60s 节流防闪断刷屏。"""
        import datetime as _dt
        try:
            now_t = _dt.datetime.now()
            with self._lock:
                if hasattr(self, "_last_dump") and \
                        (now_t - self._last_dump).total_seconds() < 60:
                    return False
                self._last_dump = now_t
                w = dict(self._w)
                entries = list(self.entries)
                ep = dict(self.episodes)
                self.entries = []
                self.episodes = {"kill": 0, "assist": 0, "died": 0, "tower": 0}
            lines = [f"## 对局 {now_t:%Y-%m-%d %H:%M}",
                     "",
                     "### 局中学习条目"]
            if entries:
                lines += entries
            else:
                lines.append("- (本局暂无学习事件)")
            lines += ["", "### 进化后权重 (下一局立即生效)",
                      f"``` {w} ```", "", "### 本局战绩",
                      f"- 击杀{ep['kill']} 助攻{ep['assist']} 死亡{ep['died']} 推塔{ep['tower']}"]
            if extra:
                lines += ["", "### 局末复盘", extra]
            lines += ["", "---", ""]
            p = Path(path)
            p.parent.mkdir(parents=True, exist_ok=True)
            with open(p, "a", encoding="utf-8") as f:
                f.write("\n".join(lines) + "\n")
            return True
        except Exception:
            return False
