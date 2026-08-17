# -*- coding: utf-8 -*-
"""动作执行器（M0 单实例）。

控制通道选择：
  - adb（默认）：常驻 adb server + `adb -s <serial> shell input ...`，
    比旧 touch.py 每次 subprocess.run('adb') 更快，且带序列号直连。
  - mumu（备用）：mumu-python-api（MuMuManager 封装，每次调用起子进程，仅作对照）。

所有动作都写入事件流（JSON Lines，带时间戳），供回放、延迟统计与训练对齐使用。
"""
import subprocess
import threading
import time
from pathlib import Path

ADB = "adb"


def discover_mumu_adb_devices():
    """探测已连接的 MuMu 模拟器 adb 序列号（优先 127.0.0.1:*）。"""
    try:
        out = subprocess.run([ADB, "devices"], capture_output=True, text=True, timeout=5).stdout
    except Exception:
        return []
    serials = [ln.split("\t")[0] for ln in out.splitlines()[1:] if "\tdevice" in ln]
    loop = [s for s in serials if s.startswith("127.0.0.1:")]
    return loop or serials


class AdbExecutor:
    """基于 adb 的触摸执行器：tap / swipe / key，含延迟统计与动作事件流。"""

    def __init__(self, serial=None, event_log=None):
        self.serial = serial or self._auto_serial()
        self.event_log = Path(event_log) if event_log else None
        self._lock = threading.Lock()
        self._event_fp = None
        self.last_cmd_ms = 0.0
        self.cmd_count = 0

    def _auto_serial(self):
        serials = discover_mumu_adb_devices()
        if not serials:
            raise RuntimeError("未发现 MuMu ADB 设备，请先启动模拟器并确认 adb 已连接")
        return serials[0]

    # ---------- 基础命令 ----------
    def _run(self, args):
        t0 = time.perf_counter()
        with self._lock:
            r = subprocess.run(
                [ADB, "-s", self.serial] + args,
                capture_output=True, text=True, timeout=3,
            )
        self.last_cmd_ms = (time.perf_counter() - t0) * 1000.0
        self.cmd_count += 1
        return r

    # ---------- 动作原语 ----------
    def tap(self, x, y, source="policy"):
        x, y = int(round(x)), int(round(y))
        r = self._run(["shell", "input", "tap", str(x), str(y)])
        self._log({"type": "tap", "x": x, "y": y, "source": source,
                   "latency_ms": round(self.last_cmd_ms, 2)})
        return r.returncode == 0

    def swipe(self, x1, y1, x2, y2, duration=100, source="policy"):
        x1, y1, x2, y2 = (int(round(v)) for v in (x1, y1, x2, y2))
        r = self._run(["shell", "input", "swipe",
                       str(x1), str(y1), str(x2), str(y2), str(int(duration))])
        self._log({"type": "swipe", "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                   "duration": int(duration), "source": source,
                   "latency_ms": round(self.last_cmd_ms, 2)})
        return r.returncode == 0

    def key(self, keycode, source="policy"):
        r = self._run(["shell", "input", "keyevent", str(keycode)])
        self._log({"type": "key", "keycode": str(keycode), "source": source,
                   "latency_ms": round(self.last_cmd_ms, 2)})
        return r.returncode == 0

    def shell(self, cmd):
        """任意 shell 命令（如 getprop 测试往返延迟）。"""
        r = self._run(["shell"] + cmd.split())
        return r

    # ---------- 事件流 ----------
    def _log(self, event):
        event = {"t": time.time(), **event}
        if self.event_log:
            if self._event_fp is None:
                self.event_log.parent.mkdir(parents=True, exist_ok=True)
                self._event_fp = open(self.event_log, "a", encoding="utf-8")
            self._event_fp.write(__import__("json").dumps(event, ensure_ascii=False) + "\n")
            self._event_fp.flush()

    def close(self):
        if self._event_fp:
            self._event_fp.close()
            self._event_fp = None
