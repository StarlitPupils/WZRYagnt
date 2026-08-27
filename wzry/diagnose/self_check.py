# -*- coding: utf-8 -*-
"""AI 自我诊断 v2 (DeepSeek-视觉审检模式): 每 N 秒截帧留存 -> 由 DeepSeek 视觉(本助手)读帧诊断。

- 运行中持续留存 temp/self_check/live_*.png (最近 40 张环形)
- 帧元信息(检测结果快照)随存 temp/self_check/meta_<ts>.json
- DeepSeek 视觉审检后: 野怪误检样本 -> temp/self_check/mislabeled/ (帧+label)
- 不调用第三方 API (DeepSeek 官方无视觉接口; 视觉由 DeepSeek 模型本助手承担)
"""
import json
import threading
import time
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "temp" / "self_check"
MAX_KEEP = 40


class SelfCheckDiagnostician:
    def __init__(self, interval_s=20.0):
        self.interval = interval_s
        self.last_frame = None
        self.last_dets = []
        self.lock = threading.Lock()
        self._stop = False
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.saved = 0

    def start(self):
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        self.thread.start()

    def stop(self):
        self._stop = True

    def feed(self, frame, dets):
        with self.lock:
            self.last_frame = frame
            self.last_dets = list(dets)

    def _loop(self):
        while not self._stop:
            time.sleep(self.interval)
            try:
                with self.lock:
                    fr = self.last_frame
                    dets = self.last_dets
                if fr is None:
                    continue
                ts = time.strftime("%Y%m%d_%H%M%S")
                img_p = OUT_DIR / f"live_{ts}.png"
                cv2.imwrite(str(img_p), fr)
                meta = [{
                    "cls": d.cls,
                    "conf": round(float(d.conf), 2),
                    "cx": float(d.center[0]),
                    "cy": float(d.center[1]),
                    "box": [round(float(v), 1) for v in d.xyxy],
                } for d in dets]
                (OUT_DIR / f"meta_{ts}.json").write_text(
                    json.dumps({"t": ts, "dets": meta}, ensure_ascii=False),
                    encoding="utf-8")
                self.saved += 1
                # 环形清理: 只留最近 MAX_KEEP 张
                for p in sorted(OUT_DIR.glob("live_*.png"))[:-MAX_KEEP]:
                    p.unlink(missing_ok=True)
                for p in sorted(OUT_DIR.glob("meta_*.json"))[:-MAX_KEEP]:
                    p.unlink(missing_ok=True)
            except Exception:
                pass
