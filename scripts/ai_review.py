# -*- coding: utf-8 -*-
"""AI 快审蒙太奇: 把最近诊断帧拼成 4x3 网格 + 元信息, 供 DeepSeek 视觉一次性审检。
输出 temp/self_check/montage.png (供AI读取) + review_input.json (检测/血量读数)。"""
import json
from pathlib import Path

import cv2
import numpy as np

OUT = Path(__file__).resolve().parents[1] / "temp" / "self_check"


def build(max_frames=12):
    pngs = sorted(OUT.glob("live_*.png"))[-max_frames:]
    if not pngs:
        print("no frames")
        return
    tpl_w, tpl_h = 640, 360
    cols = 3
    rows = (len(pngs) + cols - 1) // cols
    canvas = np.zeros((rows * tpl_h, cols * tpl_w, 3), np.uint8)
    metas = []
    for i, p in enumerate(pngs):
        img = cv2.imread(str(p))
        if img is None:
            continue
        img = cv2.resize(img, (tpl_w, tpl_h))
        r, c = i // cols, i % cols
        canvas[r * tpl_h:(r + 1) * tpl_h, c * tpl_w:(c + 1) * tpl_w] = img
        m = Path(str(p).replace("live_", "meta_").replace(".png", ".json"))
        if m.exists():
            try:
                d = json.loads(m.read_text(encoding="utf-8"))
                metas.append({"t": p.stem.replace("live_", ""),
                              "ui": d.get("ui"),
                              "dets": [f"{x['cls']}:{x['conf']}" for x in d.get("dets", [])]})
            except Exception:
                metas.append({"t": p.stem.replace("live_", "")})
    canvas = cv2.resize(canvas, (1280, rows * 240))
    cv2.imwrite(str(OUT / "montage.png"), canvas)
    (OUT / "review_input.json").write_text(
        json.dumps(metas, ensure_ascii=False), encoding="utf-8")
    print(f"montage: {len(pngs)} 帧 -> temp/self_check/montage.png")


if __name__ == "__main__":
    build()
