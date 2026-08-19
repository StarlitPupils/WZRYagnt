# -*- coding: utf-8 -*-
"""理解层全景分析：对一帧运行完整感知管线，输出 agent 实际"看到"的全部信息。

用于向用户展示理解层能力与盲区。运行：
    venv\\Scripts\\python.exe scripts\\understanding_audit.py <frame.png>
"""
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def audit(path, model=None):
    img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        print("无法读取", path)
        return
    h, w = img.shape[:2]
    out = [f"===== {path} ({w}x{h}) ====="]

    # 1. YOLO 检测
    if model is not None:
        dets = model.detect(img)
        out.append(f"[YOLO] {len(dets)} 个目标:")
        for d in dets[:10]:
            x1, y1, x2, y2 = (int(v) for v in d.xyxy)
            out.append(f"  {d.cls:16s} conf={d.conf:.2f} 框=({x1},{y1})-({x2},{y2})")
    else:
        out.append("[YOLO] 未加载模型")

    # 2. 小地图（生产路径：MinimapTracker 方形框，与主循环一致）
    try:
        from wzry.vision.minimap_tracker import MinimapTracker
        from wzry.vision.terrain import DEFAULT_BOX
        from wzry.calib import load_calibration
        calib, _ = load_calibration()
        h, w = img.shape[:2]
        mc = calib.get("minimap_center", [0.086, 0.129])
        tracker = MinimapTracker(prior_center=(int(mc[0] * w), int(mc[1] * h)),
                                 box_prior=DEFAULT_BOX)
        mm = tracker.update(img)
        if mm.get("found"):
            dots = mm.get("dots", {})
            out.append(f"[小地图] 找到: 绿(自己)={len(dots.get('green', []))} "
                       f"蓝(队友)={len(dots.get('blue', []))} "
                       f"红(敌)={len(dots.get('red', []))} "
                       f"黄(野怪)={len(dots.get('yellow', []))}")
            for k in ("green", "blue", "red"):
                pts = dots.get(k, [])
                if pts:
                    out.append(f"  {k} 位置: {[(round(p[0],2), round(p[1],2)) for p in pts[:6]]}")
            tw = mm.get("towers", [])
            if tw:
                out.append(f"  塔: {[(t[0], t[1], t[2]) for t in tw[:8]]}")
        else:
            out.append("[小地图] 未找到")
    except Exception as e:
        out.append(f"[小地图] 异常: {e}")

    # 3. 自己血条 HP/MP + 英雄位置
    try:
        from wzry.vision.self_bars import self_hp_mp, detect_all_bars
        hp, mp, pos = self_hp_mp(img)
        out.append(f"[自己] HP={hp} MP={mp} 英雄位置={pos}")
        bars = detect_all_bars(img)
        out.append(f"[血条] 自己绿条={len(bars['self'])} 队友蓝条={len(bars['allies'])} "
                   f"敌人红条={len(bars['enemies'])}")
        if bars["enemies"]:
            out.append(f"  敌人红条: {[(b['x'], b['y'], b['w']) for b in bars['enemies'][:5]]}")
        if bars["allies"]:
            out.append(f"  队友蓝条: {[(b['x'], b['y'], b['w']) for b in bars['allies'][:5]]}")
    except Exception as e:
        out.append(f"[血条] 异常: {e}")

    # 4. 顶部队友头像条
    try:
        from wzry.vision.teammate_bar import detect_teammates
        tb = detect_teammates(img)
        out.append(f"[队友条] 头像={len(tb)} 位置={[(a[0], a[1]) for a in tb[:5]]}")
    except Exception as e:
        out.append(f"[队友条] 异常: {e}")

    # 5. 技能状态（校准坐标）
    try:
        import json
        pts = json.loads((ROOT / "configs" / "calibration_absolute.json").read_text(encoding="utf-8"))["points"]
        from wzry.vision.ui_reader import skill_ready_state
        ss = skill_ready_state(img, {1: pts["skill1"], 2: pts["skill2"], 3: pts["skill3"]})
        out.append(f"[技能] {ss}")
    except Exception as e:
        out.append(f"[技能] 异常: {e}")

    print("\n".join(out))


if __name__ == "__main__":
    p = sys.argv[1] if len(sys.argv) > 1 else "temp/v3_120s.png"
    model = None
    if "--yolo" in sys.argv:
        from wzry.vision.detector import YoloDetector
        model = YoloDetector(str(ROOT / "runs" / "detect" / "zhongkui_11cls"
                                 / "weights" / "best.pt"), conf=0.35)
    audit(p, model)
