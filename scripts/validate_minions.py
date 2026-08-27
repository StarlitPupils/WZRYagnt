# -*- coding: utf-8 -*-
"""小兵点模板匹配 留一法验证（当前帧的模板从训练集剔除）。"""
import sys
from pathlib import Path
import numpy as np
import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from wzry.vision.mm_rules_v7 import MMDetectorV7, _get_dot_templates

TR = ROOT / "temp" / "mm_dot_templates"
# 每帧模板索引映射: (side, frame) -> 模板文件索引
tpl_of_frame = {"ally": {}, "enemy": {}}
for side in ("ally", "enemy"):
    for j, f in enumerate(sorted(TR.glob(f"{side}_*.png"))):
        # 文件名 ally_003.png 无帧信息 => 重新生成带帧的模板集
        pass

# 直接用"每帧生成模板文件"方式: 重新从真值生成带帧标记的模板
tpl_by_frame = {"ally": {}, "enemy": {}}
idx = {}
for i in range(1, 13):
    fp = ROOT / "temp" / "ann" / f"s{i:02d}.png"
    gtf = fp.with_suffix(".txt")
    if not fp.exists():
        continue
    img = cv2.imdecode(np.fromfile(str(fp), dtype=np.uint8), cv2.IMREAD_COLOR)
    mm = img[:720, :1280][0:232, 0:232]
    hsv = cv2.cvtColor(mm, cv2.COLOR_BGR2HSV)
    H, S, V = hsv[..., 0].astype(int), hsv[..., 1].astype(int), hsv[..., 2].astype(int)
    for ln in gtf.read_text(encoding="utf-8").splitlines():
        p = ln.split()
        if len(p) != 5:
            continue
        c = int(p[0])
        if c not in (18, 19):
            continue
        px, py = int(float(p[1]) * 1280), int(float(p[2]) * 720)
        if not (10 <= px <= 222 and 10 <= py <= 222):
            continue
        t = mm[py - 9:py + 10, px - 9:px + 10]
        if t.shape[:2] != (19, 19):
            continue
        key = "ally" if c == 18 else "enemy"
        sig = (H >= 80) & (H <= 150) & (S > 60) if key == "ally" else ((H <= 15) | (H >= 160)) & (S > 60)
        if int(sig[py - 3:py + 4, px - 3:px + 4].sum()) >= 8:
            tpl_by_frame[key].setdefault(i, []).append(t)

total = {18: [0, 0], 19: [0, 0]}
for i in range(1, 13):
    fp = ROOT / "temp" / "ann" / f"s{i:02d}.png"
    gtf = fp.with_suffix(".txt")
    if not fp.exists():
        continue
    img = cv2.imdecode(np.fromfile(str(fp), dtype=np.uint8), cv2.IMREAD_COLOR)
    # 留一模板: 其他帧的模板
    loo = {"ally": [], "enemy": []}
    for key in ("ally", "enemy"):
        for fi, arr in tpl_by_frame[key].items():
            if fi != i:
                loo[key].extend(arr)
    # 手动匹配 (与 mm_rules_v7 相同: 暗环带 V<=128)
    mm = img[:720, :1280][0:232, 0:232]
    hsvv = cv2.cvtColor(mm, cv2.COLOR_BGR2HSV)
    Vv = hsvv[..., 2].astype(int)
    yy, xx = np.mgrid[-14:15, -14:15]
    dd = np.sqrt(xx ** 2 + yy ** 2)
    rsel = (dd >= 6) & (dd <= 12)
    pred = {18: [], 19: []}
    for key, cid in (("ally", 18), ("enemy", 19)):
        cands = []
        for t in loo[key]:
            r = cv2.matchTemplate(mm, t, cv2.TM_CCOEFF_NORMED)
            ys, xs = np.nonzero(r > 0.62)
            for v, x, y in zip(r[ys, xs].tolist(), xs.tolist(), ys.tolist()):
                cands.append((float(v), int(x) + 9, int(y) + 9))
        cands.sort(reverse=True)
        for v, px, py in cands:
            if not (8 <= px <= 224 and 8 <= py <= 224):
                continue
            vwin = Vv[py - 14:py + 15, px - 14:px + 15]
            if vwin.shape != (29, 29) or float(vwin[rsel].mean()) > 128:
                continue
            # 点内质量: 5x5 青/红像素数
            Hw = hsvv[..., 0].astype(int)[py - 2:py + 3, px - 2:px + 3]
            Sw = hsvv[..., 1].astype(int)[py - 2:py + 3, px - 2:px + 3]
            if key == "ally":
                mass = int(((Hw >= 80) & (Hw <= 150) & (Sw > 60)).sum())
            else:
                mass = int((((Hw <= 15) | (Hw >= 160)) & (Sw > 60)).sum())
            if mass < 10:
                continue
            if all((px - a) ** 2 + (py - b) ** 2 > 36 for a, b in pred[cid]):
                pred[cid].append((px, py))
    for ln in gtf.read_text(encoding="utf-8").splitlines():
        p = ln.split()
        if len(p) != 5:
            continue
        c = int(p[0])
        if c not in (18, 19):
            continue
        gx, gy = float(p[1]) * 1280, float(p[2]) * 720
        total[c][0] += 1
        ok = any((px - gx) ** 2 + (py - gy) ** 2 <= 12 ** 2 for px, py in pred[c])
        total[c][1] += int(ok)
    print(f"s{i:02d} 我兵: GT{sum(1 for ln in gtf.read_text(encoding='utf-8').splitlines() if len(ln.split())==5 and int(ln.split()[0])==18)} 检{len(pred[18])} | "
          f"敌兵: GT{sum(1 for ln in gtf.read_text(encoding='utf-8').splitlines() if len(ln.split())==5 and int(ln.split()[0])==19)} 检{len(pred[19])}")
print()
for c, name in ((18, "我兵"), (19, "敌兵")):
    print(f"{name} 留一法: {total[c][1]}/{total[c][0]} = {total[c][1]/max(1,total[c][0])*100:.0f}%")
