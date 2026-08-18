# -*- coding: utf-8 -*-
"""理解层集成测试：单帧 -> 完整结构化局势（供决策层/调试）。

用法：
    venv\\Scripts\\python.exe scripts\\train\\understanding_test.py [--image xxx.png]
    （缺省用示范局 s01 帧）
"""
import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from wzry.vision.understanding import Understanding  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", default=str(ROOT / "temp" / "ann" / "s01.png"))
    ap.add_argument("--recognize", action="store_true", help="额外做英雄识别(慢)")
    args = ap.parse_args()

    img = cv2.imread(args.image)
    if img is None:
        print(f"无法读取 {args.image}")
        return 1

    u = Understanding()
    st = u.update(img, recognize=args.recognize)

    print("=" * 50)
    print(f"理解层输出: {args.image}")
    print("=" * 50)

    mm = st["minimap"]
    print(f"\n[小地图] 中心{mm['center']} 半径{mm['radius']}")
    print(f"  自己(绿圈): {len(mm['self'])} 个 {mm['self']}")
    print(f"  队友(蓝圈): {len(mm['allies'])} 个 {mm['allies']}")
    print(f"  敌人(红圈): {len(mm['enemies'])} 个 {mm['enemies']}")
    print(f"  野怪(黄点): {len(mm['monsters'])} 个")
    print(f"  我方小兵: {len(mm['ally_minions'])} | 敌方小兵: {len(mm['enemy_minions'])}")
    print(f"  塔: {len(mm['towers'])} 个 {mm['towers']}")

    selfs = st["self"]
    print(f"\n[自己] HP={selfs['hp']} MP={selfs['mp']}")
    for k, v in (selfs.get("skills") or {}).items():
        print(f"  技能{k}: 解锁={v['unlocked']} 就绪={v['ready']} 亮度={v['mean_v']:.0f}")

    if args.recognize and st.get("heroes"):
        h = st["heroes"]
        print(f"\n[英雄识别] 自己={h['self']}")
        print(f"  队友: {h['allies'][:3]}")
        print(f"  敌人: {h['enemies'][:3]}")

    # roster（如已存在）
    roster_file = ROOT / "data" / "roster.json"
    if roster_file.exists():
        roster = json.loads(roster_file.read_text(encoding="utf-8"))
        print(f"\n[阵容] 我方 {roster.get('ally')} | 敌方 {roster.get('enemy')}")

    print("\n" + "=" * 50)
    return 0


if __name__ == "__main__":
    sys.exit(main())
