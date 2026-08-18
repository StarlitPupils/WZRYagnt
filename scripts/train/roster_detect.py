# -*- coding: utf-8 -*-
"""开局阵容识别 v3（理解层）：选英雄界面 -> 双方英雄名（modlens 视觉识别）。

用户语义：
  - 选英雄界面文字含双方英雄名（上/下两侧）
  - 含"钟馗"（用户自己的英雄）的一侧 = 我方
  - 用 modlens（glm-4.6v）识别英雄名，JSON 输出

流程：
  1. 输入选英雄界面截图（或实时截屏）
  2. modlens 识别 upper/lower 英雄名
  3. 含钟馗侧 = 我方，另一侧 = 敌方
  4. 保存 data/roster.json

用法：
    venv\\Scripts\\python.exe scripts\\train\\roster_detect.py --image xxx.png
    venv\\Scripts\\python.exe scripts\\train\\roster_detect.py --serial 127.0.0.1:16384
"""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

SELF_HERO = "钟馗"
PROMPT = ('这是王者荣耀选英雄界面。请列出上方区域(y0-360)的所有英雄名和'
          '下方区域(y360-720)的所有英雄名，用JSON格式输出：'
          '{"upper": ["英雄名"...], "lower": ["英雄名"...]}。'
          '只输出英雄名（去掉皮肤名），格式必须严格是JSON。')


def modlens_ask(image_path):
    """调用 modlens_ask.py 子进程识别。"""
    r = subprocess.run(
        [sys.executable, "-X", "utf8",
         str(ROOT / "scripts" / "train" / "modlens_ask.py"),
         str(image_path), PROMPT],
        capture_output=True, text=True, encoding="utf-8", errors="replace",
        timeout=180)
    if r.returncode != 0:
        raise RuntimeError(f"modlens 调用失败: {r.stderr[:200]}")
    return r.stdout.strip()


def parse_roster(resp_text):
    """解析 modlens JSON 响应，含钟馗侧为我方。"""
    # 提取 JSON（响应可能含额外文字）
    start = resp_text.find("{")
    end = resp_text.rfind("}")
    if start < 0 or end < 0:
        raise ValueError(f"无法解析响应: {resp_text[:200]}")
    data = json.loads(resp_text[start:end + 1])
    upper = data.get("upper", [])
    lower = data.get("lower", [])
    if SELF_HERO in upper:
        ally, enemy = upper, lower
    elif SELF_HERO in lower:
        ally, enemy = lower, upper
    else:
        # 无钟馗：按上下猜测（上方=我方通常，可配置）
        print(f"[roster] 未找到{SELF_HERO}，默认上方为我方")
        ally, enemy = upper, lower
    return {"ally": ally, "enemy": enemy, "self_hero": SELF_HERO}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", default=None, help="选英雄界面截图（缺省实时截屏）")
    ap.add_argument("--serial", default="127.0.0.1:16384")
    ap.add_argument("--save", action="store_true", help="保存 roster.json")
    args = ap.parse_args()

    if args.image:
        frame = cv2.imread(args.image)
    else:
        subprocess.run(["adb", "connect", args.serial], capture_output=True)
        out = subprocess.run(["adb", "-s", args.serial, "exec-out",
                              "screencap", "-p"], capture_output=True)
        import numpy as np
        frame = cv2.imdecode(np.frombuffer(out.stdout, np.uint8), cv2.IMREAD_COLOR)
    if frame is None:
        print("无法获取画面")
        return 1
    print(f"画面 {frame.shape[1]}x{frame.shape[0]}")

    tmp = ROOT / "temp" / "roster_frame.png"
    cv2.imwrite(str(tmp), frame)
    print("调用视觉模型识别阵容（约 10-30 秒）...")
    t0 = time.time()
    try:
        resp = modlens_ask(tmp)
        roster = parse_roster(resp)
    except Exception as e:
        print(f"识别失败: {e}")
        return 1
    print(f"识别耗时 {time.time()-t0:.0f}s")
    print(f"=== 阵容 ===")
    print(f"我方({len(roster['ally'])}): {roster['ally']}")
    print(f"敌方({len(roster['enemy'])}): {roster['enemy']}")
    if args.save:
        roster["ts"] = time.time()
        out = ROOT / "data" / "roster.json"
        out.write_text(json.dumps(roster, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"已保存 {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
