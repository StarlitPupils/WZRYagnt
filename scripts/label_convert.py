# -*- coding: utf-8 -*-
"""标注表述脚本 v1: 把人工标注(图片画框 或 坐标文本) 转成 YOLO 训练标签。

用法一(图片画框): 把画好框的图放 temp/label_me/annotated/<原名>.png
  读取: temp/label_me/<原名>.png 为原图, 标注文件按下面约定:
    仅需在标注图上画出矩形, 无需写文本 —— 但为保证类别正确, 请在文件名或
    附加 describe.txt 里写明: 每行 "<图片名> <类别> <x1> <y1> <x2> <y2>"
    类别: monster / ally / enemy / self
用法二(坐标文本, 推荐): 直接编辑 temp/label_me/describe.txt 每行:
    <图片名不带扩展名> <类别> <x1> <y1> <x2> <y2>   (像素坐标 1280x720)
    e.g.  m1 monster 420 470 520 570
          m1 ally 700 300 800 400
          m1 enemy 900 280 1010 400
          m1 self 560 250 680 400
输出:
    temp/label_me/yolo/<图片名>.txt   (YOLO 归一化标签, 类别映射先按约定)
    temp/label_me/rendered/<图片名>.png (校验渲染图)
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import cv2  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DIR = ROOT / "temp" / "label_me"
CLS_MAP = {"monster": 8, "ally": 1, "enemy": 0, "self": 1}   # 与 11 类训练集一致
COLORS = {"monster": (0, 255, 255), "ally": (255, 128, 0),
          "enemy": (0, 0, 255), "self": (255, 0, 255)}


def main():
    desc = DIR / "describe.txt"
    if not desc.exists():
        print("未找到 temp/label_me/describe.txt —— 请按脚本头注释格式填写")
        return
    out_y = DIR / "yolo"; out_r = DIR / "rendered"
    out_y.mkdir(exist_ok=True); out_r.mkdir(exist_ok=True)
    n = 0
    for ln in desc.read_text(encoding="utf-8").splitlines():
        ln = ln.strip()
        if not ln or ln.startswith("#"):
            continue
        parts = ln.split()
        if len(parts) < 6:
            print("跳过(格式错):", ln)
            continue
        name, cls, x1, y1, x2, y2 = (parts[0], parts[1],
                                    *[int(v) for v in parts[2:6]])
        img = cv2.imdecode(np.fromfile(str(DIR / f"{name}.png"), dtype=np.uint8),
                           cv2.IMREAD_COLOR)
        if img is None:
            print("缺图:", name)
            continue
        h, w = img.shape[:2]
        cx, cy = (x1 + x2) / 2 / w, (y1 + y2) / 2 / h
        bw, bh = (x2 - x1) / w, (y2 - y1) / h
        ycls = CLS_MAP.get(cls)
        if ycls is None:
            print("未知类别:", cls)
            continue
        with open(out_y / f"{name}.txt", "a", encoding="utf-8") as f:
            f.write(f"{ycls} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")
        cv2.rectangle(img, (x1, y1), (x2, y2), COLORS[cls], 3)
        cv2.putText(img, cls, (x1, max(16, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, COLORS[cls], 2)
        cv2.imwrite(str(out_r / f"{name}.png"), img)
        n += 1
    print(f"完成: {n} 条标注 -> temp/label_me/yolo/*.txt (每图一行一条)")


if __name__ == "__main__":
    main()
