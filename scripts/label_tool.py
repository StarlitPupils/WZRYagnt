# -*- coding: utf-8 -*-
"""标注工具 v3 (tkinter 原生, Windows 鼠标事件可靠)。

用法: python scripts/label_tool.py
操作: 左键拖框; 数字1-4选类别; W保存下一张; S跳过; Z撤销; Q退出
输出: temp/label_me/yolo/<图名>.txt + annotated/<图名>.png
"""
import argparse
import sys
import tkinter as tk
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CLS_KEYS = {"1": ("野怪", 8, "#00ffff"),
            "2": ("我方英雄", 1, "#ff8000"),
            "3": ("敌方英雄", 0, "#ff0000"),
            "4": ("自己", 1, "#ff00ff")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="temp/label_me")
    args = ap.parse_args()
    img_dir = ROOT / args.dir
    files = sorted(p for p in img_dir.glob("*.png")
                   if p.stem[:1] in ("g", "s") and len(p.stem) == 2)
    if not files:
        files = sorted(img_dir.glob("*.png"))
    yolo_dir = img_dir / "yolo"
    ann_dir = img_dir / "annotated"
    yolo_dir.mkdir(exist_ok=True)
    ann_dir.mkdir(exist_ok=True)

    import cv2
    from PIL import Image, ImageTk

    root = tk.Tk()
    root.title("标注工具")
    idx = {"i": 0}
    cur = {"k": "1"}
    boxes = []          # (cls, name, x1,y1,x2,y2) 画布坐标(缩放后)
    state = {"down": None, "tmp": None, "scale": 1.0}
    canvas = tk.Canvas(root, width=1280, height=720, bg="black")
    canvas.pack(fill="both", expand=True)
    info = tk.Label(root, text="", fg="white", bg="#222", anchor="w", font=("微软雅黑", 12))
    info.pack(fill="x")

    def cur_color():
        return CLS_KEYS[cur["k"]][2]

    def redraw():
        canvas.delete("all")
        img = _current_img()
        if img is None:
            return
        canvas.im = ImageTk.PhotoImage(Image.fromarray(img))
        canvas.create_image(0, 0, anchor="nw", image=canvas.im)
        for (cl, nm, x1, y1, x2, y2) in boxes:
            canvas.create_rectangle(x1, y1, x2, y2, outline=cur_color_of(nm), width=2)
            canvas.create_text(x1 + 4, max(14, y1 - 10), text=nm, fill=cur_color_of(nm),
                               anchor="w", font=("微软雅黑", 11, "bold"))
        if state["tmp"]:
            x1, y1, x2, y2 = state["tmp"]
            canvas.create_rectangle(x1, y1, x2, y2, outline="#aaffaa", width=1)
        info.config(text=f"类别[{CLS_KEYS[cur['k']][0]}]  已框 {len(boxes)}  |  "
                         f"1野怪 2我方 3敌方 4自己 | W保存/下一张  S跳过  Z撤销  Q退出")

    def cur_color_of(nm):
        for k, (n, c, col) in CLS_KEYS.items():
            if n == nm:
                return col
        return "#ffffff"

    def _current_img():
        i = idx["i"]
        if 0 <= i < len(files):
            img = cv2.imread(str(files[i])) if False else None
            import cv2 as _cv2
            img = _cv2.imdecode(np.fromfile(str(files[i]), dtype=np.uint8), _cv2.IMREAD_COLOR)
            return img[:, :, ::-1]  # BGR->RGB
        return None

    def on_press(e):
        state["down"] = (e.x, e.y)

    def on_drag(e):
        if state["down"]:
            state["tmp"] = (*state["down"], e.x, e.y)
            redraw()

    def on_release(e):
        if state["down"]:
            x1, y1 = state["down"]
            x2, y2 = e.x, e.y
            state["down"] = None
            state["tmp"] = None
            if abs(x2 - x1) > 12 and abs(y2 - y1) > 12:
                nm, cl, col = CLS_KEYS[cur["k"]]
                boxes.append((cl, nm, min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)))
                print(f"[框] {nm} ({min(x1,x2)},{min(y1,y2)})-({max(x1,x2)},{max(y1,y2)}) 共{len(boxes)}")
            redraw()

    canvas.bind("<ButtonPress-1>", on_press)
    canvas.bind("<B1-Motion>", on_drag)
    canvas.bind("<ButtonRelease-1>", on_release)

    def save(skip=False):
        i = idx["i"]
        if 0 <= i < len(files):
            img = _current_img()
            h, w = img.shape[:2]
            yf = yolo_dir / (files[i].stem + ".txt")
            with open(yf, "w", encoding="utf-8") as f:
                for (cl, nm, x1, y1, x2, y2) in boxes:
                    f.write(f"{cl} {((x1+x2)/2)/w:.6f} {((y1+y2)/2)/h:.6f} "
                            f"{abs(x2-x1)/w:.6f} {abs(y2-y1)/h:.6f}\n")
            # 校验图(BGR)
            import cv2 as _cv2
            bgr = img[:, :, ::-1]
            for (cl, nm, x1, y1, x2, y2) in boxes:
                from PIL import ImageDraw
                pil = Image.fromarray(bgr[:, :, ::-1])
                dr = ImageDraw.Draw(pil)
                dr.rectangle([x1, y1, x2, y2], outline=cur_color_of(nm), width=2)
                dr.text((x1 + 3, max(2, y1 - 16)), nm, fill=cur_color_of(nm))
            _cv2.imwrite(str(ann_dir / (files[i].stem + ".png")),
                         np.array(Image.fromarray(
                             np.asarray(Image.fromarray(bgr), dtype=np.uint8))))
            print("已存:", files[i].stem, f"{len(boxes)}框")
        if not skip:
            idx["i"] += 1
            load_boxes()
            redraw()

    def load_boxes():
        boxes.clear()
        i = idx["i"]
        if 0 <= i < len(files):
            yf = yolo_dir / (files[i].stem + ".txt")
            img = _current_img()
            if yf.exists() and img is not None:
                h, w = img.shape[:2]
                for l in yf.read_text(encoding="utf-8").splitlines():
                    p = l.split()
                    if len(p) == 5:
                        cl, cx, cy, bw, bh = float(p[0]), *[float(v) for v in p[1:]]
                        nm = next((n for k, (n, c, col) in CLS_KEYS.items() if c == int(cl)), "?")
                        boxes.append((int(cl), nm, int((cx - bw / 2) * w),
                                      int((cy - bh / 2) * h), int((cx + bw / 2) * w),
                                      int((cy + bh / 2) * h)))

    def key_handler(event):
        k = event.keysym.lower()
        if k in ("1", "2", "3", "4"):
            cur["k"] = k
        elif k == "w":
            save(False)
        elif k == "s":
            save(True)
        elif k in ("d", "right"):
            idx["i"] = min(len(files) - 1, idx["i"] + 1)
            load_boxes()
        elif k in ("a", "left"):
            idx["i"] = max(0, idx["i"] - 1)
            load_boxes()
        elif k == "z" and boxes:
            boxes.pop()
        elif k == "q":
            root.destroy()
            return
        redraw()

    root.bind("<Key>", key_handler)
    load_boxes()
    redraw()
    print(f"素材 {len(files)} 张: {[f.name for f in files]}")
    root.mainloop()
    print("完成")


if __name__ == "__main__":
    main()
