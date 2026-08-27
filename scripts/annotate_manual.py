# -*- coding: utf-8 -*-
"""手动标注工具：框选全屏单位 + 小地图元素，输出 YOLO 格式标注。

用法：
  venv\\Scripts\\python.exe scripts\\annotate_manual.py [图片路径...]
  不传参数则标注 temp/ann/s*.png

操作：
  m            切换 全屏模式 <-> 小地图模式（小地图放大 3 倍）
  数字键 1-9,0 选择当前类别（类别名显示在标题栏）
  鼠标左键拖拽 画一个框
  d            删除最后一个框
  n / p        下一张 / 上一张（自动保存当前）
  s            手动保存
  ESC          退出

类别：
  全屏模式:  1敌方英雄 2我方英雄 7野怪 8自己
  (其余类别已精简; 小地图模式不适用本批素材)
  小地图模式: 1自己 2队友 3敌人 4蓝方塔 5红方塔 6野怪点 7buff
             8我方小兵 9敌方小兵
输出：
  每张图: temp/annot_manual/<name>.png 同目录 <name>.txt (YOLO格式)
  类别0-10=全屏, 11-19=小地图(坐标已转全屏)
"""
import sys
from pathlib import Path

import cv2
import numpy as np

# ---- 中文渲染（OpenCV putText 不支持中文 -> 用 PIL + 微软雅黑）----
from PIL import Image, ImageDraw, ImageFont  # noqa: E402

_FONT_CACHE = {}


def _font(size):
    if size not in _FONT_CACHE:
        for fp in (r"C:\Windows\Fonts\msyh.ttc", r"C:\Windows\Fonts\msyhbd.ttc",
                   r"C:\Windows\Fonts\simhei.ttf", r"C:\Windows\Fonts\simsun.ttc"):
            if Path(fp).exists():
                _FONT_CACHE[size] = ImageFont.truetype(fp, size)
                break
        else:
            _FONT_CACHE[size] = None
    return _FONT_CACHE[size]


def put_text_cn(img, text, org, size=16, color=(255, 255, 0)):
    """在 BGR 图像上绘制中文（PIL 渲染后写回）。"""
    f = _font(size)
    if f is None:
        return
    x, y = int(org[0]), int(org[1])
    h, w = img.shape[:2]
    if x >= w or y >= h:
        return
    pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    dr = ImageDraw.Draw(pil)
    dr.text((x, y), text, font=f, fill=(color[2], color[1], color[0]))
    img[:] = cv2.cvtColor(np.asarray(pil), cv2.COLOR_RGB2BGR)

ROOT = Path(__file__).resolve().parent.parent
MM = (0, 0, 232, 232)          # 小地图区域（全屏坐标）
MM_ZOOM = 3

# v2 仅四类(用户本次标注): 1敌方英雄 2我方英雄 7野怪 8自己(新类11)
FULL_CLASSES = {
    ord("1"): (0, "敌方英雄"),
    ord("2"): (1, "我方英雄"),
    ord("7"): (8, "野怪"),
    ord("8"): (11, "自己"),
}
MM_CLASSES = {
    ord("1"): (11, "自己"), ord("2"): (12, "队友"), ord("3"): (13, "敌人"),
    ord("4"): (14, "蓝方塔"), ord("5"): (15, "红方塔"), ord("6"): (16, "野怪点"),
    ord("7"): (17, "buff"), ord("8"): (18, "我方小兵"), ord("9"): (19, "敌方小兵"),
}
COLORS = {
    0: (0, 0, 255), 1: (255, 128, 0), 2: (0, 100, 255), 3: (100, 200, 0),
    4: (0, 0, 200), 5: (200, 100, 0), 6: (255, 0, 200), 7: (200, 255, 0),
    8: (0, 255, 255), 9: (255, 0, 255), 10: (255, 255, 0),
    11: (0, 255, 0), 12: (255, 128, 0), 13: (0, 0, 255),
    14: (200, 100, 0), 15: (0, 100, 200), 16: (0, 255, 255),
    17: (255, 0, 255), 18: (150, 255, 150), 19: (150, 150, 255),
}
LABELS = {0: "敌方英雄", 1: "我方英雄", 2: "敌方小兵", 3: "我方小兵",
          4: "敌方塔", 5: "我方塔", 6: "敌方水晶", 7: "我方水晶",
          8: "野怪", 9: "钩子", 10: "技能特效", 11: "自己",
          16: "野怪点", 17: "buff", 18: "我方小兵", 19: "敌方小兵"}
COLORS[11] = (255, 0, 255)   # 自己=紫


class ManualAnnotator:
    def __init__(self, paths):
        self.paths = list(paths)
        self.idx = 0
        self.mm_mode = False
        self.cls = 0
        self.boxes = []          # [(cls, x1, y1, x2, y2)] 全屏坐标
        self.drag = None
        self.out_dir = ROOT / "temp" / "annot_manual"
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.img = None
        self.vis = None

    # ---------- 加载/保存 ----------
    def load(self):
        p = self.paths[self.idx]
        data = np.fromfile(str(p), dtype=np.uint8)
        self.img = cv2.imdecode(data, cv2.IMREAD_COLOR)
        self.boxes = []
        txt = p.with_suffix(".txt")
        if txt.exists():
            for ln in txt.read_text(encoding="utf-8").splitlines():
                parts = ln.split()
                if len(parts) != 5:
                    continue
                c, cx, cy, bw, bh = map(float, parts)
                h, w = self.img.shape[:2]
                self.boxes.append((int(c), int((cx - bw / 2) * w),
                                   int((cy - bh / 2) * h),
                                   int((cx + bw / 2) * w),
                                   int((cy + bh / 2) * h)))
        print(f"[{self.idx + 1}/{len(self.paths)}] {p.name} 已加载框 {len(self.boxes)}")

    def save(self):
        if self.img is None:
            return
        p = self.paths[self.idx]
        h, w = self.img.shape[:2]
        lines = []
        for c, x1, y1, x2, y2 in self.boxes:
            cx = ((x1 + x2) / 2) / w
            cy = ((y1 + y2) / 2) / h
            bw = (x2 - x1) / w
            bh = (y2 - y1) / h
            lines.append(f"{c} {cx:.4f} {cy:.4f} {bw:.4f} {bh:.4f}")
        (p.with_suffix(".txt")).write_text("\n".join(lines), encoding="utf-8")
        print(f"已保存 {p.name}: {len(lines)} 框")

    # ---------- 渲染 ----------
    def render(self):
        if self.img is None:
            return
        if not self.mm_mode:
            vis = self.img.copy()
            scale = 1.0
            ox = oy = 0
            h, w = self.img.shape[:2]
        else:
            x0, y0, x1, y1 = MM
            mm = self.img[y0:y1, x0:x1]
            mm = cv2.resize(mm, ((x1 - x0) * MM_ZOOM, (y1 - y0) * MM_ZOOM),
                            interpolation=cv2.INTER_NEAREST)
            vis = mm.copy()
            scale = MM_ZOOM
            ox, oy = -x0 * MM_ZOOM, -y0 * MM_ZOOM
            h, w = vis.shape[:2]
        # 已存框
        for c, bx1, by1, bx2, by2 in self.boxes:
            sx1, sy1 = int(bx1 * scale + ox), int(by1 * scale + oy)
            sx2, sy2 = int(bx2 * scale + ox), int(by2 * scale + oy)
            cv2.rectangle(vis, (sx1, sy1), (sx2, sy2), COLORS.get(c, (200, 200, 200)), 2)
            put_text_cn(vis, LABELS.get(c, str(c)), (sx1, max(6, sy1 - 22)),
                        14, COLORS.get(c, (200, 200, 200)))
        # 拖动中的框
        if self.drag:
            (dx1, dy1), (dx2, dy2) = self.drag
            cv2.rectangle(vis, (dx1, dy1), (dx2, dy2),
                          COLORS.get(self.cls, (200, 200, 200)), 2)
        mode = "小地图(放大x3)" if self.mm_mode else "全屏"
        put_text_cn(vis, f"{mode} | 类别: {LABELS.get(self.cls, self.cls)}",
                    (10, 30), 20, (0, 255, 255))
        put_text_cn(vis, f"{self.paths[self.idx].name} | 框数: {len(self.boxes)}",
                    (10, 60), 16, (255, 255, 255))
        cv2.imshow("manual-annotator", vis)
        self.vis = vis

    def to_full(self, sx, sy):
        """屏幕坐标 -> 全屏坐标。"""
        if not self.mm_mode:
            return sx, sy
        x0, y0, _, _ = MM
        return int(sx / MM_ZOOM + x0), int(sy / MM_ZOOM + y0)

    # ---------- 事件 ----------
    def on_mouse(self, evt, x, y, flags, param):
        if evt == cv2.EVENT_LBUTTONDOWN:
            self.drag = ((x, y), (x, y))
        elif evt == cv2.EVENT_MOUSEMOVE and self.drag:
            self.drag = (self.drag[0], (x, y))
        elif evt == cv2.EVENT_LBUTTONUP and self.drag:
            (x1, y1), (x2, y2) = self.drag
            self.drag = None
            fx1, fy1 = self.to_full(x1, y1)
            fx2, fy2 = self.to_full(x2, y2)
            if abs(fx2 - fx1) < 3 or abs(fy2 - fy1) < 3:
                return
            self.boxes.append((self.cls, min(fx1, fx2), min(fy1, fy2),
                               max(fx1, fx2), max(fy1, fy2)))
            print(f"  添加: {LABELS.get(self.cls)} ({min(fx1,fx2)},{min(fy1,fy2)})-"
                  f"({max(fx1,fx2)},{max(fy1,fy2)})")
            self.render()

    def run(self):
        cv2.namedWindow("manual-annotator")
        cv2.setMouseCallback("manual-annotator", self.on_mouse)
        self.load()
        self.render()
        while True:
            key = cv2.waitKey(0) & 0xFF
            if key == 27:                       # ESC
                self.save()
                break
            elif key == ord("m"):
                self.mm_mode = not self.mm_mode
                print("模式:", "小地图" if self.mm_mode else "全屏")
                self.render()
            elif key in FULL_CLASSES and not self.mm_mode:
                self.cls, name = FULL_CLASSES[key]
                print("类别:", name)
            elif key in MM_CLASSES and self.mm_mode:
                self.cls, name = MM_CLASSES[key]
                print("类别:", name)
            elif key == ord("d") and self.boxes:
                c = self.boxes.pop()
                print("删除:", LABELS.get(c[0]))
                self.render()
            elif key == ord("s"):
                self.save()
            elif key == ord("n"):
                self.save()
                self.idx = min(len(self.paths) - 1, self.idx + 1)
                self.load()
                self.render()
            elif key == ord("p"):
                self.save()
                self.idx = max(0, self.idx - 1)
                self.load()
                self.render()
        cv2.destroyAllWindows()


def main():
    args = sys.argv[1:]
    if args:
        paths = [Path(a) for a in args if Path(a).exists()]
    else:
        paths = sorted((ROOT / "temp" / "ann").glob("s*.png"))
    if not paths:
        print("没有可标注的图片")
        return
    ManualAnnotator(paths).run()


if __name__ == "__main__":
    main()
