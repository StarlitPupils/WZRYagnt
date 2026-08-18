# -*- coding: utf-8 -*-
"""英雄网格图 -> 模板库（UTF-8 文件名安全版）。

输入：data/heroes_annotate/img1.png ~ img4.png（用户提供的 4 张英雄网格图）
输出：data/heroes/<英雄名>.png（96x96）+ index.json

用法：
    venv\\Scripts\\python.exe scripts\\train\\build_hero_templates.py
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

ANNOTATE_DIR = ROOT / "data" / "heroes_annotate"
OUT_DIR = ROOT / "data" / "heroes"

# 每张图的英雄名（行优先，10 列；img4 第 2 行 2 个）
GRIDS = {
    "img1.png": [
        ["蛇焰使者", "心魔六耳", "元流之子", "大禹", "元流之子", "蚩尤", "孙权",
         "元流之子", "空空儿", "影"],
        ["少司缘", "元流之子", "元流之子", "大司命", "敖隐", "海诺", "朵莉亚",
         "亚连", "姬小满", "莱西奥"],
        ["赵怀真", "海月", "戈娅", "桑启", "羿", "云缨", "艾琳", "金蝉", "司空震", "澜"],
        ["夏洛特", "阿古朵", "蒙恬", "镜", "蒙犽", "鲁班大师", "西施", "马超", "曜", "云中君"],
    ],
    "img2.png": [
        ["瑶", "盘古", "猪八戒", "嫦娥", "上官婉儿", "李信", "沈梦溪", "伽罗", "盾山", "司马懿"],
        ["宫本武藏", "孙策", "元歌", "米莱狄", "狂铁", "弈星", "裴擒虎", "杨玉环",
         "公孙离", "明世隐"],
        ["梦奇", "女娲", "苏烈", "百里玄策", "百里守约", "铠", "鬼谷子", "干将莫邪",
         "东皇太一", "大乔"],
        ["苍", "黄忠", "诸葛亮", "哪吒", "太乙真人", "杨戬", "橘右京", "马可波罗",
         "雅典娜", "夏侯惇"],
    ],
    "img3.png": [
        ["蔡文姬", "关羽", "虞姬", "不知火舞", "刘邦", "李元芳", "钟馗", "李白",
         "娜可露露", "兰陵王"],
        ["刘备", "张飞", "后羿", "牛魔", "孙悟空", "亚瑟", "张良", "花木兰", "王昭君", "韩信"],
        ["姜子牙", "露娜", "程咬金", "安琪拉", "貂蝉", "老夫子", "武则天", "项羽",
         "达摩", "狄仁杰"],
        ["典韦", "曹操", "甄姬", "周瑜", "吕布", "芈月", "白起", "扁鹊", "孙膑", "钟无艳"],
    ],
    "img4.png": [
        ["阿轲", "高渐离", "刘禅", "庄周", "鲁班七号", "孙尚香", "赢政", "妲己", "墨子", "赵云"],
        ["小乔", "廉颇"],
    ],
}


def find_bands(img):
    """检测头像行带（彩色像素行投影 + 平滑）。"""
    h, w = img.shape[:2]
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    colorful = (hsv[..., 1] > 80).astype(np.uint8)
    row_counts = colorful.sum(axis=1)
    kernel = np.ones(15) / 15
    smooth = np.convolve(row_counts, kernel, mode="same")
    thr = smooth.max() * 0.2
    bands = []
    in_b = False
    for y in range(len(smooth)):
        if smooth[y] > thr and not in_b:
            start = y
            in_b = True
        elif smooth[y] <= thr and in_b:
            if y - start > 20:
                bands.append((start, y))
            in_b = False
    if in_b and len(smooth) - 1 - start > 20:
        bands.append((start, len(smooth) - 1))
    return bands


def find_col_centers(img, y0, y1):
    """行内头像列中心（彩色像素列投影 + 合并）。"""
    w = img.shape[1]
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    colorful = (hsv[..., 1] > 80).astype(np.uint8)
    col_counts = colorful[y0:y1, :].sum(axis=0)
    thr_c = col_counts.max() * 0.25
    cols = []
    in_c = False
    for x in range(w):
        if col_counts[x] > thr_c and not in_c:
            start = x
            in_c = True
        elif col_counts[x] <= thr_c and in_c:
            cols.append((start, x))
            in_c = False
    if in_c:
        cols.append((start, w - 1))
    merged = []
    for c in cols:
        if merged and c[0] - merged[-1][1] < 30:
            merged[-1] = (merged[-1][0], c[1])
        else:
            merged.append(list(c))
    return merged


def main():
    if not ANNOTATE_DIR.exists():
        print(f"请先把 4 张英雄图放到 {ANNOTATE_DIR}（img1.png~img4.png）")
        return 1
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    # 清理旧文件
    for f in OUT_DIR.glob("*.png"):
        f.unlink()
    n_saved = 0
    for fname, rows in GRIDS.items():
        img = cv2.imread(str(ANNOTATE_DIR / fname))
        if img is None:
            print(f"{fname} 读取失败")
            continue
        h, w = img.shape[:2]
        bands = find_bands(img)
        if fname == "img4.png" and len(bands) < 2:
            bands.append((150, 250))   # img4 第二行特殊处理
        for ri, row_names in enumerate(rows):
            if ri >= len(bands):
                print(f"  {fname} 行{ri} 无行带")
                continue
            y0, y1 = bands[ri]
            cy = (y0 + y1) // 2
            cols = find_col_centers(img, y0, y1)
            n_cols = len(row_names)
            cols.sort(key=lambda c: c[1] - c[0], reverse=True)
            top = cols[:n_cols]
            top.sort(key=lambda c: c[0])
            col_centers = [(a + b) // 2 for a, b in top]
            step = col_centers[1] - col_centers[0] if len(col_centers) >= 2 else 160
            r = int(step * 0.42)
            for ci, name in enumerate(row_names):
                if ci >= len(col_centers):
                    break
                cx = col_centers[ci]
                crop = img[max(0, cy - r):cy + r, max(0, cx - r):cx + r]
                if crop.size == 0:
                    continue
                crop = cv2.resize(crop, (96, 96), interpolation=cv2.INTER_AREA)
                ok, buf = cv2.imencode(".png", crop)
                if ok:
                    (OUT_DIR / f"{name}.png").write_bytes(buf.tobytes())
                    n_saved += 1
        print(f"{fname}: OK")
    # index.json
    files = sorted(OUT_DIR.glob("*.png"))
    index = {f.stem: str(f) for f in files}
    (OUT_DIR / "index.json").write_text(
        json.dumps(index, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"完成: {n_saved} 个头像, {len(index)} 个唯一英雄 -> {OUT_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
