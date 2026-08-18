# -*- coding: utf-8 -*-
"""英雄头像识别（理解层高级能力）：小地图头像/顶部头像条 -> 英雄名。

数据源（用户提供 4 张英雄网格图，头像下方是名字）：
  - 第1张（12英雄）：阿轲 高渐离 刘禅 庄周 鲁班七号 孙尚香 赢政 妲己 墨子 赵云 小乔 廉颇
  - 第2张（36英雄）：蛇焰使者 心魔六耳 元流之子 大禹 蚩尤 孙权 空空儿 影 少司缘 大司命
    敖隐 海诺 朵莉亚 亚连 姬小满 莱西奥 赵怀真 海月 戈娅 桑启 羿 云缨 艾琳 金蝉
    司空震 澜 夏洛特 阿古朵 蒙恬 镜 蒙犽 鲁班大师 西施 马超 曜 云中君
  - 第3张（40英雄）：瑶 盘古 猪八戒 嫦娥 上官婉儿 李信 沈梦溪 伽罗 盾山 司马懿
    宫本武藏 孙策 元歌 米莱狄 狂铁 弈星 裴擒虎 杨玉环 公孙离 明世隐 梦奇 女娲
    苏烈 百里玄策 百里守约 铠 鬼谷子 干将莫邪 东皇太一 大乔 苍 黄忠 诸葛亮 哪吒
    太乙真人 杨戬 橘右京 马可波罗 雅典娜 夏侯惇
  - 第4张（40英雄）：蔡文姬 关羽 虞姬 不知火舞 刘邦 李元芳 钟馗 李白 娜可露露 兰陵王
    刘备 张飞 后羿 牛魔 孙悟空 亚瑟 张良 花木兰 王昭君 韩信 姜子牙 露娜 程咬金
    安琪拉 貂蝉 老夫子 武则天 项羽 达摩 狄仁杰 典韦 曹操 甄姬 周瑜 吕布 芈月
    白起 扁鹊 孙膑 钟无艳

匹配策略（对局中实时）：
  1. 从图像中裁剪头像区域（小地图英雄圈 / 顶部头像条 / 选英雄界面）
  2. 与模板库做相似度匹配（感知哈希 + 颜色直方图 + 模板匹配）
  3. 输出英雄名

数据目录结构：
  data/heroes/<hero_name>.png       标准头像模板（96x96）
  data/heroes/index.json            名字 -> 文件映射
"""
import json
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
HERO_DIR = ROOT / "data" / "heroes"
INDEX_FILE = HERO_DIR / "index.json"

# 用户提供的 4 张图的英雄名单（顺序即网格顺序）
HERO_GRIDS = [
    ["阿轲", "高渐离", "刘禅", "庄周", "鲁班七号", "孙尚香", "赢政", "妲己", "墨子", "赵云", "小乔", "廉颇"],
    ["蛇焰使者", "心魔六耳", "元流之子", "大禹", "蚩尤", "孙权", "空空儿", "影", "少司缘",
     "大司命", "敖隐", "海诺", "朵莉亚", "亚连", "姬小满", "莱西奥", "赵怀真", "海月", "戈娅",
     "桑启", "羿", "云缨", "艾琳", "金蝉", "司空震", "澜", "夏洛特", "阿古朵", "蒙恬", "镜",
     "蒙犽", "鲁班大师", "西施", "马超", "曜", "云中君"],
    ["瑶", "盘古", "猪八戒", "嫦娥", "上官婉儿", "李信", "沈梦溪", "伽罗", "盾山", "司马懿",
     "宫本武藏", "孙策", "元歌", "米莱狄", "狂铁", "弈星", "裴擒虎", "杨玉环", "公孙离",
     "明世隐", "梦奇", "女娲", "苏烈", "百里玄策", "百里守约", "铠", "鬼谷子", "干将莫邪",
     "东皇太一", "大乔", "苍", "黄忠", "诸葛亮", "哪吒", "太乙真人", "杨戬", "橘右京",
     "马可波罗", "雅典娜", "夏侯惇"],
    ["蔡文姬", "关羽", "虞姬", "不知火舞", "刘邦", "李元芳", "钟馗", "李白", "娜可露露", "兰陵王",
     "刘备", "张飞", "后羿", "牛魔", "孙悟空", "亚瑟", "张良", "花木兰", "王昭君", "韩信",
     "姜子牙", "露娜", "程咬金", "安琪拉", "貂蝉", "老夫子", "武则天", "项羽", "达摩", "狄仁杰",
     "典韦", "曹操", "甄姬", "周瑜", "吕布", "芈月", "白起", "扁鹊", "孙膑", "钟无艳"],
]
ALL_HEROES = sorted({h for grid in HERO_GRIDS for h in grid})


def build_index():
    """扫描 data/heroes/*.png 生成 index.json。"""
    if not HERO_DIR.exists():
        return {}
    index = {}
    for f in sorted(HERO_DIR.glob("*.png")):
        index[f.stem] = str(f)
    (HERO_DIR / "index.json").write_text(
        json.dumps(index, ensure_ascii=False, indent=2), encoding="utf-8")
    return index


def load_index():
    """加载 index.json（不存在则扫描）。"""
    if INDEX_FILE.exists():
        return json.loads(INDEX_FILE.read_text(encoding="utf-8"))
    return build_index()


def _phash(img, size=16):
    """感知哈希：缩放到 size x size，转灰度，DCT 低频比较 -> 64bit。"""
    g = cv2.cvtColor(cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA),
                     cv2.COLOR_BGR2GRAY).astype(np.float32)
    dct = cv2.dct(g / 255.0)
    low = dct[:8, :8].flatten()
    return (low > np.median(low)).astype(np.uint8)


def hamming(a, b):
    return int(np.count_nonzero(a != b))


class HeroRecognizer:
    """英雄头像识别器：模板库 + 多特征匹配。"""

    def __init__(self, index=None):
        self.index = index or load_index()
        self._templates = {}
        for name, path in self.index.items():
            img = cv2.imread(str(path))
            if img is not None:
                self._templates[name] = img
        print(f"[heroes] 加载 {len(self._templates)} 个英雄模板")

    def recognize(self, img, top_k=3):
        """识别单个头像裁剪图 -> [(英雄名, 分数), ...]。

        特征：感知哈希(0.5) + 颜色直方图(0.3) + 尺寸归一化模板匹配(0.2)。
        """
        if not self._templates:
            return []
        target = cv2.resize(img, (96, 96), interpolation=cv2.INTER_AREA)
        target_hash = _phash(target)
        target_hist = cv2.calcHist([target], [0, 1, 2], None, [8, 8, 8],
                                   [0, 256, 0, 256, 0, 256])
        cv2.normalize(target_hist, target_hist)
        scores = []
        for name, tpl in self._templates.items():
            t96 = cv2.resize(tpl, (96, 96), interpolation=cv2.INTER_AREA)
            # 感知哈希
            h = hamming(target_hash, _phash(t96))
            phash_score = 1.0 - h / 64.0
            # 颜色直方图
            th = cv2.calcHist([t96], [0, 1, 2], None, [8, 8, 8],
                              [0, 256, 0, 256, 0, 256])
            cv2.normalize(th, th)
            hist_score = float(cv2.compareHist(target_hist, th, cv2.HISTCMP_CORREL))
            # 综合
            score = 0.6 * phash_score + 0.4 * max(0.0, hist_score)
            scores.append((name, score))
        scores.sort(key=lambda x: -x[1])
        return scores[:top_k]


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print(f"英雄总数: {len(ALL_HEROES)}")
        print(f"模板库: {HERO_DIR}（{len(list(HERO_DIR.glob('*.png')) if HERO_DIR.exists() else [])} 个）")
        print("用法: python hero_recognition.py <头像图片路径>")
    else:
        img = cv2.imread(sys.argv[1])
        if img is None:
            print("无法读取图片")
        else:
            r = HeroRecognizer()
            print("识别结果:", r.recognize(img))
