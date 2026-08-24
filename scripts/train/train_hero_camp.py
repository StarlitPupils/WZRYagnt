# -*- coding: utf-8 -*-
"""小地图英雄圈 3 类阵营 CNN 训练（用户真值强化训练）。

数据：用户标注的真值英雄圈（自己8/队友44/敌人24 = 76 个）
输入：圈内头像 32x32（HSV/通道归一化）
类别：0 自己 / 1 队友 / 2 敌人

输出：runs/mm_hero/hero_camp.pt
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from torch.utils.data import Dataset, DataLoader  # noqa: E402

import cv2  # noqa: E402

SIZE = 32
LABELS = {11: 0, 12: 1, 13: 2}     # 标注类 -> 阵营类


class HeroCampNet(nn.Module):
    def __init__(self, nc=3):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.fc = nn.Linear(128 * 4 * 4, nc)
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.max_pool2d(x, 2)
        x = x.view(x.size(0), -1)
        return self.fc(x)


def load_samples():
    """从用户标注裁剪真值英雄圈 -> (images, labels)。"""
    imgs, labels = [], []
    for i in range(1, 13):
        img = cv2.imdecode(np.fromfile(f"temp/ann/s{i:02d}.png", dtype=np.uint8),
                           cv2.IMREAD_COLOR)
        h, w = img.shape[:2]
        mm = img[0:232, 0:232]
        for ln in open(f"temp/ann/s{i:02d}.txt", encoding="utf-8").read().splitlines():
            p = ln.split()
            if len(p) != 5:
                continue
            c = int(p[0])
            if c not in LABELS:
                continue
            cx, cy, bw, bh = map(float, p[1:])
            px, py = int(cx * w), int(cy * h)
            pw_, ph_ = int(bw * w), int(bh * h)
            crop = mm[max(0, py):py + ph_, max(0, px):px + pw_]
            if crop.size == 0:
                continue
            crop = cv2.resize(crop, (SIZE, SIZE), interpolation=cv2.INTER_AREA)
            imgs.append(crop)
            labels.append(LABELS[c])
    print(f"真值样本: {len(imgs)} (己/队/敌: "
          f"{labels.count(0)}/{labels.count(1)}/{labels.count(2)})")
    return imgs, labels


def augment(img, rng):
    im = img.copy()
    if rng.random() < 0.5:
        im = cv2.flip(im, 1)
    a = rng.uniform(0.75, 1.25)
    b = rng.uniform(-25, 25)
    im = cv2.convertScaleAbs(im, alpha=a, beta=b)
    if rng.random() < 0.3:
        im = cv2.rotate(im, rng.choice([cv2.ROTATE_90_CLOCKWISE,
                                        cv2.ROTATE_90_COUNTERCLOCKWISE]))
    return im


class HeroDS(Dataset):
    def __init__(self, imgs, labels, train=True, n_aug=12, rng=None):
        self.train = train
        self.rng = rng or np.random.RandomState(1)
        self.base = imgs
        self.labels = labels
        self.n_aug = n_aug

    def __len__(self):
        return len(self.base) * (self.n_aug if self.train else 1)

    def __getitem__(self, idx):
        i = idx // self.n_aug if self.train else idx
        img = self.base[i]
        if self.train:
            img = augment(img, np.random.RandomState(idx))
        x = img.astype(np.float32) / 255.0
        x = x.transpose(2, 0, 1)
        return torch.from_numpy(x), self.labels[i]


def main():
    imgs, labels = load_samples()
    # 训练/验证划分（真值按样本 80/20）
    n = len(imgs)
    idx = np.arange(n)
    np.random.seed(3)
    np.random.shuffle(idx)
    n_tr = int(n * 0.8)
    tr_ds = HeroDS([imgs[i] for i in idx[:n_tr]], [labels[i] for i in idx[:n_tr]],
                   train=True)
    va_ds = HeroDS([imgs[i] for i in idx[n_tr:]], [labels[i] for i in idx[n_tr:]],
                   train=False)
    tr_ld = DataLoader(tr_ds, batch_size=32, shuffle=True)
    va_ld = DataLoader(va_ds, batch_size=32)

    net = HeroCampNet(3)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    best = 0.0
    for ep in range(120):
        net.train()
        for xb, yb in tr_ld:
            opt.zero_grad()
            out = net(xb)
            loss = F.cross_entropy(out, yb)
            loss.backward()
            opt.step()
        net.eval()
        correct = total = 0
        with torch.no_grad():
            for xb, yb in va_ld:
                pred = net(xb).argmax(1)
                correct += (pred == yb).sum().item()
                total += len(yb)
        acc = correct / max(1, total)
        if acc > best:
            best = acc
            out_dir = ROOT / "runs" / "mm_hero"
            out_dir.mkdir(parents=True, exist_ok=True)
            torch.save(net.state_dict(), out_dir / "hero_camp.pt")
        if ep % 20 == 0:
            print(f"ep{ep}: val acc {acc:.2f}")
    print(f"最佳 val acc: {best:.2f} -> runs/mm_hero/hero_camp.pt")


if __name__ == "__main__":
    main()
