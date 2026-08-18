# -*- coding: utf-8 -*-
"""英雄识别强化训练：128 模板 -> 数据增强 -> CNN 分类器。

数据：data/heroes/<英雄名>.png（96x96，用户提供的网格图裁剪）
增强：旋转 ±20°、缩放 0.8-1.2、亮度/对比度抖动、平移、噪声、HSV 扰动
模型：小型 CNN（3 conv + 2 fc），128 类
输出：runs/heroes/hero_cnn.pt + 类别映射 hero_classes.json

用法：
    venv\\Scripts\\python.exe scripts\\train\\train_hero_cnn.py [--epochs 60] [--aug 20]
"""
import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

HERO_DIR = ROOT / "data" / "heroes"
OUT_DIR = ROOT / "runs" / "heroes"


def imread_unicode(path):
    data = np.fromfile(str(path), dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


class HeroCNN(nn.Module):
    """小型 CNN：96x96x3 -> 128 类。"""

    def __init__(self, n_classes):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d(2),                                   # 48
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2),                                   # 24
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.MaxPool2d(2),                                   # 12
            nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.MaxPool2d(2),                                   # 6
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 6 * 6, 256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, n_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


def augment(img, rng):
    """随机增强单个头像（96x96）。"""
    h, w = img.shape[:2]
    # 旋转
    ang = rng.uniform(-20, 20)
    M = cv2.getRotationMatrix2D((w / 2, h / 2), ang, 1.0)
    img = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REPLICATE)
    # 缩放 + 平移
    s = rng.uniform(0.8, 1.2)
    dx = rng.uniform(-8, 8)
    dy = rng.uniform(-8, 8)
    M = np.float32([[s, 0, dx], [0, s, dy]])
    img = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REPLICATE)
    # 亮度/对比度
    alpha = rng.uniform(0.8, 1.2)
    beta = rng.uniform(-20, 20)
    img = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
    # HSV 扰动
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[..., 0] = (hsv[..., 0] + rng.uniform(-10, 10)) % 180
    hsv[..., 1] = np.clip(hsv[..., 1] * rng.uniform(0.85, 1.15), 0, 255)
    img = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    # 高斯噪声
    if rng.random() < 0.5:
        noise = rng.normal(0, 4, img.shape).astype(np.float32)
        img = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--aug", type=int, default=20, help="每模板增强倍数")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    args = ap.parse_args()

    # 加载模板
    files = sorted(HERO_DIR.glob("*.png"))
    names = [f.stem for f in files]
    names.sort()
    cls2id = {n: i for i, n in enumerate(names)}
    id2cls = {i: n for n, i in cls2id.items()}
    print(f"英雄类别: {len(names)} 个")

    # 数据增强构建训练集
    rng = np.random.RandomState(0)
    X, y = [], []
    for name in names:
        img = imread_unicode(HERO_DIR / f"{name}.png")
        if img is None:
            continue
        img = cv2.resize(img, (96, 96), interpolation=cv2.INTER_AREA)
        X.append(img)
        y.append(cls2id[name])
        for _ in range(args.aug):
            X.append(augment(img, rng))
            y.append(cls2id[name])
    X = np.stack(X).astype(np.float32) / 255.0   # (N,96,96,3)
    X = X.transpose(0, 3, 1, 2)                  # (N,3,96,96)
    y = np.array(y)
    print(f"训练样本: {len(X)}（含增强）")

    # 划分验证集（每类最后 2 张 = 原始+1增强）
    n_classes = len(names)
    val_mask = np.zeros(len(y), bool)
    # 取每类最后 2 个样本做验证
    for c in range(n_classes):
        idx = np.where(y == c)[0]
        val_mask[idx[-2:]] = True
    X_tr, y_tr = X[~val_mask], y[~val_mask]
    X_va, y_va = X[val_mask], y[val_mask]
    print(f"训练 {len(X_tr)} / 验证 {len(X_va)}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"设备: {device}")
    net = HeroCNN(n_classes).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=20, gamma=0.5)
    crit = nn.CrossEntropyLoss()

    def batch_iter(x, yy, bs, shuffle=True):
        n = len(x)
        idx = np.random.permutation(n) if shuffle else np.arange(n)
        for i in range(0, n, bs):
            sel = idx[i:i + bs]
            yield (torch.from_numpy(x[sel]).to(device),
                   torch.from_numpy(yy[sel]).long().to(device))

    t0 = time.time()
    for epoch in range(args.epochs):
        net.train()
        losses = []
        for xb, yb in batch_iter(X_tr, y_tr, args.batch):
            opt.zero_grad()
            loss = crit(net(xb), yb)
            loss.backward()
            opt.step()
            losses.append(float(loss))
        sched.step()
        # 验证
        net.eval()
        correct = 0
        with torch.no_grad():
            for xb, yb in batch_iter(X_va, y_va, args.batch, shuffle=False):
                pred = net(xb).argmax(1)
                correct += (pred == yb).sum().item()
        acc = correct / len(y_va)
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"epoch {epoch+1}/{args.epochs} loss={sum(losses)/len(losses):.4f} "
                  f"val_acc={acc*100:.1f}% ({time.time()-t0:.0f}s)")

    # 保存
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(net.state_dict(), OUT_DIR / "hero_cnn.pt")
    (OUT_DIR / "hero_classes.json").write_text(
        json.dumps(id2cls, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"模型已保存: {OUT_DIR / 'hero_cnn.pt'}（{len(names)} 类）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
