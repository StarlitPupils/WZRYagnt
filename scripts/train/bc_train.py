# -*- coding: utf-8 -*-
"""行为克隆训练（M4 前置）：从 npz 特征库训练 BCNet。

数据：scripts/train/encoding.py 的 states_to_dataset 产出
  （units/unit_mask/grid/ui/actions/metas）。

用法：
    venv\\Scripts\\python.exe scripts\\train\\bc_train.py
        [--data data/datasets/bc_v0.npz] [--epochs 50] [--batch 64]
        [--lr 1e-3] [--out runs/bc/bc_v0.pt]

loss = CE(move_theta) + MSE(move_r) + CE(act) + MSE(target)，权重可调。
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from wzry.policy.model import BCNet  # noqa: E402


def load_npz(path):
    d = np.load(path)
    return (torch.from_numpy(d["units"]).float(),
            torch.from_numpy(d["unit_mask"]).float(),
            torch.from_numpy(d["grid"]).float(),
            torch.from_numpy(d["ui"]).float(),
            torch.from_numpy(d["actions"]).float())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(ROOT / "data" / "datasets" / "bc_v0.npz"))
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--out", default=str(ROOT / "runs" / "bc" / "bc_v0.pt"))
    ap.add_argument("--w-move", type=float, default=1.0)
    ap.add_argument("--w-r", type=float, default=1.0)
    ap.add_argument("--w-act", type=float, default=1.0)
    ap.add_argument("--w-target", type=float, default=1.0)
    args = ap.parse_args()

    data_path = Path(args.data)
    if not data_path.exists():
        print(f"数据集不存在: {data_path}（先采集对局并运行 states_to_dataset）")
        return 1
    units, unit_mask, grid, ui, actions = load_npz(str(data_path))
    n = units.shape[0]
    print(f"数据集: {n} 样本  units={tuple(units.shape)} grid={tuple(grid.shape)}")

    # 动作编码布局（与 encoding.encode_action 一致）
    # [0:6) one-hot: move/skill/attack/buy/recall/none, [6]=theta_norm, [7]=r, [8:10]=target
    act_onehot = actions[:, :6]
    act_theta = actions[:, 6] * (2 * np.pi + 1e-6)   # 还原角度
    act_r = actions[:, 7]
    act_target = actions[:, 8:10]

    net = BCNet()
    opt = torch.optim.Adam(net.parameters(), lr=args.lr)
    n_batches = max(1, n // args.batch)
    t0 = time.time()
    for epoch in range(args.epochs):
        perm = torch.randperm(n)
        losses = []
        for b in range(n_batches):
            idx = perm[b * args.batch:(b + 1) * args.batch]
            u, m, g, ui_b = units[idx], unit_mask[idx], grid[idx], ui[idx]
            out = net(u, m, g, ui_b)
            loss = (args.w_move * torch.nn.functional.cross_entropy(out["move_theta"], act_theta[idx].long())
                    + args.w_r * torch.nn.functional.mse_loss(out["move_r"].squeeze(-1), act_r[idx])
                    + args.w_act * torch.nn.functional.cross_entropy(out["act"], act_onehot[idx].argmax(-1))
                    + args.w_target * torch.nn.functional.mse_loss(out["target"], act_target[idx]))
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(float(loss))
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"epoch {epoch + 1}/{args.epochs}  loss={sum(losses) / len(losses):.4f}  "
                  f"({time.time() - t0:.0f}s)")
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(net.state_dict(), out_path)
    print(f"模型已保存: {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
