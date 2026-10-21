"""一次性诊断 · T2 为什么没降低 tOF

三个问题，一次回答：
Q1. T2 模型的 l_temp 是否真的比逐帧模型低？（损失有没有被优化）
Q2. l_temp 的梯度范数占主损失的多少？（λ=0.3 是否太小）
Q3. 光度掩码是否饱和/失效？（mean(mask)、被掩码丢弃的比例）

另外验证坐标约定 bug 的影响：align_corners=False + linspace 网格 vs 对齐写法，
比较 warped 差多少。
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data import SPLIT  # noqa: E402
from src.models.unet import UNetEnhance  # noqa: E402
from src.synthesis import LowLightConfig  # noqa: E402
from src.train_utils import (  # noqa: E402
    LowLightPairDataset, enhancement_loss, temporal_warp_loss,
)
from torch.utils.data import DataLoader  # noqa: E402

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[env] device={DEV}")

cfg = LowLightConfig()
ds = LowLightPairDataset(SPLIT.val, cfg, patch=256, base_seed=9000, return_next_low=True)
dl = DataLoader(ds, batch_size=8, shuffle=False)
print(f"[data] val batches={len(dl)}")

ckpts = {
    "per-frame (unet_v0)": ROOT / "runs" / "unet_v0" / "best.pt",
    "T2 lam=3.0 (unet_t2_lam3)": ROOT / "runs" / "unet_t2_lam3" / "best.pt",
}

nets = {}
for name, p in ckpts.items():
    net = UNetEnhance().to(DEV).eval()
    sd = torch.load(p, map_location=DEV, weights_only=False)
    net.load_state_dict(sd["model"])
    nets[name] = net
    print(f"[ckpt] {name}: iter={sd.get('iter')}")

# ---------------------------------------------------------------- Q1+Q3
print("\n== Q1/Q3: l_temp 与掩码统计（val，无梯度）==")
with torch.no_grad():
    for name, net in nets.items():
        vals, masks, diffs = [], [], []
        for i, batch in enumerate(dl):
            if i >= 6:
                break
            lo_t, _gt, _gt1, lo_t1, flow = batch
            pred = net(lo_t.to(DEV))
            pred1 = net(lo_t1.to(DEV))
            # 复用 loss 内部逻辑拿 mask/diff
            b, _, h, w = pred.shape
            gx, gy = torch.meshgrid(
                torch.linspace(-1, 1, w, device=DEV), torch.linspace(-1, 1, h, device=DEV),
                indexing="xy")
            base = torch.stack([gx, gy], dim=0)[None]
            scale = torch.tensor([2.0 / (w - 1), 2.0 / (h - 1)], device=DEV).view(1, 2, 1, 1)
            grid = (base + flow.to(DEV) * scale).permute(0, 2, 3, 1)
            warped = torch.nn.functional.grid_sample(
                pred, grid, mode="bilinear", padding_mode="border", align_corners=False)
            diff = (warped - pred1).abs().mean(dim=1, keepdim=True)
            mask = torch.exp(-diff / 0.1)
            vals.append((mask * diff).mean().item())
            masks.append(mask.mean().item())
            diffs.append(diff.mean().item())
        print(f"  {name:22s}  l_temp={np.mean(vals):.5f}  mean|warp diff|={np.mean(diffs):.5f}  "
              f"mean(mask)={np.mean(masks):.4f}")

# ---------------------------------------------------------------- Q2
print("\n== Q2: 梯度范数占比（逐帧 ckpt，一个 batch）==")
net = nets["per-frame (unet_v0)"]
net.train()
batch = next(iter(dl))
lo_t, gt_t, gt_t1, lo_t1, flow = batch
lo_t, gt_t, gt_t1 = lo_t.to(DEV), gt_t.to(DEV), gt_t1.to(DEV)
lo_t1, flow = lo_t1.to(DEV), flow.to(DEV)

pred = net(lo_t)
pred1 = net(lo_t1)
main_loss = enhancement_loss(pred, gt_t) + enhancement_loss(pred1, gt_t1)
l_temp = temporal_warp_loss(pred, pred1, flow)
print(f"  main_loss={main_loss.item():.4f}  l_temp={l_temp.item():.5f}  "
      f"ratio(l_temp/main)={l_temp.item()/main_loss.item():.4f}")

for tag, loss in [("main", main_loss), ("l_temp alone", l_temp),
                  ("total lam=0.3", main_loss + 0.3 * l_temp),
                  ("total lam=3.0", main_loss + 3.0 * l_temp)]:
    net.zero_grad(set_to_none=True)
    loss.backward(retain_graph=True)
    gn = torch.sqrt(sum((p.grad ** 2).sum() for p in net.parameters() if p.grad is not None))
    print(f"  grad_norm[{tag:14s}] = {gn.item():.4f}")
net.zero_grad(set_to_none=True)

# ---------------------------------------------------------------- 坐标约定
print("\n== 坐标约定检查：linspace 网格 + align_corners=False 的系统错位 ==")
# 纯平移流场 flow=(2,0)（2 像素向右），warp 一张斜坡图，理想输出=图向右平移 2px。
H = W = 256
img = torch.arange(W, dtype=torch.float32).repeat(H, 1) / W  # 水平斜坡
img = img[None, None].to(DEV)
flow = torch.zeros(1, 2, H, W, device=DEV)
flow[:, 0] = 2.0  # 目标→源位移 +2px（取源在右边 2px → 输出整体左移 2px）
for ac in (False, True):
    gx, gy = torch.meshgrid(
        torch.linspace(-1, 1, W, device=DEV), torch.linspace(-1, 1, H, device=DEV),
        indexing="xy")
    base = torch.stack([gx, gy], dim=0)[None]
    scale = torch.tensor([2.0 / (W - 1), 2.0 / (H - 1)], device=DEV).view(1, 2, 1, 1)
    grid = (base + flow * scale).permute(0, 2, 3, 1)
    out = torch.nn.functional.grid_sample(img, grid, mode="bilinear",
                                          padding_mode="border", align_corners=ac)
    # 理想：out[y,x] = img[y,x+2]（align_corners=True 口径下）
    err = (out[0, 0, :, 5:-5] - img[0, 0, :, 7:-3]).abs().mean().item()
    print(f"  align_corners={ac}: 与理想平移的 MAE = {err:.6f} (像素值域 [0,1], 2px 位移)")

print("\n[done]")
