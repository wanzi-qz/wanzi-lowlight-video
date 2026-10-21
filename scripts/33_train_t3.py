"""M4-T3 · 循环 U-Net 训练（ConvGRU，truncated BPTT-2）

============================================================
训练设计
============================================================
- 数据：复用 T2 的帧对管道（LowLightPairDataset, return_next_low=True），
  每个样本 = 相邻 2 帧（lo_t, gt_t, gt_t1, lo_t1, flow）
- 前向（truncated BPTT，长度 2）：
    out_t,  h = net(lo_t,  h=None)     # 第 t 帧：无历史
    out_t1, _ = net(lo_t1, h)          # 第 t+1 帧：带着 t 的记忆
  两帧都过主损失（同 T2），h 的梯度沿 t+1 → t 回传 —— 这是循环网络
  "学会记忆"的唯一途径；只监督 t+1 的话 h 学不到东西。
- 时序损失：λ·temporal_warp_loss(out_t, out_t1, flow)，λ=3.0
  （T2 的教训：λ 必须让 l_temp 梯度占比 ≥10%，否则等于没加。
   启动后第一个打印点就检查 l_temp 数值是否在降。）
- h 在样本边界清零（batch 内样本来自不同序列，不能互相污染）

用法：
    python scripts/33_train_t3.py --iters 4000 --batch 8 --patch 256
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data import SPLIT  # noqa: E402
from src.metrics import psnr as psnr_np  # noqa: E402
from src.models.unet_gru import UNetRecurrent, count_params  # noqa: E402
from src.synthesis import LowLightConfig  # noqa: E402
from src.train_utils import (  # noqa: E402
    LowLightPairDataset, enhancement_loss, lr_lambda, temporal_warp_loss,
)


def to_u8(t: torch.Tensor) -> np.ndarray:
    return (t.clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).round().astype(np.uint8)


@torch.no_grad()
def quick_val(net, loader, device, max_batches: int = 4) -> float:
    """val PSNR：stateful 方式（batch 内帧对顺序前向，h 跨两帧）——测真实形态。"""
    net.eval()
    ps = []
    for i, batch in enumerate(loader):
        if i >= max_batches:
            break
        lo_t, gt_t, _gt_t1, lo_t1, _flow = batch
        out_t, h = net.forward_state(lo_t.to(device), None)
        out_t1, _ = net.forward_state(lo_t1.to(device), h)
        for b in range(out_t.shape[0]):
            ps.append(psnr_np(to_u8(out_t[b]), to_u8(gt_t[b])))
            ps.append(psnr_np(to_u8(out_t1[b]), to_u8(_gt_t1[b])))
    net.train()
    return float(np.mean(ps))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=4000)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--patch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--warmup", type=int, default=200)
    ap.add_argument("--val-every", type=int, default=250)
    ap.add_argument("--num-workers", type=int, default=0)
    ap.add_argument("--temporal-weight", type=float, default=3.0)
    ap.add_argument("--run-name", default="unet_t3")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    RUNS = ROOT / "runs" / args.run_name
    RUNS.mkdir(parents=True, exist_ok=True)
    dev = torch.device(args.device)
    print(f"[env] device={dev}  torch={torch.__version__}")

    cfg = LowLightConfig()
    net = UNetRecurrent().to(dev)
    print(f"[model] UNetRecurrent  params={count_params(net)/1e6:.2f}M  "
          f"(GRU 部分 +{(count_params(net)-3_895_203)/1e6:.2f}M 相对逐帧基线)")

    ds_train = LowLightPairDataset(SPLIT.train, cfg, patch=args.patch, base_seed=1000,
                                   return_next_low=True)
    ds_val = LowLightPairDataset(SPLIT.val, cfg, patch=args.patch, base_seed=9000,
                                 return_next_low=True)
    dl_train = DataLoader(ds_train, batch_size=args.batch, shuffle=True,
                          num_workers=args.num_workers, drop_last=True, pin_memory=True)
    dl_val = DataLoader(ds_val, batch_size=args.batch, shuffle=False,
                        num_workers=args.num_workers, pin_memory=True)

    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: lr_lambda(s, args.warmup, args.iters))
    scaler = torch.amp.GradScaler("cuda", enabled=(dev.type == "cuda"))

    log_f = open(RUNS / "log.csv", "w", newline="", encoding="utf-8")
    log = csv.writer(log_f)
    log.writerow(["iter", "lr", "loss", "l_temp", "val_psnr", "sec_per_iter"])

    best_psnr = -1.0
    it = 0
    t_start = time.perf_counter()
    hist = []
    print(f"[train] iters={args.iters} batch={args.batch} patch={args.patch} "
          f"lambda_temp={args.temporal_weight}")

    while it < args.iters:
        for batch in dl_train:
            if it >= args.iters:
                break
            ds_train.set_epoch(it // max(1, len(dl_train)))

            lo_t, gt_t, gt_t1, lo_t1, flow = batch
            lo_t = lo_t.to(dev, non_blocking=True)
            gt_t = gt_t.to(dev, non_blocking=True)
            gt_t1 = gt_t1.to(dev, non_blocking=True)
            lo_t1 = lo_t1.to(dev, non_blocking=True)
            flow = flow.to(dev, non_blocking=True)

            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=(dev.type == "cuda")):
                # truncated BPTT-2：t 帧无历史，t+1 帧带记忆，两帧都监督
                out_t, h = net.forward_state(lo_t, None)
                out_t1, _ = net.forward_state(lo_t1, h)
                loss = (enhancement_loss(out_t, gt_t)
                        + enhancement_loss(out_t1, gt_t1)
                        + args.temporal_weight * temporal_warp_loss(out_t, out_t1, flow))
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            sched.step()

            if it % 25 == 0:
                dt = (time.perf_counter() - t_start) / max(1, it + 1)
                print(f"  it={it:5d}  loss={loss.item():.4f}  lr={opt.param_groups[0]['lr']:.2e}  "
                      f"{dt:.2f}s/it", flush=True)

            if it % args.val_every == 0 or it == args.iters - 1:
                vp = quick_val(net, dl_val, dev)
                hist.append((it, float(loss.item()), vp))
                log.writerow([it, opt.param_groups[0]["lr"], float(loss.item()),
                              "", "", vp, (time.perf_counter() - t_start) / (it + 1)])
                log_f.flush()
                marker = ""
                if vp > best_psnr:
                    best_psnr = vp
                    torch.save({"model": net.state_dict(), "iter": it,
                                "cfg": cfg.to_dict(), "args": vars(args)},
                               RUNS / "best.pt")
                    marker = "  ← best"
                print(f"  [val] it={it}  val_psnr={vp:.2f}  best={best_psnr:.2f}{marker}", flush=True)
            it += 1

    log_f.close()
    torch.save({"model": net.state_dict(), "iter": it, "cfg": cfg.to_dict(),
                "args": vars(args)}, RUNS / "last.pt")
    print(f"[done] best val PSNR={best_psnr:.2f}  → {RUNS/'best.pt'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
