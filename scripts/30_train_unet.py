"""M3 · 训练脚本（逐帧 U-Net baseline）

用法（GPU, RTX 4060 8GB，约 15-25 分钟）：
    python scripts/30_train_unet.py --iters 4000 --batch 8 --patch 256

烟囱测试（1 分钟，验证代码能跑，不是真训练）：
    python scripts/30_train_unet.py --iters 30 --batch 4 --patch 128 --val-every 10

输出：
    runs/unet_v0/best.pt        按 val PSNR 存最优
    runs/unet_v0/log.csv        每 val_every 次记录一次
    runs/unet_v0/curves.png     训练曲线
"""

from __future__ import annotations

import argparse
import csv
import json
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
from src.models.unet import UNetEnhance, count_params  # noqa: E402
from src.synthesis import LowLightConfig  # noqa: E402
from src.train_utils import (  # noqa: E402
    LowLightPairDataset, enhancement_loss, lr_lambda, temporal_warp_loss,
)

RUNS = ROOT / "runs" / "unet_v0"


def to_u8(t: torch.Tensor) -> np.ndarray:
    return (t.clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).round().astype(np.uint8)


@torch.no_grad()
def quick_val(net, loader, device, max_batches: int = 4) -> float:
    """在 val 序列上算 PSNR。用 DataLoader（固定 seed 子集），不追求全量，够看趋势。"""
    net.eval()
    ps = []
    for i, (lo_t, gt_t, _gt_t1) in enumerate(loader):
        if i >= max_batches:
            break
        pred = net(lo_t.to(device))
        for b in range(pred.shape[0]):
            ps.append(psnr_np(to_u8(pred[b]), to_u8(gt_t[b])))
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
    ap.add_argument("--num-workers", type=int, default=0, help="Windows 上>0 经常崩，默认 0")
    ap.add_argument("--temporal-weight", type=float, default=0.0,
                    help="T2 warp 一致性损失权重 λ。0=纯逐帧（M3），>0=加时序项（M4-T2）")
    ap.add_argument("--run-name", default="unet_v0")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    RUNS = ROOT / "runs" / args.run_name
    RUNS.mkdir(parents=True, exist_ok=True)
    dev = torch.device(args.device)
    print(f"[env] device={dev}  torch={torch.__version__}")

    cfg = LowLightConfig()
    net = UNetEnhance().to(dev)
    n_params = count_params(net)
    print(f"[model] UNetEnhance  params={n_params/1e6:.2f}M")

    use_t2 = args.temporal_weight > 0
    ds_train = LowLightPairDataset(SPLIT.train, cfg, patch=args.patch, base_seed=1000,
                                   return_next_low=use_t2)
    ds_val = LowLightPairDataset(SPLIT.val, cfg, patch=args.patch, base_seed=9000)
    # val 的噪声 seed 固定（不 set_epoch），保证 val 曲线可比
    dl_train = DataLoader(ds_train, batch_size=args.batch, shuffle=True,
                          num_workers=args.num_workers, drop_last=True, pin_memory=True)
    dl_val = DataLoader(ds_val, batch_size=args.batch, shuffle=False,
                        num_workers=args.num_workers, pin_memory=True)

    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: lr_lambda(s, args.warmup, args.iters))
    scaler = torch.amp.GradScaler("cuda", enabled=(dev.type == "cuda"))

    log_path = RUNS / "log.csv"
    log_f = open(log_path, "w", newline="", encoding="utf-8")
    log = csv.writer(log_f)
    log.writerow(["iter", "lr", "loss", "loss_l1", "loss_ssim", "val_psnr", "sec_per_iter"])

    best_psnr = -1.0
    it = 0
    t_start = time.perf_counter()
    hist = []
    print(f"[train] iters={args.iters} batch={args.batch} patch={args.patch} "
          f"train_pairs={len(ds_train)} val_pairs={len(ds_val)}")

    while it < args.iters:
        for batch in dl_train:
            if it >= args.iters:
                break
            ds_train.set_epoch(it // max(1, len(dl_train)))  # 换 epoch → 换噪声

            if use_t2:
                lo_t, gt_t, _gt_t1, lo_t1, flow = batch
                lo_t1 = lo_t1.to(dev, non_blocking=True)
                flow = flow.to(dev, non_blocking=True)
            else:
                lo_t, gt_t, _gt_t1 = batch

            lo_t = lo_t.to(dev, non_blocking=True)
            gt_t = gt_t.to(dev, non_blocking=True)

            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=(dev.type == "cuda")):
                pred = net(lo_t)
                loss = enhancement_loss(pred, gt_t)
                if use_t2:
                    # T2：两帧都前向，warp 一致性把相邻输出绑在一起。
                    # 注意 gt_t1 也参与监督（第二帧的 L1/SSIM），否则网络只在 t 上对齐。
                    pred1 = net(lo_t1)
                    loss = loss + enhancement_loss(pred1, _gt_t1.to(dev))
                    l_temp = temporal_warp_loss(pred, pred1, flow)
                    loss = loss + args.temporal_weight * l_temp
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
                log.writerow([it, opt.param_groups[0]["lr"], float(loss.item()), "", "", vp,
                              (time.perf_counter() - t_start) / (it + 1)])
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

    # 训练曲线
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        its = [h[0] for h in hist]
        losses = [h[1] for h in hist]
        vps = [h[2] for h in hist]
        fig, ax1 = plt.subplots(figsize=(7, 4))
        ax1.plot(its, losses, label="train loss", color="tab:blue")
        ax1.set_xlabel("iter"); ax1.set_ylabel("loss")
        ax2 = ax1.twinx()
        ax2.plot(its, vps, label="val PSNR", color="tab:orange")
        ax2.set_ylabel("val PSNR (dB)")
        fig.legend(loc="upper right")
        fig.tight_layout()
        fig.savefig(RUNS / "curves.png", dpi=120)
        print(f"[out] {RUNS/'curves.png'}")
    except Exception as e:
        print(f"[warn] 画曲线失败: {e}")

    print(f"[done] best val PSNR={best_psnr:.2f}  → {RUNS/'best.pt'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
