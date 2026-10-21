"""M4-T3b · 循环 U-Net BPTT-K 训练（检验「记忆长度是 T3 瓶颈」假设）

============================================================
与 33_train_t3.py（BPTT-2）的唯一区别
============================================================
训练片段从 2 帧扩到 K 帧（默认 6）：h 在训练时累积 K-1 步，
弥合「训练只有 1 步历史、测试 24 帧连续推理」的分布差——
这是 T3-BPTT2 负结果（循环收益仅 +1.5%）的首要嫌疑根因。

假设检验的判读（先于实验写下）：
- 若 stateful vs per-frame 重置的循环收益显著拉大（≥10%）→ 记忆长度确实是瓶颈，
  T3 原负结果归因成立，方法有效但训练方式要改
- 若仍 ~1-2% → 「记忆没用上」不是长度问题，是任务/数据层面没有可记忆的
  跨帧结构红利，T3 负结果更扎实

用法：
    python scripts/35_train_t3_long.py --iters 3000 --batch 4 --clip-len 6
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
    LowLightClipDataset, enhancement_loss, lr_lambda, temporal_warp_loss,
)


def to_u8(t: torch.Tensor) -> np.ndarray:
    return (t.clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).round().astype(np.uint8)


@torch.no_grad()
def quick_val(net, loader, device, max_batches: int = 3) -> float:
    net.eval()
    ps = []
    for i, (lo, gt, _flow) in enumerate(loader):
        if i >= max_batches:
            break
        K = lo.shape[1]
        h = None
        outs = []
        for k in range(K):
            out_k, h = net.forward_state(lo[:, k].to(device), h)
            outs.append(out_k)
        for k in range(K):
            for b in range(outs[k].shape[0]):
                ps.append(psnr_np(to_u8(outs[k][b]), to_u8(gt[:, k][b])))
    net.train()
    return float(np.mean(ps))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=3000)
    ap.add_argument("--batch", type=int, default=4, help="clip 数/batch；有效帧数 = batch×K")
    ap.add_argument("--clip-len", type=int, default=6)
    ap.add_argument("--patch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--warmup", type=int, default=200)
    ap.add_argument("--val-every", type=int, default=250)
    ap.add_argument("--num-workers", type=int, default=0)
    ap.add_argument("--temporal-weight", type=float, default=3.0)
    ap.add_argument("--run-name", default="unet_t3_k6")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    RUNS = ROOT / "runs" / args.run_name
    RUNS.mkdir(parents=True, exist_ok=True)
    dev = torch.device(args.device)
    print(f"[env] device={dev}  torch={torch.__version__}")

    cfg = LowLightConfig()
    net = UNetRecurrent().to(dev)
    print(f"[model] UNetRecurrent  params={count_params(net)/1e6:.2f}M  "
          f"BPTT-K={args.clip_len}  batch={args.batch}(={args.batch*args.clip_len}帧/iter)")

    ds_train = LowLightClipDataset(SPLIT.train, cfg, clip_len=args.clip_len,
                                   patch=args.patch, base_seed=1000)
    ds_val = LowLightClipDataset(SPLIT.val, cfg, clip_len=args.clip_len,
                                 patch=args.patch, base_seed=9000)
    dl_train = DataLoader(ds_train, batch_size=args.batch, shuffle=True,
                          num_workers=args.num_workers, drop_last=True, pin_memory=True)
    dl_val = DataLoader(ds_val, batch_size=args.batch, shuffle=False,
                        num_workers=args.num_workers, pin_memory=True)
    print(f"[data] train clips={len(ds_train)}  val clips={len(ds_val)}")

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
    print(f"[train] iters={args.iters}  lambda_temp={args.temporal_weight}")

    while it < args.iters:
        for batch in dl_train:
            if it >= args.iters:
                break
            ds_train.set_epoch(it // max(1, len(dl_train)))
            lo, gt, flow = batch
            lo = lo.to(dev, non_blocking=True)
            gt = gt.to(dev, non_blocking=True)
            flow = flow.to(dev, non_blocking=True)
            K = lo.shape[1]

            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=(dev.type == "cuda")):
                # BPTT-K：h 跨 K 帧累积，梯度沿全部 K-1 步回传
                h = None
                outs = []
                loss_enh = lo.new_zeros(())
                loss_temp = lo.new_zeros(())
                for k in range(K):
                    out_k, h = net.forward_state(lo[:, k], h)
                    outs.append(out_k)
                    loss_enh = loss_enh + enhancement_loss(out_k, gt[:, k])
                    if k > 0:
                        loss_temp = loss_temp + temporal_warp_loss(outs[k - 1], out_k, flow[:, k - 1])
                loss = loss_enh / K + args.temporal_weight * loss_temp / (K - 1)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            sched.step()

            if it % 25 == 0:
                dt = (time.perf_counter() - t_start) / max(1, it + 1)
                print(f"  it={it:5d}  loss={loss.item():.4f}  "
                      f"l_temp={loss_temp.item()/max(1,K-1):.5f}  "
                      f"lr={opt.param_groups[0]['lr']:.2e}  {dt:.2f}s/it", flush=True)

            if it % args.val_every == 0 or it == args.iters - 1:
                vp = quick_val(net, dl_val, dev)
                log.writerow([it, opt.param_groups[0]["lr"], float(loss.item()),
                              float(loss_temp.item()) / max(1, K - 1), vp,
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
    print(f"[done] best val PSNR={best_psnr:.2f}  → {RUNS/'best.pt'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
