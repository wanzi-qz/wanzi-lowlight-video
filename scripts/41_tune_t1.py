"""M4 · T1 调参 —— **只在 val 序列上扫**，test 一次性使用（与 M2 同一纪律）

扫什么：
- past/future 帧数（平均的独立噪声帧数 N 是 tOF 下降的根本来源）
- phot_sigma（光度权重的保守程度：太小 → 几乎不融合；太大 → ghosting）
- fb_thresh（遮挡掩码松紧）

不扫 flow_blur / weight_decay（影响小，固定省时间 —— 网格成本是乘法关系）。

输出：reports/m4_t1_tune.md + json（含 val 最优配置）

用法：
    python scripts/41_tune_t1.py --max-frames 16
"""

from __future__ import annotations

import argparse
import json
import sys
from itertools import product
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data import DAVIS_ROOT, SPLIT, list_frames  # noqa: E402
from src.metrics import temporal_warping_error, psnr, occlusion_mask  # noqa: E402
from src.models.unet import UNetEnhance  # noqa: E402
from src.synthesis import LowLightConfig, degrade_frame, frame_rng, load_rgb  # noqa: E402
from src.temporal.fuse import FuseParams, fuse_sequence  # noqa: E402

REPORTS = ROOT / "reports"


def load_seq(seq, n_frames, seed, cfg):
    files = list_frames(DAVIS_ROOT, seq)
    if len(files) > n_frames:
        start = max(0, (len(files) - n_frames) // 2)
        files = files[start:start + n_frames]
    gts, lows = [], []
    for t, f in enumerate(files):
        gt = load_rgb(f)
        gts.append(gt)
        lows.append(degrade_frame(gt, cfg, frame_rng(seed, seq, t)))
    return gts, lows


@torch.no_grad()
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=str(ROOT / "runs" / "unet_v0" / "best.pt"))
    ap.add_argument("--max-frames", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(args.ckpt, map_location=dev, weights_only=False)
    cfg = LowLightConfig(**{k: v for k, v in ck["cfg"].items() if k != "alpha"})
    net = UNetEnhance().to(dev)
    net.load_state_dict(ck["model"])
    bks = cfg.brightness_keep

    seqs = SPLIT.val
    print(f"[val] {seqs} × {args.max_frames} 帧", flush=True)

    data = {}
    for seq in seqs:
        gts, lows = load_seq(seq, args.max_frames, args.seed, cfg)
        flows = [occlusion_mask(cv2.cvtColor(gts[t], cv2.COLOR_RGB2GRAY),
                                cv2.cvtColor(gts[t + 1], cv2.COLOR_RGB2GRAY))
                 for t in range(len(gts) - 1)]
        enh = []
        for lo in lows:
            x = torch.from_numpy(lo.astype(np.float32) / 255.0 / bks).permute(2, 0, 1)[None].to(dev)
            y = net(x)
            enh.append((y[0].clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).round().astype(np.uint8))
        base_tof = temporal_warping_error(enh, flows)
        base_psnr = float(np.mean([psnr(p, g) for p, g in zip(enh, gts)]))
        data[seq] = {"gts": gts, "lows": lows, "flows": flows, "enh": enh,
                     "base_tof": base_tof, "base_psnr": base_psnr}
        print(f"  {seq}: UNet tOF={base_tof:.3f} PSNR={base_psnr:.2f}", flush=True)

    grid = list(product((1, 2, 4, 6), (1, 2, 4, 6), (0.05, 0.15, 0.35), (0.5, 1.5)))
    print(f"[grid] {len(grid)} 组（past, future, phot_sigma, fb_thresh）", flush=True)

    rows = []
    for past, future, ps, fb in grid:
        p = FuseParams(past=past, future=future, phot_sigma=ps, fb_thresh=fb)
        tofs, psnrs = [], []
        for seq, d in data.items():
            fused = fuse_sequence(d["enh"], d["lows"], p)
            tofs.append(temporal_warping_error(fused, d["flows"]))
            psnrs.append(float(np.mean([psnr(a, g) for a, g in zip(fused, d["gts"])])))
        r = {"past": past, "future": future, "phot_sigma": ps, "fb_thresh": fb,
             "tof": float(np.mean(tofs)), "psnr": float(np.mean(psnrs)),
             "base_tof": float(np.mean([d["base_tof"] for d in data.values()])),
             "base_psnr": float(np.mean([d["base_psnr"] for d in data.values()]))}
        r["tof_drop_pct"] = (r["base_tof"] - r["tof"]) / r["base_tof"] * 100
        r["d_psnr"] = r["psnr"] - r["base_psnr"]
        rows.append(r)
        print(f"  past={past} future={future} σ={ps:.2f} fb={fb:.1f} "
              f"→ tOF {r['base_tof']:.2f}→{r['tof']:.2f} ({r['tof_drop_pct']:+.1f}%) "
              f"PSNR {r['d_psnr']:+.2f}dB", flush=True)

    # 选参标准：在 PSNR 代价 ≤0.2dB 的配置里选 tOF 最低（保真度优先，这是我们给
    # T1 定的验收线；tOF 与 PSNR 的完整 trade-off 留给 M5 的 λ 扫描）
    ok = [r for r in rows if r["d_psnr"] >= -0.2]
    best = min(ok, key=lambda r: r["tof"]) if ok else min(rows, key=lambda r: r["tof"])

    REPORTS.mkdir(exist_ok=True)
    (REPORTS / "m4_t1_tune.json").write_text(json.dumps(
        {"rows": rows, "best": best, "val_seqs": list(seqs)},
        ensure_ascii=False, indent=2), encoding="utf-8")

    md = [
        "# M4 · T1 调参（val 集）",
        "",
        f"- val 序列：{', '.join(seqs)} × {args.max_frames} 帧，噪声 seed={args.seed}",
        f"- 基线（逐帧 U-Net）：tOF={best['base_tof']:.3f}  PSNR={best['base_psnr']:.2f}",
        "",
        "| past | future | phot_sigma | fb_thresh | tOF↓ | tOF降幅 | ΔPSNR |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in sorted(rows, key=lambda r: r["tof"])[:15]:
        md.append(f"| {r['past']} | {r['future']} | {r['phot_sigma']:.2f} | {r['fb_thresh']:.1f} "
                  f"| {r['tof']:.3f} | {r['tof_drop_pct']:+.1f}% | {r['d_psnr']:+.2f}dB |")
    md += [
        "",
        f"**val 最优**（PSNR 代价 ≤0.2dB 约束下 tOF 最低）："
        f"past={best['past']}, future={best['future']}, "
        f"phot_sigma={best['phot_sigma']}, fb_thresh={best['fb_thresh']} → "
        f"tOF {best['tof']:.3f}（降 {best['tof_drop_pct']:.1f}%），ΔPSNR {best['d_psnr']:+.2f}dB",
        "",
        "> 表格只列 tOF 最低的前 15 行，全量见 m4_t1_tune.json。test 数字由 40_eval_t1.py 一次性产生。",
    ]
    (REPORTS / "m4_t1_tune.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"\n[best] past={best['past']} future={best['future']} phot_sigma={best['phot_sigma']} "
          f"fb_thresh={best['fb_thresh']}  tOF→{best['tof']:.3f} ({best['tof_drop_pct']:+.1f}%)")
    print(f"[out] {REPORTS / 'm4_t1_tune.md'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
