"""M4-T3 验收 · 循环 U-Net 评估（stateful vs per-frame 重置对照）

============================================================
对照设计（结论合法性的底线）
============================================================
- **T3 stateful**：整条序列顺序推理，h 跨帧传递 —— T3 的真实形态
- **T3 per-frame**：每帧 h=None —— 等价于逐帧模型（+GRU 参数量）
  如果 stateful 不明显好于 per-frame，收益就不来自"循环记忆"本身。

验收线（与 T1/T2 同一口径，先于实验写下）：
  [PASS] stateful 的 tOF_ratio 相对 per-frame 重置下降 ≥10%
  [PASS] stateful 的 PSNR 下降 ≤0.2dB（时序收益不能靠牺牲保真度买）
  [PASS] stateful 的 PSNR 相对逐帧基线 unet_v0 不更差

用法：
    python scripts/34_eval_t3.py --ckpt runs/unet_t3/best.pt
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data import DAVIS_ROOT, SPLIT, list_frames  # noqa: E402
from src.metrics import (  # noqa: E402
    evaluate_sequence, lpips_metric, occlusion_mask, psnr,
)
from src.models.unet import UNetEnhance  # noqa: E402
from src.models.unet_gru import UNetRecurrent, count_params  # noqa: E402
from src.synthesis import LowLightConfig, degrade_frame, frame_rng, load_rgb  # noqa: E402

REPORTS = ROOT / "reports"


def load_seq(seq: str, n_frames: int, seed: int, cfg: LowLightConfig):
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


def gt_flows(gts):
    return [occlusion_mask(cv2.cvtColor(gts[t], cv2.COLOR_RGB2GRAY),
                           cv2.cvtColor(gts[t + 1], cv2.COLOR_RGB2GRAY))
            for t in range(len(gts) - 1)]


def to_u8(y: torch.Tensor) -> np.ndarray:
    return (y[0].clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).round().astype(np.uint8)


@torch.no_grad()
def run_t3(net, lows, dev, bks, stateful: bool):
    """stateful=True：h 跨帧传递；False：每帧重置（=逐帧对照）。"""
    net.eval()
    preds, h = [], None
    for lo in lows:
        x = torch.from_numpy(lo.astype(np.float32) / 255.0 / bks).permute(2, 0, 1)[None].to(dev)
        y, h_new = net.forward_state(x, h if stateful else None)
        preds.append(to_u8(y))
        h = h_new
    return preds


@torch.no_grad()
def run_unet(net, lows, dev, bks):
    net.eval()
    preds = []
    for lo in lows:
        x = torch.from_numpy(lo.astype(np.float32) / 255.0 / bks).permute(2, 0, 1)[None].to(dev)
        preds.append(to_u8(net(x)))
    return preds


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=str(ROOT / "runs" / "unet_t3" / "best.pt"))
    ap.add_argument("--perframe-ckpt", default=str(ROOT / "runs" / "unet_v0" / "best.pt"))
    ap.add_argument("--max-frames", type=int, default=24)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--with-lpips", action="store_true")
    ap.add_argument("--out-prefix", default="m4_t3")
    args = ap.parse_args()

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(args.ckpt, map_location=dev, weights_only=False)
    cfg = LowLightConfig(**{k: v for k, v in ck["cfg"].items() if k != "alpha"})
    bks = cfg.brightness_keep
    net = UNetRecurrent().to(dev)
    net.load_state_dict(ck["model"])
    print(f"[ckpt] {args.ckpt}  iter={ck.get('iter')}  params={count_params(net)/1e6:.2f}M")

    net_b0 = UNetEnhance().to(dev)
    ck0 = torch.load(args.perframe_ckpt, map_location=dev, weights_only=False)
    net_b0.load_state_dict(ck0["model"])

    lp = lpips_metric() if args.with_lpips else None

    rows = []
    for seq in SPLIT.test:
        gts, lows = load_seq(seq, args.max_frames, args.seed, cfg)
        flows = gt_flows(gts)
        print(f"[seq] {seq}  frames={len(gts)}")

        for name, preds in [
            ("low-light input", lows),
            ("T3 stateful", run_t3(net, lows, dev, bks, stateful=True)),
            ("T3 per-frame reset", run_t3(net, lows, dev, bks, stateful=False)),
            ("U-Net per-frame (unet_v0)", run_unet(net_b0, lows, dev, bks)),
        ]:
            m = evaluate_sequence(preds, gts, flows, input_frames=lows, lpips_fn=lp)
            rows.append({"seq": seq, "method": name, **m})
        for r in rows[-4:]:
            print(f"    {r['method']:<28} PSNR={r['psnr']:.2f}  tOF={r['tof']:.3f}  "
                  f"ratio={r['tof_ratio']:.3f}")

    names = ["low-light input", "T3 stateful", "T3 per-frame reset", "U-Net per-frame (unet_v0)"]
    summary = []
    for n in names:
        rs = [r for r in rows if r["method"] == n]
        summary.append({
            "method": n,
            "psnr": float(np.mean([r["psnr"] for r in rs])),
            "ssim": float(np.mean([r["ssim"] for r in rs])),
            "tof": float(np.mean([r["tof"] for r in rs])),
            "tof_ratio": float(np.mean([r["tof_ratio"] for r in rs])),
        })

    s = {m["method"]: m for m in summary}
    t3_sf, t3_pf, b0 = s["T3 stateful"], s["T3 per-frame reset"], s["U-Net per-frame (unet_v0)"]
    rec_drop = (t3_pf["tof_ratio"] - t3_sf["tof_ratio"]) / t3_pf["tof_ratio"]
    checks = [
        (rec_drop >= 0.10, f"stateful 相对 per-frame 重置 tOF 下降 ≥10%（实际 {rec_drop:+.1%}）"),
        (t3_sf["psnr"] >= t3_pf["psnr"] - 0.2, "stateful PSNR 下降 ≤0.2dB"),
        (t3_sf["psnr"] >= b0["psnr"] - 1e-9, "stateful PSNR 不差于逐帧基线 unet_v0"),
    ]

    with open(REPORTS / f"{args.out_prefix}.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    with open(REPORTS / f"{args.out_prefix}.json", "w", encoding="utf-8") as f:
        json.dump({"ckpt": args.ckpt, "summary": summary,
                   "recurrent_gain": rec_drop}, f, ensure_ascii=False, indent=2)

    L = [f"# 验收报告 · {args.out_prefix}（T3 循环 U-Net）", "",
         f"- checkpoint: `{Path(args.ckpt).name}` @ iter {ck.get('iter')}，"
         f"参数 {count_params(net)/1e6:.2f}M（逐帧基线 1.95M）",
         f"- test：{', '.join(SPLIT.test)}，各 {args.max_frames} 帧，噪声 seed={args.seed}", "",
         "| 方法 | PSNR↑ | SSIM↑ | tOF↓ | tOF/tOF_in ↓ |", "|---|---|---|---|---|"]
    for m in summary:
        L.append(f"| {m['method']} | {m['psnr']:.2f} | {m['ssim']:.4f} | "
                 f"{m['tof']:.3f} | {m['tof_ratio']:.3f} |")
    L += ["", "## 验收核对", ""]
    for ok, msg in checks:
        L.append(f"- [{'PASS' if ok else 'FAIL'}] {msg}")
    (REPORTS / f"{args.out_prefix}.md").write_text("\n".join(L) + "\n", encoding="utf-8")

    print("\n== summary ==")
    for m in summary:
        print(f"  {m['method']:<28} PSNR={m['psnr']:.2f} SSIM={m['ssim']:.4f} "
              f"tOF={m['tof']:.3f} ratio={m['tof_ratio']:.3f}")
    print(f"  循环收益（stateful vs per-frame 重置）: tOF {rec_drop:+.1%}")
    for ok, msg in checks:
        print(f"  [{'PASS' if ok else 'FAIL'}] {msg}")
    print(f"[out] {REPORTS / (args.out_prefix + '.md')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
