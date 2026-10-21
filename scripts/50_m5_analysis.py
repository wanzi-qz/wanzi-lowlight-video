"""M5 · 汇总分析：Wilcoxon 配对检验 / trade-off 散点 / 失败案例 / 算力表

============================================================
四个问题
============================================================
Q1 T1/T2 的收益在统计上显著吗？——逐帧 PSNR 配对 Wilcoxon 符号秩检验
   + bootstrap 95% CI。test 只有 3 序列（序列级 tOF 无法配对，n=3），
   逐帧配对（n=72）是唯一有统计功效的口径——声明：帧间相关会使
   有效样本量 < 72，检验偏乐观，结论只当辅助证据。
Q2 保真度-稳定性 trade-off 长什么样？——全方法 (PSNR, tOF_ratio) 散点
Q3 失败案例长什么样？——bmx-trees（快速运动）逐帧 PSNR 曲线
Q4 各方法算力多少？——单帧推理时间表

用法：
    python scripts/50_m5_analysis.py   # eval-only，几分钟
"""

from __future__ import annotations

import csv
import json
import sys
import time
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data import DAVIS_ROOT, SPLIT, list_frames  # noqa: E402
from src.metrics import (  # noqa: E402
    evaluate_sequence, occlusion_mask, psnr,
)
from src.models.unet import UNetEnhance  # noqa: E402
from src.synthesis import LowLightConfig, degrade_frame, frame_rng, load_rgb  # noqa: E402
from src.temporal.fuse import FuseParams, fuse_sequence  # noqa: E402

REPORTS = ROOT / "reports"
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TUNE = {"past": 2, "future": 2, "phot_sigma": 0.10, "fb_thresh": 1.0}
_tune = REPORTS / "m4_t1_tune.json"
if _tune.exists():
    TUNE.update({k: v for k, v in json.loads(_tune.read_text(encoding="utf-8"))["best"].items()
                 if k in TUNE})


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


def run_unet(net, lows, bks):
    net.eval()
    out = []
    with torch.no_grad():
        for lo in lows:
            x = torch.from_numpy(lo.astype(np.float32) / 255.0 / bks).permute(2, 0, 1)[None].to(DEV)
            y = net(x)
            out.append((y[0].clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).round().astype(np.uint8))
    return out


def gamma15(x):
    return (np.clip(x.astype(np.float32) / 255.0, 0, 1) ** (1 / 1.5) * 255).round().astype(np.uint8)


def bootstrap_ci(diff, n=10000, seed=0):
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(diff), size=(n, len(diff)))
    means = diff[idx].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def main() -> int:
    cfg = LowLightConfig()
    bks = cfg.brightness_keep

    ckpts = {
        "v0 (per-frame)": ROOT / "runs" / "unet_v0" / "best.pt",
        "T2 lam=0.3": ROOT / "runs" / "unet_t2" / "best.pt",
        "T2 lam=3": ROOT / "runs" / "unet_t2_lam3" / "best.pt",
        "T2 lam=10": ROOT / "runs" / "unet_t2_lam10" / "best.pt",
        "T2 lam=30": ROOT / "runs" / "unet_t2_lam30" / "best.pt",
    }
    nets = {}
    for name, p in ckpts.items():
        if not p.exists():
            print(f"[skip] {name}: {p.name} 不存在（λ 扫描可能未完成）")
            continue
        net = UNetEnhance().to(DEV)
        ck = torch.load(p, map_location=DEV, weights_only=False)
        if ck.get("iter", 0) < 500:
            print(f"[skip] {name}: iter={ck.get('iter')} 未训练完成（初始化快照，指标无效）")
            continue
        net.load_state_dict(ck["model"])
        nets[name] = net
        print(f"[ckpt] {name}  iter={ck.get('iter')}")

    # ---------------- 逐序列逐帧计算 ----------------
    per_frame: dict[str, list[float]] = {}
    scatter: list[dict] = []
    fuse_p = FuseParams(**TUNE)
    curves_bmx: dict[str, list[float]] = {}

    for seq in SPLIT.test:
        gts, lows = load_seq(seq, 24, 0, cfg)
        flows = [occlusion_mask(cv2.cvtColor(gts[t], cv2.COLOR_RGB2GRAY),
                                cv2.cvtColor(gts[t + 1], cv2.COLOR_RGB2GRAY))
                 for t in range(len(gts) - 1)]
        results: dict[str, list[np.ndarray]] = {"low-light input": list(lows)}
        results["gamma(1.5)"] = [gamma15(x) for x in lows]
        for name, net in nets.items():
            enh = run_unet(net, lows, bks)
            results[name] = enh
            if name == "v0 (per-frame)":
                results["+ T1 fused"] = fuse_sequence(enh, lows, fuse_p)
        for name, preds in results.items():
            m = evaluate_sequence(preds, gts, flows, input_frames=lows)
            scatter.append({"seq": seq, "method": name, "psnr": m["psnr"],
                            "tof_ratio": m["tof_ratio"]})
            for p, g in zip(preds, gts):
                per_frame.setdefault(name, []).append(psnr(p, g))
            if seq == "bmx-trees":
                curves_bmx[name] = [psnr(p, g) for p, g in zip(preds, gts)]
        print(f"[done] {seq}")

    # ---------------- Q1 Wilcoxon ----------------
    base = np.array(per_frame["v0 (per-frame)"])
    stat_rows = []
    for name in ["gamma(1.5)", "+ T1 fused", "T2 lam=3", "T2 lam=10", "T2 lam=30"]:
        if name not in per_frame:
            continue
        other = np.array(per_frame[name])
        diff = other - base
        w_stat, p_val = stats.wilcoxon(other, base)
        lo_ci, hi_ci = bootstrap_ci(diff)
        stat_rows.append({
            "vs": name, "median_diff": float(np.median(diff)),
            "mean_diff": float(diff.mean()), "ci95": (lo_ci, hi_ci),
            "win_rate": float((diff > 0).mean()), "W": float(w_stat), "p": float(p_val),
        })
        sig = "significant" if p_val < 0.05 else "n.s."
        print(f"  {name:>14} vs v0:  Δmean={diff.mean():+.2f}dB  "
              f"95%CI=[{lo_ci:+.2f},{hi_ci:+.2f}]  win={100*(diff>0).mean():.0f}%  "
              f"p={p_val:.2e} ({sig})")

    # ---------------- Q2 散点 ----------------
    agg: dict[str, dict[str, list[float]]] = {}
    for r in scatter:
        agg.setdefault(r["method"], {"psnr": [], "tof": []})
        agg[r["method"]]["psnr"].append(r["psnr"])
        agg[r["method"]]["tof"].append(r["tof_ratio"])

    fig, ax = plt.subplots(figsize=(7, 5))
    for name, v in agg.items():
        ax.scatter(np.mean(v["tof"]), np.mean(v["psnr"]), s=80)
        ax.annotate(name, (np.mean(v["tof"]), np.mean(v["psnr"])),
                    textcoords="offset points", xytext=(8, 4), fontsize=8)
    ax.set_xlabel("tOF / tOF_input  (lower = more stable)")
    ax.set_ylabel("PSNR dB (higher = better)")
    ax.set_title("M5 · fidelity-stability trade-off (test, 3 seq × 24 frames)")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(REPORTS / "m5_scatter.png", dpi=130)
    print(f"[out] {REPORTS/'m5_scatter.png'}")

    # ---------------- Q3 失败案例 ----------------
    fig, ax = plt.subplots(figsize=(8, 4))
    for name, c in curves_bmx.items():
        ax.plot(c, label=name, linewidth=1.2)
    ax.set_xlabel("frame")
    ax.set_ylabel("PSNR dB")
    ax.set_title("M5 · failure case: bmx-trees (fast motion) per-frame PSNR")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(REPORTS / "m5_failure_bmx.png", dpi=130)
    print(f"[out] {REPORTS/'m5_failure_bmx.png'}")

    # ---------------- Q4 算力表 ----------------
    timing = {}
    dummy = np.random.randint(0, 255, (540, 960, 3), np.uint8)
    x = torch.from_numpy(dummy.astype(np.float32) / 255.0 / bks).permute(2, 0, 1)[None].to(DEV)
    for name, net in nets.items():
        if name != "v0 (per-frame)":
            continue
        with torch.no_grad():
            for _ in range(5):
                net(x)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(20):
                net(x)
            torch.cuda.synchronize()
            timing[name] = (time.perf_counter() - t0) / 20 * 1000
    t_fuse0 = time.perf_counter()
    _ = fuse_sequence([dummy] * 10, [dummy] * 10, fuse_p)
    t_fuse = (time.perf_counter() - t_fuse0) / 10 * 1000
    for r in scatter:
        pass
    print(f"[time] UNet 960x540: {timing.get('v0 (per-frame)', float('nan')):.1f} ms/frame; "
          f"T1 fusion: ~{t_fuse:.0f} ms/frame (CPU 光流为主)")

    # ---------------- 落盘 ----------------
    with open(REPORTS / "m5_per_frame_psnr.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["method"] + list(range(72)))
        for name, v in per_frame.items():
            w.writerow([name] + [f"{x:.2f}" for x in v])
    with open(REPORTS / "m5_scatter.json", "w", encoding="utf-8") as f:
        json.dump({"scatter": scatter, "stats": [
            {**r, "ci95": list(r["ci95"])} for r in stat_rows]}, f,
            ensure_ascii=False, indent=2)

    L = ["# M5 · 汇总分析", "",
         "## Q1 逐帧 PSNR 配对检验（Wilcoxon 符号秩，n=72 帧）", "",
         "基线 = unet_v0（逐帧 U-Net）。声明：同一序列相邻帧相关，有效样本量 < 72，"
         "检验偏乐观——只作辅助证据，不当独立结论。", "",
         "| 方法 vs 基线 | Δmean dB | 95% CI (bootstrap) | 胜率 | p 值 | 显著? |",
         "|---|---|---|---|---|---|"]
    for r in stat_rows:
        lo, hi = r["ci95"]
        L.append(f"| {r['vs']} | {r['mean_diff']:+.3f} | [{lo:+.3f}, {hi:+.3f}] | "
                 f"{100*r['win_rate']:.0f}% | {r['p']:.2e} | {'是' if r['p']<0.05 else '否'} |")
    L += ["", "## Q4 算力（RTX 4060, 960×540 单帧）", "",
          f"- UNet 前向: **{timing.get('v0 (per-frame)', float('nan')):.1f} ms/帧**",
          f"- T1 融合（past={TUNE['past']}, future={TUNE['future']}）: ~{t_fuse:.0f} ms/帧"
          "（Farneback 在 CPU，占大头）",
          "- T3 循环模型: 前向开销与 UNet 同量级（GRU 在 1/8 分辨率瓶颈层）", "",
          "图：`m5_scatter.png`（trade-off 散点）、`m5_failure_bmx.png`（失败案例逐帧曲线）"]
    (REPORTS / "m5_analysis.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"[out] {REPORTS/'m5_analysis.md'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
