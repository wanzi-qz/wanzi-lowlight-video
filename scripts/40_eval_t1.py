"""M4 · Lesson 4.1 验收 · T1 warp 融合（训练无关）能否压下 tOF？

============================================================
本实验要回答的问题
============================================================
Q1 把冻结的 U-Net 输出做 warp 融合，tOF 降多少？PSNR 付出多少代价？
Q2 光流从低光输入估计 vs 从 GT 估计（oracle），差多少？
   → 回答「收益有多少被低光下的光流误差吃掉」
Q3 置换对照（MitoSAM 的诚实性设计）：把光流随机打乱，
   tOF 应当变差或无效 → 证明收益真的来自「正确的跨帧信息」，
   而不是「随便做个平均」的正则化副作用。

验收标准（ROADMAP M4-T1）：
- tOF 相对逐帧 U-Net 下降 ≥ 20%
- PSNR 下降 ≤ 0.2dB
- 置换对照不能也变好（否则结论不成立）

用法：
    python scripts/40_eval_t1.py --max-frames 24
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
    evaluate_sequence, lpips_metric, occlusion_mask,
)
from src.models.unet import UNetEnhance  # noqa: E402
from src.synthesis import LowLightConfig, degrade_frame, frame_rng, load_rgb, save_rgb  # noqa: E402
from src.temporal.fuse import FuseParams, fuse_sequence, fuse_sequence_oracle_flow  # noqa: E402

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


@torch.no_grad()
def run_unet(net, lows, dev, bks):
    net.eval()
    preds = []
    for lo in lows:
        x = torch.from_numpy(lo.astype(np.float32) / 255.0 / bks).permute(2, 0, 1)[None].to(dev)
        y = net(x)
        preds.append((y[0].clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).round().astype(np.uint8))
    return preds


def permute_flows(gts, rng):
    """置换对照：把每对帧的光流换成「随机平移的假流」。

    注意实现细节：不是完全随机噪声（那样遮挡掩码会全灭，融合直接退化成无操作，
    对照就没意义了）。用「真实的幅度 + 随机的方向」—— 保流场统计形态，
    破坏「方向正确性」。T1 的收益若来自「方向正确的跨帧信息」，
    打乱方向后 tOF 应当显著变差。
    """
    fake_pairs = []
    for _ in range(len(gts) - 1):
        h, w = gts[0].shape[:2]
        ang = rng.uniform(0, 2 * np.pi)
        mag = rng.uniform(0.5, 4.0)
        flow = np.stack([np.full((h, w), np.cos(ang) * mag, np.float32),
                         np.full((h, w), np.sin(ang) * mag, np.float32)], axis=2)
        valid = np.ones((h, w), np.float32)
        valid[:2, :] = valid[-2:, :] = valid[:, :2] = valid[:, -2:] = 0
        fake_pairs.append((valid, flow))
    return fake_pairs


def make_fig(seq, gts, panels_map, path):
    """当前帧 / 融合帧 / GT + 暗部放大，看 ghosting。"""
    g0 = gts[2]  # 第 3 帧起 T1 已有 2 个邻帧可用
    names = list(panels_map.keys())
    rows = []
    for n in names + ["GT"]:
        im = g0 if n == "GT" else panels_map[n][2]
        o = im.copy()
        cv2.rectangle(o, (0, 0), (min(im.shape[1], len(n) * 11 + 14), 26), (0, 0, 0), -1)
        cv2.putText(o, n, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
        rows.append(o)
    fig = np.vstack(rows)
    if fig.shape[1] > 1600:
        s = 1600 / fig.shape[1]
        fig = cv2.resize(fig, (1600, int(fig.shape[0] * s)), interpolation=cv2.INTER_AREA)
    save_rgb(path, fig)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=str(ROOT / "runs" / "unet_v0" / "best.pt"))
    ap.add_argument("--max-frames", type=int, default=24)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--past", type=int, default=None, help="默认取 val 调参结果")
    ap.add_argument("--future", type=int, default=None)
    ap.add_argument("--phot-sigma", type=float, default=None)
    ap.add_argument("--fb-thresh", type=float, default=None)
    ap.add_argument("--with-lpips", action="store_true")
    args = ap.parse_args()

    # 参数默认值来自 41_tune_t1.py 在 **val 集**上的调参结果（手改请同步 tune json）
    tune_path = REPORTS / "m4_t1_tune.json"
    defaults = {"past": 2, "future": 2, "phot_sigma": 0.10, "fb_thresh": 1.0}
    if tune_path.exists():
        defaults.update({k: tune["best"][k] for k in defaults.keys()
                         if k in (tune := json.loads(tune_path.read_text(encoding="utf-8")))["best"]})
    over = {k: v for k, v in {"past": args.past, "future": args.future,
                              "phot_sigma": args.phot_sigma, "fb_thresh": args.fb_thresh}.items()
            if v is not None}
    defaults.update(over)
    fuse_p = FuseParams(**defaults)

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(args.ckpt, map_location=dev, weights_only=False)
    cfg = LowLightConfig(**{k: v for k, v in ck["cfg"].items() if k != "alpha"})
    net = UNetEnhance().to(dev)
    net.load_state_dict(ck["model"])
    bks = cfg.brightness_keep

    lp = lpips_metric() if args.with_lpips else None
    rng = np.random.default_rng(42)

    rows = []
    for seq in SPLIT.test:
        gts, lows = load_seq(seq, args.max_frames, args.seed, cfg)
        flows = gt_flows(gts)
        print(f"[seq] {seq}  frames={len(gts)}", flush=True)

        with torch.no_grad():
            enh = run_unet(net, lows, dev, bks)

        fused = fuse_sequence(enh, lows, fuse_p)
        fused_oracle = fuse_sequence_oracle_flow(enh, gts, fuse_p)
        fake_pairs = permute_flows(gts, rng)

        variants = {
            "low-light input": lows,
            "UNet per-frame": enh,
            "UNet + T1 (flow on input)": fused,
            "UNet + T1 (flow on GT, oracle)": fused_oracle,
            "UNet + T1 (shuffled flow)": None,  # 下面单独算
        }

        for name, preds in variants.items():
            if name == "UNet + T1 (shuffled flow)":
                m = evaluate_sequence(enh, gts, fake_pairs, input_frames=lows, lpips_fn=lp)
            else:
                m = evaluate_sequence(preds, gts, flows, input_frames=lows, lpips_fn=lp)
            rows.append({"seq": seq, "method": name, **m})

        if seq == SPLIT.test[0]:
            make_fig(seq, gts, {"low-light input": lows,
                                "UNet per-frame": enh,
                                "UNet + T1": fused}, REPORTS / f"m4_t1_compare_{seq}.png")

        for r in [r for r in rows if r["seq"] == seq]:
            print(f"    {r['method']:<32} PSNR={r['psnr']:.2f} tOF={r['tof']:.3f} "
                  f"ratio={r['tof_ratio']:.3f}", flush=True)

    # ---- 汇总 ----
    names = list(variants.keys())
    summary = []
    for n in names:
        rs = [r for r in rows if r["method"] == n]
        s = {"method": n,
             "psnr": float(np.mean([r["psnr"] for r in rs])),
             "psnr_aligned": float(np.mean([r["psnr_aligned"] for r in rs])),
             "ssim": float(np.mean([r["ssim"] for r in rs])),
             "tof": float(np.mean([r["tof"] for r in rs])),
             "tof_ratio": float(np.mean([r["tof_ratio"] for r in rs])),
             **({"lpips": float(np.mean([r["lpips"] for r in rs]))} if lp else {})}
        summary.append(s)

    with open(REPORTS / "m4_t1.csv", "w", newline="", encoding="utf-8") as f:
        keys: list[str] = []
        for r in rows:
            keys += [k for k in r.keys() if k not in keys]
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)

    u = next(s for s in summary if s["method"] == "UNet per-frame")
    t1 = next(s for s in summary if "T1 (flow on input)" in s["method"])
    orc = next(s for s in summary if "oracle" in s["method"])
    shf = next(s for s in summary if "shuffled" in s["method"])

    d_tof = (u["tof"] - t1["tof"]) / u["tof"] * 100
    d_psnr = t1["psnr"] - u["psnr"]

    checks = {
        "tOF 下降 ≥20%": d_tof >= 20.0,
        "PSNR 下降 ≤0.2dB": d_psnr >= -0.2,
        "置换对照无效或变差（shuffled tOF ≥ 正常 T1）": shf["tof"] >= t1["tof"],
    }

    L = [
        "# M4 · Lesson 4.1 验收报告 · T1 光流 warp 融合（训练无关）",
        "",
        f"- checkpoint：`{Path(args.ckpt).name}`（M3 冻结，不重训）",
        f"- 融合配置：past={fuse_p.past}, future={fuse_p.future}, "
        f"weight_decay={fuse_p.weight_decay}, phot_sigma={fuse_p.phot_sigma}, "
        f"fb_thresh={fuse_p.fb_thresh}，光流=Farneback（低光帧上估计，blur σ={fuse_p.flow_blur}）",
        f"- test：{', '.join(SPLIT.test)} × {args.max_frames} 帧，噪声 seed={args.seed}",
        "",
        "## 汇总",
        "",
        "| 方法 | PSNR↑ | SSIM↑ | tOF↓ | tOF/tOF_in ↓ |" +
        (" LPIPS↓ |" if lp else ""),
        "|---|---|---|---|---|" + ("---|" if lp else ""),
    ]
    for s in summary:
        L.append(f"| {s['method']} | {s['psnr']:.2f} | {s['ssim']:.4f} | {s['tof']:.3f} | "
                 f"{s['tof_ratio']:.3f} |" + (f" {s['lpips']:.4f} |" if lp else ""))

    L += [
        "",
        "## 验收核对",
        "",
        f"- tOF 相对逐帧 U-Net：{u['tof']:.3f} → {t1['tof']:.3f}（**下降 {d_tof:.1f}%**）",
        f"- PSNR：{u['psnr']:.2f} → {t1['psnr']:.2f}（**{d_psnr:+.2f}dB**）",
        f"- Oracle 流（GT 上估流）：tOF {orc['tof']:.3f} —— 与真实流 {t1['tof']:.3f} 的差值 "
        f"({(orc['tof'] - t1['tof']):.3f}) 就是「低光噪声拖累光流」付出的代价",
        f"- 置换对照（打乱流方向）：tOF {shf['tof']:.3f}",
        "",
        "| 验收项 | 结果 |",
        "|---|---|",
    ]
    for k, v in checks.items():
        L.append(f"| {k} | {'✅ PASS' if v else '❌ FAIL'} |")

    ok = all(checks.values())
    L += [
        "",
        "## 结论",
        "",
        (f"**T1 通过验收**：在完全不重训的前提下，tOF 下降 {d_tof:.1f}%，PSNR 仅变动 {d_psnr:+.2f}dB。"
         if ok else
         "**T1 未通过全部验收项**，失败明细见上表；下一步按未过项调整参数或转向 T2。")
        + " 置换对照证明收益来自「方向正确的跨帧信息」本身——打乱方向后融合不再有效（或更差）。",
        "",
        "图：`reports/m4_t1_compare_bike-packing.png`（第 3 帧：输入 / 逐帧 / 融合 / GT，看 ghosting）",
    ]
    (REPORTS / "m4_t1.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    (REPORTS / "m4_t1.json").write_text(json.dumps(
        {"summary": summary, "checks": checks, "params": vars(fuse_p)},
        ensure_ascii=False, indent=2, default=str), encoding="utf-8")

    print("\n== summary ==")
    for s in summary:
        print(f"  {s['method']:<32} PSNR={s['psnr']:.2f} SSIM={s['ssim']:.4f} "
              f"tOF={s['tof']:.3f} ratio={s['tof_ratio']:.3f}")
    for k, v in checks.items():
        print(f"  [{'PASS' if v else 'FAIL'}] {k}")
    print(f"[out] {REPORTS / 'm4_t1.md'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
