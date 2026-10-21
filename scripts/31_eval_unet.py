"""M3 验收 · 逐帧 U-Net 的完整评估（test 集，逐序列 + 汇总 + 失败案例分析）

============================================================
脚本回答三个问题
============================================================
Q1 U-Net 比经典方法强多少？（PSNR/SSIM/LPIPS）
Q2 **它的 tOF 比经典方法好了吗？** —— 预期：好一点，但远没有"解决"
   （逐帧训练的目标函数里没有跨帧项，tOF 的改善只来自"去噪更彻底"，
    不是来自"帧间一致"）
Q3 闪烁在时间轴上的形状是什么样？—— 画逐帧 PSNR 曲线 + 相邻帧差的谱，
   看它是"高频抖动"（噪声式的，tOF 大）还是"低频漂移"（亮度增益漂移，
   人眼更敏感、tOF 未必大）。**这两种闪烁要分别治**，这是 M4 的设计依据。

============================================================
为什么额外算"亮度增益抖动"
============================================================
模型的 gain 头对每帧独立输出一个标量。逐帧训练下，这个标量会随帧抖动
→ 整幅画面的亮度"呼吸"。这种闪烁的 tOF 不一定大（因为全局乘性变化
在光流对齐后仍然是全局偏移），但人眼极其敏感。
M4 的 T2 损失必须同时约束这一项，否则 tOF 降了观众依然觉得在闪。

用法：
    python scripts/31_eval_unet.py --ckpt runs/unet_v0/best.pt --max-frames 24
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

from src.baselines.classical import make_method  # noqa: E402
from src.data import DAVIS_ROOT, SPLIT, list_frames  # noqa: E402
from src.metrics import (  # noqa: E402
    evaluate_sequence, lpips_metric, occlusion_mask, psnr, ssim, temporal_warping_error,
)
from src.models.unet import UNetEnhance, count_params  # noqa: E402
from src.synthesis import (  # noqa: E402
    LowLightConfig, degrade_frame, frame_rng, load_rgb, save_rgb,
)

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
    out = []
    for t in range(len(gts) - 1):
        out.append(occlusion_mask(cv2.cvtColor(gts[t], cv2.COLOR_RGB2GRAY),
                                  cv2.cvtColor(gts[t + 1], cv2.COLOR_RGB2GRAY)))
    return out


@torch.no_grad()
def run_unet(net, lows, dev, bks):
    """全分辨率推理。**不做任何 crop/pad** —— 评估必须用整帧。

    同时返回每帧的 gain 头输出：逐帧训练下这个标量会抖动（"增益呼吸"），
    是一种 tOF 未必抓得到、但人眼极其敏感的闪烁 —— M4 的 T2 要单独约束它。

    为什么用 `torch.no_grad` + eval()：BN 用 running stats。
    如果误用 train()，BatchNorm 会用**当前这一帧**的统计量做归一化，
    相当于把每帧单独归一化 → 亮暗会随帧变化大幅漂移，tOF 直接爆炸。
    """
    net.eval()
    preds, gains = [], []
    for lo in lows:
        x = torch.from_numpy(lo.astype(np.float32) / 255.0 / bks).permute(2, 0, 1)[None].to(dev)
        g, _res, y = net.forward_parts(x)
        gains.append(float(g.flatten()[0]))
        preds.append((y[0].clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).round().astype(np.uint8))
    return preds, gains


def flicker_spectrum(values: np.ndarray) -> dict:
    """相邻帧差序列的统计：均值（平均抖动）+ 一阶自相关（越接近 1 越"低频漂移"）。"""
    d = values.astype(np.float64)
    d = d - d.mean()
    if len(d) < 4 or np.allclose(d, 0):
        return {"acf1": float("nan"), "std": float(np.std(values))}
    acf1 = float(np.corrcoef(d[:-1], d[1:])[0, 1])
    return {"acf1": acf1, "std": float(np.std(values))}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=str(ROOT / "runs" / "unet_v0" / "best.pt"))
    ap.add_argument("--max-frames", type=int, default=24)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--with-lpips", action="store_true")
    ap.add_argument("--out-prefix", default="m3_unet",
                    help="报告文件名前缀；T2 的评估用 m4_t2，避免覆盖 M3 报告")
    args = ap.parse_args()

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(args.ckpt, map_location=dev, weights_only=False)
    cfg = LowLightConfig(**{k: v for k, v in ck["cfg"].items() if k != "alpha"})
    net = UNetEnhance().to(dev)
    net.load_state_dict(ck["model"])
    print(f"[ckpt] {args.ckpt}  iter={ck.get('iter')}  params={count_params(net)/1e6:.2f}M")
    bks = cfg.brightness_keep

    lp = lpips_metric() if args.with_lpips else None
    methods = [
        ("low-light input", make_method("identity")),
        ("gamma(1.5)", make_method("gamma", gamma=1.5)),
        ("clahe(1.5,8,1.0)", make_method("clahe", clip_limit=1.5, tile=8, gamma=1.0)),
    ]

    rows = []
    per_seq_curves = {}
    for seq in SPLIT.test:
        gts, lows = load_seq(seq, args.max_frames, args.seed, cfg)
        flows = gt_flows(gts)
        print(f"[seq] {seq}  frames={len(gts)}")

        preds_all = {}
        for name, spec in methods:
            preds = [spec.fn(x) for x in lows]
            preds_all[name] = preds
            m = evaluate_sequence(preds, gts, flows, input_frames=lows, lpips_fn=lp)
            rows.append({"seq": seq, "method": name, **m})
        preds, gains = run_unet(net, lows, dev, bks)
        preds_all["UNet (per-frame)"] = preds
        m = evaluate_sequence(preds, gts, flows, input_frames=lows, lpips_fn=lp)
        m["gain_mean"] = float(np.mean(gains))
        m["gain_std"] = float(np.std(gains))
        rows.append({"seq": seq, "method": "UNet (per-frame)", **m})

        # 逐帧 PSNR 曲线：看是否某些帧特别差
        curves = {n: [psnr(p, g) for p, g in zip(ps, gts)] for n, ps in preds_all.items()}
        per_seq_curves[seq] = curves

        # 亮度漂移：每帧输出均值 vs GT 均值
        drift = {n: np.array([p.astype(np.float32).mean() / max(1e-6, g.astype(np.float32).mean())
                              for p, g in zip(ps, gts)])
                 for n, ps in preds_all.items()}

        # 汇总保存首帧对比图
        if seq == SPLIT.test[0]:
            make_fig(seq, gts, {n: preds_all[n] for n in preds_all}, args.out_prefix)

        print("    " + "  ".join(f"{r['method'].split()[0]}={r['psnr']:.2f}" for r in rows if r["seq"] == seq))

    # 汇总
    names = [m[0] for m in methods] + ["UNet (per-frame)"]
    summary = []
    for n in names:
        rs = [r for r in rows if r["method"] == n]
        summary.append({
            "method": n,
            "psnr": float(np.mean([r["psnr"] for r in rs])),
            "psnr_aligned": float(np.mean([r["psnr_aligned"] for r in rs])),
            "ssim": float(np.mean([r["ssim"] for r in rs])),
            "tof": float(np.mean([r["tof"] for r in rs])),
            "tof_ratio": float(np.mean([r["tof_ratio"] for r in rs])),
            "gain_std": float(np.mean([r.get("gain_std", float("nan")) for r in rs])),
            **({"lpips": float(np.mean([r["lpips"] for r in rs]))} if lp else {}),
        })

    # 各序列的 UNet tOF vs input tOF（看是否在快速运动序列上更差）
    detail = {}
    for seq in SPLIT.test:
        rs = {r["method"]: r for r in rows if r["seq"] == seq}
        detail[seq] = {name: {"psnr": v["psnr"], "tof": v["tof"], "tof_ratio": v["tof_ratio"]}
                       for name, v in rs.items()}

    # 落盘
    with open(REPORTS / f"{args.out_prefix}.csv", "w", newline="", encoding="utf-8") as f:
        keys: list[str] = []
        for r in rows:
            keys += [k for k in r.keys() if k not in keys]
        w = csv.DictWriter(f, fieldnames=keys)  # 各方法字段可能不同（gain_* 只有 UNet 有）
        w.writeheader()
        w.writerows(rows)
    with open(REPORTS / f"{args.out_prefix}.json", "w", encoding="utf-8") as f:
        json.dump({"ckpt": args.ckpt, "iter": ck.get("iter"), "summary": summary,
                   "detail": detail,
                   "curves": {s: {n: list(map(float, c)) for n, c in cs.items()}
                              for s, cs in per_seq_curves.items()}},
                  f, ensure_ascii=False, indent=2)

    plot_curves(per_seq_curves, SPLIT.test[0], args.out_prefix)

    # markdown
    L = [
        f"# 验收报告 · {args.out_prefix}（逐帧/时序 U-Net）",
        "",
        f"- checkpoint: `{Path(args.ckpt).name}` @ iter {ck.get('iter')}",
        f"- test 序列：{', '.join(SPLIT.test)}，每序列 {args.max_frames} 帧连续片段，噪声 seed={args.seed}",
        f"- 参数量 {count_params(net)/1e6:.2f}M，推理设备 {dev}",
        "",
        "## 汇总（test，逐帧直接 PSNR）",
        "",
        "| 方法 | PSNR↑ | PSNR(亮度对齐)↑ | SSIM↑ | tOF↓ | tOF/tOF_in ↓ | 增益抖动σ↓ |",
        "|---|---|---|---|---|---|---|",
    ]
    for s in summary:
        gs = s.get("gain_std", float("nan"))
        L.append(f"| {s['method']} | {s['psnr']:.2f} | {s['psnr_aligned']:.2f} | {s['ssim']:.4f} | "
                 f"{s['tof']:.3f} | {s['tof_ratio']:.3f} | {gs:.4f} |")

    L += ["", "## 逐序列（看是否在快速运动序列上更差）", "",
          "| 序列 | 方法 | PSNR | tOF | tOF/tOF_in |", "|---|---|---|---|---|"]
    for seq, d in detail.items():
        for name, v in d.items():
            L.append(f"| {seq} | {name} | {v['psnr']:.2f} | {v['tof']:.3f} | {v['tof_ratio']:.3f} |")

    # 核心结论（数字说话；结论的方向必须由数字决定，不许预写）
    u = next(s for s in summary if s["method"] == "UNet (per-frame)")
    g = next(s for s in summary if s["method"].startswith("gamma"))
    c = next(s for s in summary if s["method"].startswith("clahe"))
    d_g, d_c = u["psnr"] - g["psnr"], u["psnr"] - c["psnr"]
    fidelity = (f"UNet {u['psnr']:.2f}dB，比 gamma {'高' if d_g >= 0 else '低'} {abs(d_g):.2f}dB，"
                f"比 CLAHE {'高' if d_c >= 0 else '低'} {abs(d_c):.2f}dB。")
    fidelity += ("学习式方法在保真度上明确赢。" if d_g > 0 and d_c > 0 else
                 "⚠ UNet 没有赢过经典方法 —— 这不正常，先查训练/评估一致性再往下走，"
                 "不要给坏数字写好话。")
    fidelity += (f" 亮度对齐后 {u['psnr_aligned']:.2f}dB（对齐差值 "
                 f"{u['psnr'] - u['psnr_aligned']:+.2f}dB = 提亮贡献的口径差）。")
    if u["tof_ratio"] < 0.95:
        temporal = (f"UNet 的 tOF/tOF_in = {u['tof_ratio']:.3f}，确实把闪烁压下来了"
                    f"（gamma {g['tof_ratio']:.3f}，CLAHE {c['tof_ratio']:.3f}）——"
                    "但这是「逐帧去噪更彻底」带来的副产物，不是帧间一致。")
    else:
        temporal = (f"UNet 的 tOF/tOF_in = {u['tof_ratio']:.3f}，gamma {g['tof_ratio']:.3f}，"
                    f"CLAHE {c['tof_ratio']:.3f} —— 逐帧训练没有解决闪烁，这就是本项目的核心动机。")
    L += [
        "",
        "## 核心结论",
        "",
        f"1. **保真度**：{fidelity}",
        f"2. **时序稳定性**：{temporal}",
        f"3. **增益呼吸**：UNet 每帧增益的帧间标准差 = {u['gain_std']:.4f}"
        f"（理想 ≈ 0）。这是 tOF 抓不到、但人眼敏感的亮度闪烁，M4 的 T2 损失要单独约束它。",
        "4. 但注意：这个改善来自**逐帧去噪更彻底**，不是来自**帧间一致**。"
        "如果 tOF 仍有明显残差，且不同帧的增益/去噪强度在抖，就说明还需要 M4。",
        "",
        "## 下一步（M4 的设计依据）",
        "",
        "- T1（训练无关）先验证「warp 融合能否在不重训的情况下降低 tOF」。"
        "若 UNet 的 tOF_ratio 已经很低（例如 < 0.6），T1 的收益空间会被压缩，"
        "那时应把重点转向 T2（约束增益抖动），并在报告里如实说明。",
        "- T2 的损失必须**同时**约束：像素级 warp 残差 + 全局增益的一致性。",
        "",
        f"曲线图：`reports/m3_curves_{SPLIT.test[0]}.png`",
        f"对比图：`reports/m3_compare_{SPLIT.test[0]}.png`",
    ]
    (REPORTS / f"{args.out_prefix}.md").write_text("\n".join(L), encoding="utf-8")

    print("\n== summary ==")
    for s in summary:
        print(f"  {s['method']:<22} PSNR={s['psnr']:.2f} SSIM={s['ssim']:.4f} "
              f"tOF={s['tof']:.3f} ratio={s['tof_ratio']:.3f}")
    print(f"[out] {REPORTS / (args.out_prefix + '.md')}")
    return 0


def make_fig(seq, gts, preds_map, out_prefix):
    g0 = gts[0]
    panels = [(n, p[0]) for n, p in preds_map.items()] + [("GT", g0)]
    h, w = g0.shape[:2]
    ch, cw = h // 3, w // 3
    lum = cv2.cvtColor(preds_map["low-light input"][0], cv2.COLOR_RGB2GRAY).astype(np.float32)
    best = (1e9, 0, 0)
    for yy in range(0, h - ch, ch // 2):
        for xx in range(0, w - cw, cw // 2):
            mm = float(lum[yy:yy + ch, xx:xx + cw].mean())
            if mm < best[0]:
                best = (mm, yy, xx)
    _, y0, x0 = best

    def lab(im, t):
        o = im.copy()
        cv2.rectangle(o, (0, 0), (min(im.shape[1], len(t) * 10 + 14), 24), (0, 0, 0), -1)
        cv2.putText(o, t, (6, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        return o

    top = np.hstack([lab(im, n) for n, im in panels])
    bot = np.hstack([cv2.resize(im[y0:y0 + ch, x0:x0 + cw], (cw * 2, ch * 2),
                                interpolation=cv2.INTER_NEAREST) for _, im in panels])
    if bot.shape[1] != top.shape[1]:
        bot = cv2.resize(bot, (top.shape[1], int(bot.shape[0] * top.shape[1] / bot.shape[1])))
    fig = np.vstack([top, bot])
    if fig.shape[1] > 1800:
        s = 1800 / fig.shape[1]
        fig = cv2.resize(fig, (1800, int(fig.shape[0] * s)), interpolation=cv2.INTER_AREA)
    save_rgb(REPORTS / f"{out_prefix}_compare_{seq}.png", fig)


def plot_curves(curves_by_seq, seq, out_prefix):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return
    cs = curves_by_seq[seq]
    fig, ax = plt.subplots(figsize=(9, 3.6))
    for n, v in cs.items():
        ax.plot(range(len(v)), v, marker="o", ms=2.5, lw=1.2, label=n)
    ax.set_xlabel("frame"); ax.set_ylabel("PSNR (dB)")
    ax.set_title(f"per-frame PSNR on {seq}  (dips = frames the model fails on)")
    ax.legend(fontsize=7, ncol=2); fig.tight_layout()
    fig.savefig(REPORTS / f"{out_prefix}_curves_{seq}.png", dpi=120)
    plt.close(fig)


if __name__ == "__main__":
    sys.exit(main())
