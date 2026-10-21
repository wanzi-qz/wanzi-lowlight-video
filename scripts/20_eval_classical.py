"""M2 验收 · 经典 baseline 的参数扫描 + 测试集评估 + 三联图

============================================================
脚本做了什么（按顺序）
============================================================
1. 自检划分（不许重叠）
2. **在 val 上扫参**：每个方法 × 每个候选参数 → 算 PSNR/SSIM，选 PSNR 最好的那组
3. **只在 test 上跑一次** 选好的参数 → 完整指标（PSNR/SSIM/tOF/tOF_ratio）
4. 画三联对比图（低光输入 | 方法输出 | GT）+ 放大暗部
5. 出 `reports/m2_classical.{md,csv,json,png}`

============================================================
这次要验证的三个假设（写在前面，免得事后编故事）
============================================================
H1  gamma 提亮后 PSNR **可能反而下降**（噪声放大的代价 > 亮度修正的收益）
H2  CLAHE 的 SSIM 不一定最高 —— 局部对比度拉太猛会改变结构
H3  所有逐像素方法的 **tOF_ratio ≥ 0.8**（即几乎不抑制闪烁，甚至变差）
    → 这是全项目动机的量化证据：逐帧处理天然闪烁

用法：
    python scripts/20_eval_classical.py --seqs-per-split 2 --max-frames 8
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.baselines.classical import method_grid, make_method  # noqa: E402
from src.data import DAVIS_ROOT, SPLIT, list_frames  # noqa: E402
from src.metrics import (  # noqa: E402
    evaluate_sequence, lpips_metric, occlusion_mask, psnr, ssim,
)
from src.synthesis import (  # noqa: E402
    LowLightConfig, degrade_frame, frame_rng, linear_to_srgb, load_rgb, save_rgb,
    srgb_to_linear,
)

REPORTS = ROOT / "reports"


# --------------------------------------------------------------------------
def load_seq(seq: str, n_frames: int, seed: int, cfg: LowLightConfig):
    """返回 (gts, lows)。低光用固定 seed → 同一序列每次跑都是同一份噪声（可复现）。

    n_frames 取**均匀抽样的连续片段**：从序列里取一段连续的 n 帧，
    而不是全序列（全序列 70~100 帧 × 90 条太慢）。tOF 需要真正的连续帧，
    所以不能随机跳帧。
    """
    files = list_frames(DAVIS_ROOT, seq)
    if len(files) > n_frames:
        start = max(0, (len(files) - n_frames) // 2)  # 取中间段，避开片头片尾的静止/黑场
        files = files[start:start + n_frames]

    gts, lows = [], []
    for t, f in enumerate(files):
        gt = load_rgb(f)
        gts.append(gt)
        lows.append(degrade_frame(gt, cfg, frame_rng(seed, seq, t)))
    return gts, lows


def gt_flow_pairs(gts: list[np.ndarray]):
    """在 GT 上估计光流 + 遮挡掩码 —— 一次算好，所有方法复用（保证指标可比）。"""
    pairs = []
    for t in range(len(gts) - 1):
        ga = cv2.cvtColor(gts[t], cv2.COLOR_RGB2GRAY)
        gb = cv2.cvtColor(gts[t + 1], cv2.COLOR_RGB2GRAY)
        pairs.append(occlusion_mask(ga, gb))
    return pairs


def eval_one(spec, lows, gts, flows, lpips_fn=None):
    preds = [spec.fn(x) for x in lows]
    return evaluate_sequence(preds, gts, flows, input_frames=lows, lpips_fn=lpips_fn), preds


# --------------------------------------------------------------------------
def tune_on_val(cfg: LowLightConfig, max_frames: int, seed: int):
    """在 val 的**每一帧**上扫参，按平均 PSNR 选最优。"""
    data = {}
    for seq in SPLIT.val:
        data[seq] = load_seq(seq, max_frames, seed, cfg)

    results = []
    for mname, candidates in method_grid().items():
        for cand in candidates:
            spec = make_method(mname, **cand)
            ps = []
            ss = []
            t0 = time.perf_counter()
            for seq, (gts, lows) in data.items():
                for lo, gt in zip(lows, gts):
                    out = spec.fn(lo)
                    ps.append(psnr(out, gt))
                    ss.append(ssim(out, gt))
            dt = (time.perf_counter() - t0) / max(1, sum(len(v[0]) for v in data.values())) * 1000
            results.append({
                "method": mname,
                "params": json.dumps(cand),
                "label": spec.name,
                "val_psnr": float(np.mean(ps)),
                "val_ssim": float(np.mean(ss)),
                "ms_per_frame": dt,
            })
    return results


def pick_best(tuned: list[dict]) -> dict[str, dict]:
    best: dict[str, dict] = {}
    for r in tuned:
        m = r["method"]
        if m not in best or r["val_psnr"] > best[m]["val_psnr"]:
            best[m] = r
    return best


# --------------------------------------------------------------------------
def make_figure(seq: str, gts, lows, preds_by_name: dict[str, list[np.ndarray]], frame: int = 0):
    """三联图：上排全图（低光 | 各方法 | GT），下排对应暗部 2× 放大框。"""
    gt = gts[frame]
    panels = [("low-light input", lows[frame])] + [
        (n, p[frame]) for n, p in preds_by_name.items()
    ] + [("GT", gt)]

    # 暗部裁剪框：选低光图里最暗的 1/6 区域（最能看出噪声 vs 细节）
    h, w = gt.shape[:2]
    ch, cw = h // 3, w // 3
    lum = cv2.cvtColor(lows[frame], cv2.COLOR_RGB2GRAY).astype(np.float32)
    best = (1e9, 0, 0)
    for y in range(0, h - ch, ch // 2):
        for x in range(0, w - cw, cw // 2):
            m = float(lum[y:y + ch, x:x + cw].mean())
            if m < best[0]:
                best = (m, y, x)
    _, y0, x0 = best

    def label(img, text):
        out = img.copy()
        cv2.rectangle(out, (0, 0), (min(img.shape[1], len(text) * 10 + 14), 24), (0, 0, 0), -1)
        cv2.putText(out, text, (6, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        return out

    scale = 2
    top = np.hstack([label(im, n) for n, im in panels])
    bot = np.hstack([
        label(cv2.resize(im[y0:y0 + ch, x0:x0 + cw],
                         (cw * scale, ch * scale), interpolation=cv2.INTER_NEAREST), "")
        for _, im in panels
    ])
    # 下排宽度 = 上排，若不等则补齐
    if bot.shape[1] != top.shape[1]:
        bot = cv2.resize(bot, (top.shape[1], int(bot.shape[0] * top.shape[1] / bot.shape[1])))
    fig = np.vstack([top, bot])
    # 缩小到宽度 ≤ 1800，避免 png 太大
    if fig.shape[1] > 1800:
        s = 1800 / fig.shape[1]
        fig = cv2.resize(fig, (1800, int(fig.shape[0] * s)), interpolation=cv2.INTER_AREA)
    return fig


# --------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-frames", type=int, default=8, help="每条序列取多少连续帧")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--with-lpips", action="store_true", help="额外算 LPIPS（慢，需 lpips 包）")
    ap.add_argument("--fig-seq", default=None, help="画图的序列（默认 test 第一条）")
    args = ap.parse_args()

    cfg = LowLightConfig()
    REPORTS.mkdir(exist_ok=True)

    print(f"[cfg] EV={cfg.ev} Φ={cfg.full_well} s={cfg.brightness_keep} seed={args.seed}")
    print("[split]", "train=", len(SPLIT.train), "val=", SPLIT.val, "test=", SPLIT.test)

    # 1) val 扫参
    print("\n== step 1: 在 val 上扫参 ==")
    tuned = tune_on_val(cfg, args.max_frames, args.seed)
    best = pick_best(tuned)
    for m, b in best.items():
        print(f"  {b['label']:<34} val PSNR={b['val_psnr']:.2f} SSIM={b['val_ssim']:.4f}")

    # 2) test 评估（每个方法只跑一次）
    print("\n== step 2: test 评估（一次性，不回头调参） ==")
    lp = lpips_metric() if args.with_lpips else None
    if args.with_lpips and lp is None:
        print("  [warn] lpips 不可用，跳过")

    rows = []
    fig_cache = {}
    for seq in SPLIT.test:
        gts, lows = load_seq(seq, args.max_frames, args.seed, cfg)
        flows = gt_flow_pairs(gts)
        fig_cache[seq] = (gts, lows)

        # 参照：低光输入原样（等价于 identity）
        base = make_method("identity")
        m_in, _ = eval_one(base, lows, gts, flows, lp)
        m_in.update(method="low-light input", params="{}")
        rows.append({"seq": seq, **m_in, "ms_per_frame": 0.0})

        for mname, b in best.items():
            cand = json.loads(b["params"])
            spec = make_method(mname, **cand)
            t0 = time.perf_counter()
            m_out, preds = eval_one(spec, lows, gts, flows, lp)
            dt = (time.perf_counter() - t0) / len(lows) * 1000
            m_out.update(method=spec.name, params=b["params"], ms_per_frame=dt)
            rows.append({"seq": seq, **m_out})
            fig_cache[seq] = (gts, lows)
            fig_cache.setdefault("_preds", {}).setdefault(seq, {})[spec.name] = preds

        print(f"  {seq:<16} " + "  ".join(
            f"{r['method'].split('(')[0]}={r['psnr']:.2f}" for r in rows if r["seq"] == seq))

    # 3) 汇总表
    methods = ["low-light input"] + [b["label"] for b in best.values()]
    summary = []
    for m in methods:
        rs = [r for r in rows if r["method"] == m]
        if not rs:
            continue
        summary.append({
            "method": m,
            "psnr": float(np.mean([r["psnr"] for r in rs])),
            "ssim": float(np.mean([r["ssim"] for r in rs])),
            "tof": float(np.mean([r["tof"] for r in rs])),
            "tof_ratio": float(np.mean([r["tof_ratio"] for r in rs])),
            "ms_per_frame": float(np.mean([r["ms_per_frame"] for r in rs])),
            **({"lpips": float(np.mean([r["lpips"] for r in rs]))} if lp and "lpips" in rs[0] else {}),
        })

    # 4) 图
    fig_seq = args.fig_seq or SPLIT.test[0]
    preds_map = fig_cache.get("_preds", {}).get(fig_seq, {})
    fig = make_figure(fig_seq, fig_cache[fig_seq][0], fig_cache[fig_seq][1], preds_map)
    fig_path = REPORTS / f"m2_classical_{fig_seq}.png"
    save_rgb(fig_path, fig)

    # 5) 落盘
    with open(REPORTS / "m2_classical.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    with open(REPORTS / "m2_classical.json", "w", encoding="utf-8") as f:
        json.dump({"cfg": cfg.to_dict(), "seed": args.seed, "max_frames": args.max_frames,
                   "tuned_val": tuned, "best": best, "per_seq_test": rows,
                   "summary": summary}, f, ensure_ascii=False, indent=2)

    # 6) markdown 报告
    lines = [
        "# M2 验收报告 · 经典 baseline",
        "",
        f"- 低光配置：EV={cfg.ev}, Φ={cfg.full_well}e-, s={cfg.brightness_keep}, seed={args.seed}",
        f"- 每序列帧数：{args.max_frames}（连续片段，取序列中段）",
        f"- **参数只在 val 上选**（{', '.join(SPLIT.val)}），test 一次性使用：{', '.join(SPLIT.test)}",
        "",
        "## val 选参结果",
        "",
        "| 方法 | 最好参数 | val PSNR | val SSIM |",
        "|---|---|---|---|",
    ]
    for b in best.values():
        lines.append(f"| {b['label']} | `{b['params']}` | {b['val_psnr']:.2f} | {b['val_ssim']:.4f} |")

    lines += [
        "",
        "## test 结果（3 条序列 × " + str(args.max_frames) + " 帧）",
        "",
        "| 方法 | PSNR↑ | SSIM↑ | LPIPS↓ | tOF↓ | tOF/tOF_in ↓ | ms/帧 |",
        "|---|---|---|---|---|---|---|",
    ]
    for s in summary:
        lines.append(
            f"| {s['method']} | {s['psnr']:.2f} | {s['ssim']:.4f} | "
            f"{s.get('lpips', float('nan')):.4f} | {s['tof']:.3f} | {s['tof_ratio']:.3f} | "
            f"{s['ms_per_frame']:.0f} |"
        )

    # 7) 假设检验落表（用数字回答，不用形容词）
    def g(name):
        return next((s for s in summary if s["method"] == name), None)

    inp, gam, cla, ssr = g("low-light input"), None, None, None
    for s in summary:
        if s["method"].startswith("gamma"):
            gam = s
        elif s["method"].startswith("clahe"):
            cla = s
        elif s["method"].startswith("ssr"):
            ssr = s

    def verdict(cond: bool) -> str:
        return "**成立**" if cond else "**不成立**"

    lines += ["", "## 假设检验（开跑前写下的，事后照填）", ""]
    if gam:
        lines.append(
            f"- H1 「gamma 提亮后 PSNR 可能反而下降」：输入 {inp['psnr']:.2f}dB → "
            f"{gam['psnr']:.2f}dB，{verdict(gam['psnr'] <= inp['psnr'] + 0.05)}"
        )
    if cla and ssr and gam:
        best_ssim = max(summary, key=lambda s: s["ssim"])
        lines.append(
            f"- H2 「CLAHE 的 SSIM 不一定最高」：最高 SSIM 是 **{best_ssim['method']}**"
            f"（{best_ssim['ssim']:.4f}），CLAHE {cla['ssim']:.4f}，{verdict(best_ssim['method'].startswith('clahe') is False)}"
        )
    lines.append(
        "- H3 「所有逐像素方法 tOF_ratio ≥ 0.8」："
        + "，".join(f"{s['method']}={s['tof_ratio']:.3f}"
                   for s in summary if s["method"] != "low-light input")
        + f" → {verdict(all(s['tof_ratio'] >= 0.8 for s in summary if s['method'] != 'low-light input'))}"
    )

    lines += [
        "",
        "## 口径说明：本表的 PSNR 是「未做亮度对齐」的逐帧直接 PSNR",
        "",
        f"- 低光输入的亮度对齐 PSNR ≈ 25-26dB（见 M1 报告），但本表里只有 {inp['psnr']:.2f}dB ——",
        "  这 ~10dB 的差距来自**整体变暗**，不是信息丢失。",
        "- 因此「gamma 21dB」这类提升里，大部分是**把亮度拉回来**的功劳，去噪只占一小部分；",
        "  M3（学习式 baseline）里会同时报告两种口径，把「提亮」和「去噪」拆开归因。",
        "",
        f"图：`reports/{fig_path.name}`（上排：低光输入 → 各方法 → GT；下排：暗部 2× 放大）",
        "",
        '> 注意：tOF/tOF_in 是本项目定义的归一化闪烁度。**≥1 表示「比不处理还抖」**，',
        "> 因为它把噪声放大了。这个数字就是 M3/M4 要打败的对象。",
    ]
    (REPORTS / "m2_classical.md").write_text("\n".join(lines), encoding="utf-8")

    print("\n== summary (test) ==")
    for s in summary:
        print(f"  {s['method']:<34} PSNR={s['psnr']:.2f} SSIM={s['ssim']:.4f} "
              f"tOF={s['tof']:.3f} ratio={s['tof_ratio']:.3f}")
    print(f"\n[out] {REPORTS/'m2_classical.md'}")
    print(f"[out] {fig_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
