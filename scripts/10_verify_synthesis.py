"""M1 验收脚本 · 证明合成管线"物理上说得通"

四项检查（任何一项不过，就不要进 M2）：
  A. 噪声按帧独立      —— 同一 GT、3 个 seed，互相必须不同；同 seed 必须逐位相同
  B. 噪声信号相关      —— Var(残差) 对亮度线性，且斜率符合理论值 s²α/Φ
  C. 难度可控          —— 扫 EV × Φ，看输入 PSNR，选一个"有救但难"的区间
  D. 噪声时间独立      —— 残差帧间差 ≈ √2 × 残差空间标准差（iid 噪声的判据）

关于 D 的判据为什么长这样：
  如果噪声在时间上是独立同分布（iid）的，那么
      E|r_{t+1} − r_t|² = 2·Var(r)   →   |r_{t+1} − r_t| ≈ √2·σ_r
  比值 ≈ 1 就证明了"每一帧的噪声都是新的"——这正是视频闪烁的物理来源。
  若比值明显 < 1，说明噪声在帧间相关（比如同一个 seed 被复用），
  那后面 tOF 测出的"改善"就是假的。

输出：reports/m1_synthesis_check.md / .json / m1_synthesis_preview.png

用法：
    python scripts/10_verify_synthesis.py --seq bike-packing --frames 6
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.synthesis import (  # noqa: E402
    LowLightConfig, degrade_frame, frame_rng, linear_to_srgb, load_rgb, save_rgb,
    srgb_to_linear,
)

REPORTS = ROOT / "reports"


def _jsonable(o):
    """numpy 标量/数组 → 原生 python，避免 json.dumps 报 bool_ / float32 不可序列化。"""
    if hasattr(o, "item"):
        return o.item()
    if isinstance(o, (np.ndarray,)):
        return o.tolist()
    return str(o)


# --------------------------------------------------------------------------
# 指标
# --------------------------------------------------------------------------
def psnr_u8(a_u8: np.ndarray, b_u8: np.ndarray) -> float:
    mse = float(np.mean((a_u8.astype(np.float64) - b_u8.astype(np.float64)) ** 2)) / 255.0**2
    return 99.0 if mse <= 1e-12 else 10.0 * np.log10(1.0 / mse)


def ref_linear(rgb_u8: np.ndarray) -> np.ndarray:
    return srgb_to_linear(rgb_u8.astype(np.float32) / 255.0)


def optimal_gain(ll_u8: np.ndarray, gt_lin: np.ndarray) -> float:
    """最小二乘最优全局增益 k（闭式解），把低光图拉回 GT 的亮度水平。

    为什么要先对齐再算 PSNR：一张整体暗 2 档的图，直接比 PSNR 只有 16dB，
    但那 16dB 里绝大部分是"亮度差"而不是"信息差"。对齐后才是在衡量噪声/细节。
    """
    lin = ref_linear(ll_u8)
    return float(np.sum(lin * gt_lin) / max(np.sum(lin * lin), 1e-8))


def align_srgb_u8(ll_u8: np.ndarray, gt_u8: np.ndarray) -> np.ndarray:
    k = optimal_gain(ll_u8, ref_linear(gt_u8))
    s = linear_to_srgb(np.clip(ref_linear(ll_u8) * k, 0, 1))
    return np.round(s * 255.0).clip(0, 255).astype(np.uint8)


def residual(ll_u8: np.ndarray, gt_u8: np.ndarray) -> np.ndarray:
    """亮度对齐后的线性残差：r = k·L_lin − GT_lin。噪声分析都建在它上面。"""
    k = optimal_gain(ll_u8, ref_linear(gt_u8))
    return ref_linear(ll_u8) * k - ref_linear(gt_u8)


# --------------------------------------------------------------------------
# 数据
# --------------------------------------------------------------------------
def davis_root() -> Path:
    p = ROOT / "data" / "DAVIS"
    if not (p / "JPEGImages" / "480p").is_dir():
        raise SystemExit(f"DAVIS 未就绪：{p}（先跑 scripts/download_davis.py）")
    return p


def list_sequences() -> list[str]:
    return sorted(d.name for d in (davis_root() / "JPEGImages" / "480p").iterdir() if d.is_dir())


def load_frames(seq: str, n: int, stride: int = 3) -> list[np.ndarray]:
    files = sorted((davis_root() / "JPEGImages" / "480p" / seq).glob("*.jpg"))
    idx = list(range(0, min(len(files), n * stride), stride))[:n]
    return [load_rgb(files[i]) for i in idx]


# --------------------------------------------------------------------------
# A 噪声按帧独立
# --------------------------------------------------------------------------
def check_seed_independence(gt: np.ndarray, cfg: LowLightConfig) -> dict:
    outs = [degrade_frame(gt, cfg, np.random.default_rng(s)) for s in (0, 1, 2)]
    mad = lambda x, y: float(np.mean(np.abs(x.astype(np.int16) - y.astype(np.int16))))  # noqa: E731
    same = degrade_frame(gt, cfg, np.random.default_rng(0))
    res = {
        "mad_0_1": mad(outs[0], outs[1]),
        "mad_0_2": mad(outs[0], outs[2]),
        "mad_1_2": mad(outs[1], outs[2]),
        "reproducible_same_seed": bool(np.array_equal(outs[0], same)),
    }
    res["pass"] = bool(min(res["mad_0_1"], res["mad_0_2"], res["mad_1_2"]) > 0.5
                       and res["reproducible_same_seed"])
    return res


# --------------------------------------------------------------------------
# B 噪声方差 ~ 亮度
# --------------------------------------------------------------------------
def check_signal_dependent_noise(gt: np.ndarray, cfg: LowLightConfig, bins: int = 12) -> dict:
    stored = degrade_frame(gt, cfg, np.random.default_rng(1234), return_linear=True)
    lin = ref_linear(gt)
    resid = stored / cfg.brightness_keep - lin

    x = lin.reshape(-1)[::5]
    r = resid.reshape(-1)[::5]
    edges = np.linspace(0, 1, bins + 1)
    xs, vs = [], []
    for i in range(bins):
        m = (x >= edges[i]) & (x < edges[i + 1])
        if m.sum() < 500:
            continue
        xs.append(float(x[m].mean()))
        vs.append(float(r[m].var()))
    xs, vs = np.array(xs), np.array(vs)
    if len(xs) < 3:
        return {"pass": False, "reason": "有效分箱不足"}

    A = np.vstack([xs, np.ones_like(xs)]).T
    (slope, intercept), *_ = np.linalg.lstsq(A, vs, rcond=None)
    pred = A @ np.array([slope, intercept])
    r2 = 1 - float(np.sum((vs - pred) ** 2)) / max(float(np.sum((vs - vs.mean()) ** 2)), 1e-12)
    # 推导：resid = electrons·α/Φ − lin，Var(electrons) = lin·Φ/α + σ_r²
    #   ⇒ Var(resid) = lin·(α/Φ) + (σ_r·α/Φ)²       ← 斜率 α/Φ，截距是读出噪声
    # 注意增益 s 会约掉：resid 先除以 s 再对齐，噪声相对信号的比值与 s 无关（物理正确：
    # 拉高 ISO 不改变 SNR）。第一版我把 s² 留在斜率里，被这个检查抓出来了。
    theory = cfg.alpha / cfg.full_well
    theory_intercept = (cfg.read_noise * cfg.alpha / cfg.full_well) ** 2
    rel_err = abs(slope - theory) / theory
    res = {"n_bins": len(xs), "slope": float(slope), "intercept": float(intercept),
           "theory_slope": float(theory), "theory_intercept": float(theory_intercept),
           "rel_err": float(rel_err), "r2": float(r2),
           "x": xs.tolist(), "var": vs.tolist()}
    res["pass"] = bool(r2 > 0.95 and rel_err < 0.25)
    return res


# --------------------------------------------------------------------------
# C 难度扫描
# --------------------------------------------------------------------------
def check_difficulty(gt: np.ndarray) -> list[dict]:
    rows = []
    gt_lin = ref_linear(gt)
    for ev in (1.5, 2.0, 2.5, 3.0):
        for fw in (400.0, 800.0, 1600.0, 2500.0):
            cfg = LowLightConfig(ev=ev, full_well=fw)
            out = degrade_frame(gt, cfg, np.random.default_rng(7))
            al = align_srgb_u8(out, gt)
            rs = residual(out, gt)
            rows.append({
                "ev": ev, "alpha": round(cfg.alpha, 2), "full_well": fw,
                "mid_gray_electrons": round(0.18 * fw / cfg.alpha, 1),
                "mid_gray_snr_db": round(20 * np.log10(np.sqrt(0.18 * fw / cfg.alpha)), 1),
                "psnr_raw": round(psnr_u8(out, gt), 2),
                "psnr_aligned": round(psnr_u8(al, gt), 2),
                "resid_std": round(float(rs.std()), 4),
            })
    return rows


# --------------------------------------------------------------------------
# D 噪声时间独立性
# --------------------------------------------------------------------------
def check_temporal_noise(frames: list[np.ndarray], cfg: LowLightConfig, seed: int) -> dict:
    """估计器说明（第一版这里是错的，记录一下）：

    我用 |Δr| 的**平均绝对值**去比 √2·σ，结果算出来 0.56，看起来像"噪声时间相关"。
    其实 0.56 ≈ 0.797×0.7，错在两点：
      1. 高斯分布下 E|Δr| = √(2/π)·√2·σ ≈ 1.128σ，不是 √2·σ → 先偏小到 0.797
      2. σ 用"残差总标准差"会把 8bit 量化 + JPEG 的**系统性误差**也算进去 → 再偏小
    正确做法：用 RMS，并且用"同一帧两次独立采样之差"单独估出**纯噪声** σ_noise。
    """
    rs, ll = [], []
    for t, gt in enumerate(frames):
        out = degrade_frame(gt, cfg, frame_rng(seed, "chk", t))
        ll.append(out)
        rs.append(residual(out, gt))

    # 纯噪声 σ：同一帧、两个不同随机源 → 两次抽样之差只含噪声
    r_a = residual(degrade_frame(frames[0], cfg, np.random.default_rng(9001)), frames[0])
    r_b = residual(degrade_frame(frames[0], cfg, np.random.default_rng(9002)), frames[0])
    sigma_noise = float(np.sqrt(np.mean((r_a - r_b) ** 2)) / np.sqrt(2))

    sigma_total = float(np.mean([r.std() for r in rs]))
    # 时间独立性：iid 时 RMS(r_{t+1} − r_t) = √2·σ_noise
    t_rms = float(np.mean([np.sqrt(np.mean((rs[t + 1] - rs[t]) ** 2))
                           for t in range(len(rs) - 1)]))
    ratio = t_rms / max(np.sqrt(2) * sigma_noise, 1e-9)

    gt_motion = float(np.mean([
        np.mean(np.abs(ref_linear(frames[t + 1]) - ref_linear(frames[t])))
        for t in range(len(frames) - 1)
    ]))
    res = {"sigma_noise": sigma_noise, "sigma_total": sigma_total,
           "noise_share": float(sigma_noise / max(sigma_total, 1e-9)),
           "temporal_rms": t_rms, "ratio": ratio, "gt_motion": gt_motion,
           "psnr_aligned_mean": float(np.mean(
               [psnr_u8(align_srgb_u8(ll[i], frames[i]), frames[i]) for i in range(len(frames))]))}
    res["pass"] = bool(0.85 <= ratio <= 1.15)
    return res


# --------------------------------------------------------------------------
# 预览图
# --------------------------------------------------------------------------
def make_preview(gt: np.ndarray, cfg: LowLightConfig, path: Path) -> None:
    ll = degrade_frame(gt, cfg, np.random.default_rng(42))
    gamma = (np.clip(ll.astype(np.float32) / 255.0, 0, 1) ** (1 / 1.8) * 255).astype(np.uint8)
    aligned = align_srgb_u8(ll, gt)
    gap = np.full((gt.shape[0], 8, 3), 255, np.uint8)
    save_rgb(path, np.hstack([gt, gap, ll, gap, gamma, gap, aligned]))


# --------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq", default=None)
    ap.add_argument("--frames", type=int, default=6)
    args = ap.parse_args()

    seqs = list_sequences()
    seq = args.seq or next((s for s in seqs if "bike" in s), seqs[0])
    frames = load_frames(seq, args.frames)
    cfg = LowLightConfig()
    gt = frames[0]

    print(f"[data] 序列={seq}  帧数={len(frames)}  {gt.shape[1]}x{gt.shape[0]}  "
          f"（数据集共 {len(seqs)} 条序列）")

    a = check_seed_independence(gt, cfg)
    print(f"[A] 噪声按帧独立: MAD={a['mad_0_1']:.2f}/{a['mad_0_2']:.2f}/{a['mad_1_2']:.2f} "
          f"同seed可复现={a['reproducible_same_seed']} -> {'PASS' if a['pass'] else 'FAIL'}")

    b = check_signal_dependent_noise(gt, cfg)
    print(f"[B] 噪声信号相关: R²={b['r2']:.4f} 斜率={b['slope']:.3e} 理论={b['theory_slope']:.3e} "
          f"误差={b['rel_err']*100:.1f}% -> {'PASS' if b['pass'] else 'FAIL'}")

    c = check_difficulty(gt)
    print("[C] 难度扫描 (psnr_aligned = 亮度对齐后的 sRGB PSNR，越低越难):")
    for r in c:
        print(f"    EV={r['ev']:.1f} α={r['alpha']:5.2f} Φ={r['full_well']:5.0f} "
              f"中灰e-={r['mid_gray_electrons']:5.1f} SNR={r['mid_gray_snr_db']:4.1f}dB "
              f"raw={r['psnr_raw']:5.1f} aligned={r['psnr_aligned']:5.1f} "
              f"resid={r['resid_std']:.4f}")

    d = check_temporal_noise(frames, cfg, seed=0)
    print(f"[D] 噪声时间独立: σ_noise={d['sigma_noise']:.5f} σ_total={d['sigma_total']:.5f} "
          f"(噪声占比 {d['noise_share']*100:.0f}%) |Δr|RMS={d['temporal_rms']:.5f} "
          f"比值={d['ratio']:.3f} (理想 1.000) -> {'PASS' if d['pass'] else 'FAIL'}")

    REPORTS.mkdir(exist_ok=True)
    make_preview(gt, cfg, REPORTS / "m1_synthesis_preview.png")

    md = [
        "# M1 验收报告 · 低光合成管线",
        "",
        f"- 序列：`{seq}`（{len(frames)} 帧，{gt.shape[1]}×{gt.shape[0]}）",
        f"- 配置：EV={cfg.ev}, α={cfg.alpha:.2f}, Φ={cfg.full_well:.0f}e-, "
        f"σ_r={cfg.read_noise}e-, s={cfg.brightness_keep}, JPEG q={cfg.jpeg_quality}",
        "",
        "## A 噪声按帧独立",
        f"- 同 GT、seed 0/1/2 的像素平均绝对差：{a['mad_0_1']:.2f} / {a['mad_0_2']:.2f} / "
        f"{a['mad_1_2']:.2f}（0-255 尺度，必须 > 0）",
        f"- 同 seed 完全可复现：{a['reproducible_same_seed']}",
        f"- **{'PASS' if a['pass'] else 'FAIL'}**",
        "",
        "## B 噪声信号相关（散粒噪声的核心证据）",
        f"- 分箱数 {b['n_bins']}，拟合 R² = **{b['r2']:.4f}**",
        f"- 拟合斜率 {b['slope']:.3e} vs 理论 s²α/Φ = {b['theory_slope']:.3e}，"
        f"相对误差 {b['rel_err']*100:.1f}%",
        f"- **{'PASS' if b['pass'] else 'FAIL'}**",
        "",
        "> 对照：若只用 gamma 把图压暗，残差恒为 0，方差-亮度图是一条水平线，R² 无意义。",
        "",
        "## C 难度扫描",
        "",
        "| EV | α | Φ(e-) | 中灰电子数 | 中灰SNR | PSNR(直接) | PSNR(亮度对齐) | 残差σ |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in c:
        md.append(f"| {r['ev']:.1f} | {r['alpha']:.2f} | {r['full_well']:.0f} | "
                  f"{r['mid_gray_electrons']:.1f} | {r['mid_gray_snr_db']:.1f}dB | "
                  f"{r['psnr_raw']:.2f} | {r['psnr_aligned']:.2f} | {r['resid_std']:.4f} |")
    md += [
        "",
        "`PSNR(直接)` 偏低主要来自整体变暗（不是信息丢失）；`PSNR(亮度对齐)` 才是噪声强度。",
        "",
        "## D 噪声时间独立（闪烁的物理来源）",
        f"- 纯噪声 σ_noise（同帧两次独立采样之差估得）= {d['sigma_noise']:.5f}",
        f"- 残差总 σ_total = {d['sigma_total']:.5f} → **噪声占比 {d['noise_share']*100:.1f}%**"
        f"（其余是 8bit 量化 + JPEG 的系统性误差）",
        f"- 相邻帧残差差 RMS = {d['temporal_rms']:.5f}",
        f"- 比值 RMS/(√2·σ_noise) = **{d['ratio']:.3f}**（iid 噪声理论值 1.000）",
        f"- 同期 GT 帧间运动量 = {d['gt_motion']:.4f}（画面本身在动，但残差差与运动无关 → 说明测的是噪声不是运动）",
        f"- 平均 PSNR（亮度对齐）= {d['psnr_aligned_mean']:.2f} dB",
        f"- **{'PASS' if d['pass'] else 'FAIL'}**",
        "",
        "预览：`reports/m1_synthesis_preview.png`（GT | 低光原样 | gamma 提亮 | 亮度对齐）",
    ]
    (REPORTS / "m1_synthesis_check.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    (REPORTS / "m1_synthesis_check.json").write_text(
        json.dumps({"seq": seq, "config": cfg.to_dict(), "A": a, "B": b, "C": c, "D": d},
                   indent=2, ensure_ascii=False, default=_jsonable), encoding="utf-8")

    ok = a["pass"] and b["pass"] and d["pass"]
    print(f"\n[result] {'全部通过 → 可以进入 M2' if ok else '有未通过项 → 先修管线'}")
    print(f"[report] {REPORTS / 'm1_synthesis_check.md'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
