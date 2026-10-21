"""M4 · T1 光流 warp 融合（训练无关的时序稳定后处理）

============================================================
原理：为什么"加权平均 warp 过来的邻帧"能压闪烁？
============================================================
逐帧增强的输出 E_t 里，噪声是**每帧独立**的（M1-D 验证过，比值 1.004）。
把相邻帧 warp 对齐后做遮挡感知的加权平均：
  - 噪声是独立零均值的 → 平均后方差按 1/N 下降（N 帧融合 ≈ √N 倍降噪）
  - 信号（场景内容）是对齐的 → 平均不损伤细节
这正是"多帧降噪"的原理，手机夜景模式的底层就是这个。

============================================================
三个设计决定（每个都有反例）
============================================================
1) **光流必须从"推理时可得"的帧上估计 —— 用低光输入，不用 GT。**
   用 GT 流是 oracle 实验（可以跑，但要标注清楚，不能当正式数字）。
   低光帧噪声大，Farneback 的窗口平均能扛住中等噪声（winsize=15）。

2) **遮挡感知：前向-后向一致性掩码。**
   warp 的假设是"像素没动"。遮挡/露出的地方像素真的消失/出现了，
   强行平均会产生 ghosting。判定：|F_{t→k} + warp(F_{k→t}, F_{t→k})| < θ。

3) **光度置信度 + 锚定当前帧。**
   除遮挡外，再加 exp(-|E_t − warp(E_k)|/σ) 的光度权重：warp 错了的地方
   （运动估计不准、光照变化）平均进去会产生拖影，用光度差自动降权。
   E_t 自身权重恒为 1（锚），保证最坏情况退化为原始逐帧输出——
   **融合只会变稳，不会比不用更差**（这是设计的保底性质）。

============================================================
已知局限（写进报告）
============================================================
- 快速运动处光流不准 → 掩码应当挡住大部分，但漏网的会 ghosting
- 相邻帧增益/亮度若系统性漂移，光度权重会整体下降 → 退化成逐帧输出（安全）
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from src.metrics import estimate_flow, warp_with_flow, FlowParams


@dataclass(frozen=True)
class FuseParams:
    """past/future 各融合几帧。权重按几何衰减，当前帧恒为 1（锚）。

    第一版只融合过去 2 帧，tOF 只降 4.8%（验收 FAIL）——平均的独立噪声
    帧数太少。多帧降噪的理论是方差 ~ 1/N：想砍一半闪烁，N 得翻几倍。
    离线处理没有因果约束，未来帧一样能用（手机夜景也是这么干的）。
    """
    past: int = 4
    future: int = 4
    weight_decay: float = 0.55      # 每远一帧权重乘 0.55
    phot_sigma: float = 0.10        # 光度权重的温度（0-1 尺度）。越小越"保守"
    fb_thresh: float = 1.0          # 前向-后向一致性阈值（像素）
    flow_blur: float = 1.5          # 估流前对灰度图的高斯模糊 σ（抗低光噪声）


def _gray(rgb_u8: np.ndarray, blur_sigma: float) -> np.ndarray:
    g = cv2.cvtColor(rgb_u8, cv2.COLOR_RGB2GRAY).astype(np.float32)
    return cv2.GaussianBlur(g, (0, 0), blur_sigma) if blur_sigma > 0 else g


def _fuse_one(t: int, enh: list[np.ndarray], grays: list[np.ndarray],
              p: FuseParams, n: int, phot_sigma: float | None = None) -> np.ndarray:
    acc = enh[t].astype(np.float32)
    wsum = np.ones(acc.shape[:2], np.float32)  # 当前帧权重恒为 1（锚）
    ps = p.phot_sigma if phot_sigma is None else phot_sigma

    idxs = [t - i for i in range(1, min(p.past, t) + 1)]
    idxs += [t + i for i in range(1, min(p.future, n - 1 - t) + 1)]

    for i, k in enumerate(idxs):
        # 反向流：目标帧 t 的每个像素对应源帧 k 的哪里
        f_bw = estimate_flow(grays[t], grays[k])
        # 前向流（k→t）+ 前向-后向一致性 → 遮挡掩码
        f_fw = estimate_flow(grays[k], grays[t])
        fb = np.linalg.norm(f_bw + warp_with_flow(f_fw, f_bw), axis=2)
        valid = (fb < p.fb_thresh).astype(np.float32)

        warped = warp_with_flow(enh[k].astype(np.float32), f_bw)
        # 光度置信度：warp 后与当前帧差得越少越可信
        phot = np.exp(-np.mean(np.abs(enh[t].astype(np.float32) / 255.0
                                      - warped / 255.0), axis=2) / ps)

        w = (p.weight_decay ** (i + 1)) * valid * phot
        acc += w[..., None] * warped
        wsum += w
    return np.round(np.clip(acc / wsum[..., None], 0, 255)).astype(np.uint8)


def fuse_sequence(
    enh: list[np.ndarray],
    lows: list[np.ndarray],
    p: FuseParams = FuseParams(),
) -> list[np.ndarray]:
    """T1 主函数：光流在**低光输入**上估计（推理时可得，不用 GT）。

    Args:
        enh: 逐帧增强结果（如 U-Net 输出），uint8 RGB
        lows: 对应的低光输入帧
        p: 融合参数

    Returns:
        时序稳定后的序列（uint8 RGB，与输入等长）
    """
    assert len(enh) == len(lows)
    n = len(enh)
    if n < 2:
        return [e.copy() for e in enh]
    g = [_gray(lo, p.flow_blur) for lo in lows]
    return [_fuse_one(t, enh, g, p, n) for t in range(n)]


def fuse_sequence_oracle_flow(
    enh: list[np.ndarray],
    gts: list[np.ndarray],
    p: FuseParams = FuseParams(),
) -> list[np.ndarray]:
    """Oracle 对照：光流改在 GT 帧上估计。

    只用于诊断「T1 的收益有多少被低光噪声下的光流误差吃掉了」。
    正式对比表里必须标注 oracle，不能混进正常数字。
    """
    n = len(enh)
    if n < 2:
        return [e.copy() for e in enh]
    g = [_gray(x, p.flow_blur) for x in gts]
    return [_fuse_one(t, enh, g, p, n) for t in range(n)]
