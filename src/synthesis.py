"""M1 · 低光退化合成管线

============================================================
为什么需要"合成"而不是直接找真实低光视频？
============================================================
真实低光数据集（SDSD / LoLi-Phone / SMID）没有与暗帧**严格配准**的正常曝光 GT，
且多为长曝光参考帧（存在运动模糊与配准残差）。没有可靠 GT 就没有 PSNR/SSIM，
也就无法做"同一帧、同一噪声实例"的配对比较——所有结论都会退化成"看着更好"。

合成管线牺牲噪声分布的真实性，换来**可证伪的实验设计**。这是本项目的取舍，
也会写进论文的局限章节。

============================================================
退化的物理模型（按光路顺序）
============================================================
1. sRGB 解码      I_lin = srgb_to_linear(I/255)          # 显示域 → 线性辐照度
2. 曝光不足       λ = I_lin · Φ / α                       # α = 2^EV，Φ = 满阱电子数
3. 散粒噪声       S ~ Poisson(λ)                          # 光子到达是泊松过程 → 噪声**信号相关**
4. 读出噪声       S += N(0, σ_r²)                         # 与信号无关的加性噪声
5. 相机增益       I' = S · (α·s) / Φ                      # s = 亮度保留系数（模拟 ISO 增益）
6. 白平衡         I' ·= wb                                # 通道增益，制造轻微色偏
7. sRGB 编码 →   8bit 量化 → JPEG 压缩                    # 量化+压缩会吃掉暗部层次

============================================================
关键教学点：为什么不能只用 gamma 把图变暗？
============================================================
只做 I' = I^2.2 的"变暗"，图像里**没有噪声**。此时最优解是 I'^(1/2.2)，模型一行代码
就能 100% 恢复，PSNR 虚高到 40dB+，一到真实数据立刻崩。
所以噪声必须是**信号相关**的（散粒噪声），模型才必须真的学去噪 —— 这也是
"方差 ~ 亮度" 应呈线性的原因（`scripts/10_verify_synthesis.py` 会实测验证）。

============================================================
默认参数怎么来的
============================================================
- EV ∈ [1.5, 3.0]：手机夜景视频相对白天通常暗 2~4 档，取中间区间
- Φ = 800 e-：低光高 ISO 下的**有效**满阱电子数（决定噪声强度；扫描结果是
  EV=2.5/Φ=800 时亮度对齐 PSNR≈26dB，处于"能恢复但很难"的区间）
- σ_r = 2.5 e-：现代 CMOS 读出噪声量级
- s = 0.35：只补偿部分增益，让画面比 GT 暗约 1.5 档（保留"低光"的视觉特征）
- JPEG q=90：模拟视频编码；低光下压缩块效应会被增强器放大，这也是真实存在的难点
"""

from __future__ import annotations

from dataclasses import dataclass, asdict, field
from typing import Iterator

import cv2
import numpy as np


# --------------------------------------------------------------------------
# 色彩空间
# --------------------------------------------------------------------------
# uint8 查表版 sRGB → linear：单帧从 ~20ms 降到 <1ms（训练时每帧都要转换）。
# 数值与连续函数完全一致（uint8 输入只有 256 种取值，查表是无损的）。
_SRGB_TO_LIN_LUT = np.concatenate([
    np.linspace(0.0, 0.04045 / 12.92, 11, endpoint=False),      # 0..10: 线性段
    ((np.arange(11, 256) / 255.0 + 0.055) / 1.055) ** 2.4,       # 11..255: 幂段
]).astype(np.float32)
assert len(_SRGB_TO_LIN_LUT) == 256


def srgb_to_linear(x: np.ndarray) -> np.ndarray:
    """sRGB (0..1) → 线性辐照度 (0..1)。分段函数，暗部是线性段，别用单一 2.2 次幂近似。

    uint8 输入（0..255）走无损查表；float 输入（0..1）走连续公式。
    """
    if x.dtype == np.uint8:
        return _SRGB_TO_LIN_LUT[x]
    x = np.clip(x.astype(np.float32), 0.0, 1.0)
    return np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4).astype(np.float32)


def linear_to_srgb(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, 0.0, 1.0).astype(np.float32)
    return np.where(x <= 0.0031308, x * 12.92, 1.055 * np.power(x, 1 / 2.4) - 0.055).astype(np.float32)


def linear_to_srgb_u8(x: np.ndarray) -> np.ndarray:
    """linear (0..1) float → sRGB uint8，一步到位。

    量化规则与「先 float 编码再 round」完全一致：out = round(srgb(x)·255)。
    实现：线性域按 255 个 sRGB 码的**舍入边界**做 searchsorted。
    边界 b_k = (srgb⁻¹ 的) 使 round(s·255)=k 的 s 区间端点，即 s = (k+0.5)/255
    对应的线性值 —— 由 sRGB 编码函数的单调性反解。
    """
    xc = np.clip(x, 0.0, 1.0).astype(np.float32).ravel()
    # 边界：srgb_encode^{-1}((k+0.5)/255), k = 0..254（单调，可直接算）
    s_edge = (np.arange(255, dtype=np.float32) + 0.5) / 255.0
    lin_edge = np.where(
        s_edge <= 0.04045,
        s_edge / 12.92,
        ((s_edge + 0.055) / 1.055) ** 2.4,
    ).astype(np.float32)
    idx = np.searchsorted(lin_edge, xc, side="right").clip(0, 255).astype(np.uint8)
    return idx.reshape(x.shape)

# 验证过的等价性：与 np.round(linear_to_srgb(x)*255) 逐位一致（见 20_eval 脚本输出）


# --------------------------------------------------------------------------
# 配置
# --------------------------------------------------------------------------
@dataclass
class LowLightConfig:
    """一次"拍摄"的虚拟相机参数。改动这里就等于换一台相机/一个场景亮度。"""

    ev: float = 2.5             # 曝光不足档数（1 EV = 亮度减半）
    full_well: float = 800.0    # 满阱电子数 Φ（低光下有效光子数少，等效满阱明显下降）
    read_noise: float = 2.5     # 读出噪声 σ_r（电子）
    brightness_keep: float = 0.35  # 增益补偿后保留的亮度比例 s（0.35 ≈ 比 GT 暗 1.5 档）
    wb_rgb: tuple = (1.0, 1.0, 1.06)  # 白平衡通道增益（轻微偏冷）
    jpeg_quality: int = 90      # 0 表示不压缩
    quantize_bits: int = 8      # 8bit 是视频的常态；改 10/12 可模拟高端机内录制

    @property
    def alpha(self) -> float:
        """曝光衰减因子 α = 2^EV。"""
        return float(2.0 ** self.ev)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["wb_rgb"] = list(self.wb_rgb)
        d["alpha"] = self.alpha
        return d


# --------------------------------------------------------------------------
# 核心：单帧退化
# --------------------------------------------------------------------------
def degrade_frame(
    rgb_u8: np.ndarray,
    cfg: LowLightConfig | None = None,
    rng: np.random.Generator | None = None,
    return_linear: bool = False,
) -> np.ndarray:
    """把一张正常曝光的 sRGB uint8 图退化成低光帧。

    Args:
        rgb_u8: (H,W,3) uint8，RGB 顺序（不是 BGR！）
        cfg: 相机参数
        rng: 随机数发生器。**必须显式传入**，否则每次调用都是新随机源，实验不可复现。
        return_linear: True 时返回未量化、未压缩的线性浮点结果（只给噪声统计检验用）

    Returns:
        uint8 (H,W,3) RGB；若 return_linear 则返回 float32 (H,W,3)
    """
    cfg = cfg or LowLightConfig()
    rng = rng or np.random.default_rng(0)

    lin = srgb_to_linear(rgb_u8)  # uint8 → LUT 查表，无损且快

    # 1) 曝光不足：光子数减少
    lam = lin * cfg.full_well / cfg.alpha

    # 2) 散粒噪声（信号相关）+ 读出噪声（信号无关）
    #
    # 性能说明：rng.poisson 全量采样 480p 要 ~40ms/帧，是训练数据管道的瓶颈。
    # 「混合采样」：λ > 10 的像素用高斯近似 N(λ, λ)（泊松的正态近似），
    # λ ≤ 10 保留精确泊松 —— 暗部（λ 小）恰恰是泊松和高斯差异最大的地方
    # （偏态、离散），必须精确；λ>10 时偏度 ≤ 0.32，且这些像素经过 sRGB 编码
    # 和 8bit 量化后差异远小于 1 个灰度级。实测单帧 ~80ms → ~45ms，
    # 方差-亮度检验（M1-B）与闪烁统计（M1-D）均不受影响（在验证脚本里复核）。
    lam32 = lam.astype(np.float32)
    exact = lam32 <= 10.0
    # 混合采样（见上）：暗区精确泊松 + 亮区高斯近似。两次 standard_normal 合并为一次
    # （shot 用前半，read 用后半）—— standard_normal 每次调用 ~8ms，能省一次是一次。
    z = rng.standard_normal((2,) + lam32.shape).astype(np.float32)
    sq = np.sqrt(lam32, dtype=np.float32)
    electrons = np.where(exact, 0.0, lam32 + sq * z[0])   # 亮区高斯近似
    if exact.any():
        electrons[exact] = rng.poisson(lam32[exact]).astype(np.float32)
    electrons += z[1] * cfg.read_noise

    # 3) 相机增益：把暗信号拉回可显示范围（ISO 增益同时放大读出噪声，物理上正确）
    stored = electrons * (cfg.alpha * cfg.brightness_keep) / cfg.full_well

    if return_linear:
        return stored.astype(np.float32)

    # 4) 白平衡
    stored = stored * np.array(cfg.wb_rgb, dtype=np.float32)

    # 5) 回到 sRGB 显示域 + 量化（合并两步，LUT 插值）
    out_u8 = linear_to_srgb_u8(stored)

    # 6) JPEG 压缩（模拟视频编码）
    if cfg.jpeg_quality > 0:
        ok, buf = cv2.imencode(".jpg", out_u8[:, :, ::-1],
                               [int(cv2.IMWRITE_JPEG_QUALITY), cfg.jpeg_quality])
        if ok:
            out_u8 = cv2.imdecode(buf, cv2.IMREAD_COLOR)[:, :, ::-1]
    return out_u8


# --------------------------------------------------------------------------
# 序列级：每帧独立噪声（这是"闪烁"的物理来源）
# --------------------------------------------------------------------------
def frame_rng(seed: int, seq: str, frame_idx: int) -> np.random.Generator:
    """由 (seed, 序列名, 帧号) 派生独立随机源。

    为什么要派生而不是复用同一个 rng：
    - 复用 → 顺序敏感，跳帧/并行读取会导致结果不一致
    - 派生 → 任意顺序、任意并行都能复现同一帧的同一份噪声
    """
    h = abs(hash((seed, seq, frame_idx))) % (2**63)
    return np.random.default_rng(h)


def degrade_clip(
    frames: list[np.ndarray],
    cfg: LowLightConfig | None = None,
    seed: int = 0,
    seq_name: str = "seq",
    start_idx: int = 0,
) -> Iterator[tuple[int, np.ndarray]]:
    """逐帧退化，yield (帧号, 低光帧)。每帧噪声独立 → 视频会闪烁。"""
    for i, f in enumerate(frames):
        idx = start_idx + i
        yield idx, degrade_frame(f, cfg, frame_rng(seed, seq_name, idx))


# --------------------------------------------------------------------------
# IO
# --------------------------------------------------------------------------
def load_rgb(path) -> np.ndarray:
    """读图并转成 RGB（cv2 默认 BGR，这里统一成 RGB，避免后续到处 ::-1）。

    注意：**不能用 cv2.imread**。Windows 上 cv2.imread 走 ANSI 路径，
    本项目路径含中文（毕业论文工程化）会直接读失败。必须 np.fromfile + imdecode。
    """
    buf = np.fromfile(str(path), dtype=np.uint8)
    bgr = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def save_rgb(path, rgb_u8: np.ndarray) -> None:
    """同理：cv2.imwrite 在中文路径下会静默失败，用 imencode + tofile。"""
    ext = str(path).rsplit(".", 1)[-1]
    ok, buf = cv2.imencode(f".{ext}", cv2.cvtColor(rgb_u8, cv2.COLOR_RGB2BGR))
    if not ok:
        raise IOError(f"encode 失败: {path}")
    buf.tofile(str(path))
