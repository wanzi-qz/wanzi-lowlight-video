"""M2 · 经典 low-light enhancement baseline（零训练）

============================================================
为什么花时间做这些"看起来很土"的方法？
============================================================
因为它们是**必须存在的对照组**。三个具体作用：

1. **证明"变亮"≠"变好"。** Gamma 之后画面确实亮了，但噪声被同样放大，
   PSNR 甚至可能比不处理更低。没有 B0，评审会问"你的 U-Net 到底比
   '把图乘 3' 强多少？"——这个问题必须能当场回答。

2. **给出 tOF 的下界/上界参照。** 逐像素操作的 tOF 与输入几乎同量级
   （噪声被放大后更差），这是"逐帧处理必闪"的最直接证据。

3. **B1 的初始化/对照基线。** 如果 U-Net 输不过 CLAHE，说明训练出问题了，
   而不是"任务太难"。

============================================================
三个方法的原理（各自代表一条思路）
============================================================
**gamma**：全局单调映射 out = in^(1/γ)。0 参数、0 训练。
   它能修正"曝光线性错位"，但对**空间不均匀**的光照无能为力，且对噪声是
   等比例放大（噪声和信号一起被拉），所以暗部会更脏。

**CLAHE**：在局部块内做直方图均衡 + 裁剪（clipLimit）限制对比度放大倍数。
   clipLimit 就是"允许多大程度的局部增益"，越小越保守。它工作在亮度通道（LAB 的 L），
   保持色相不变。对不均匀光照有效，但局部增益会**同时放大噪声**，
   并引入块状伪影（相邻块的直方图不连续）。

**SSR（单尺度 Retinex）**：假设 图像 = 反射率 × 光照(I = R·L)，
   L 用高斯模糊近似（L = I * G_σ），则 R = log I − log L，再指数还原。
   它把"光照"当成低频、把"反射率"当成细节，因此能压掉不均匀光照。
   经典缺陷：输出常出现光晕（halo）在强边缘处，且颜色会偏灰（log 域的通道平均）。
   我们用"逐通道 SSR + 增益/偏移线性拉伸到 GT 动态范围"的常见实用版本。

============================================================
关键约定
============================================================
所有方法输入/输出都是 **(H,W,3) uint8, RGB**。
参数扫描、选参都只在 **val 集**上做，test 集一次性使用（见 ROADMAP 第 3 节）。
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

# --------------------------------------------------------------------------
# 所有方法统一的类型：一个 callable (rgb_u8) -> rgb_u8
# --------------------------------------------------------------------------


def gamma_correct(rgb_u8: np.ndarray, gamma: float = 2.2) -> np.ndarray:
    """out = (in/255)^(1/gamma) * 255。

    gamma > 1 → 提亮；gamma < 1 → 压暗。
    为什么默认 2.2：sRGB 的编码 γ 约为 2.2，把它"反掉"是最朴素的一次曝光补偿。
    注意：**这里没有任何去噪**，噪声会被同样提亮。这正是要展示的现象。
    """
    x = rgb_u8.astype(np.float32) / 255.0
    y = np.power(np.clip(x, 0.0, 1.0), 1.0 / gamma)
    return np.round(y * 255.0).clip(0, 255).astype(np.uint8)


def clahe_lab(
    rgb_u8: np.ndarray,
    clip_limit: float = 2.0,
    tile: int = 8,
    gamma: float = 1.0,
) -> np.ndarray:
    """在 LAB 的 L 通道上做 CLAHE，可选再叠一个 gamma。

    为什么在 LAB 而不在 RGB 上做：
    直方图均衡会**独立地**改每个通道的分布，RGB 三通道各均衡一次 → 色偏（"蜡像色"）。
    LAB 的 L 是亮度、a/b 是色度，只改 L 就只改亮度，色相基本不动。

    Args:
        clip_limit: 对比度上限倍数。越大越"猛"，噪声也越猛。
        tile: 分块数（tile×tile），块越多越局部。
        gamma: 均衡后再提亮（因为均衡本身不一定够亮）。
    """
    lab = cv2.cvtColor(rgb_u8, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=float(clip_limit), tileGridSize=(int(tile), int(tile)))
    l2 = clahe.apply(l)
    out = cv2.cvtColor(cv2.merge([l2, a, b]), cv2.COLOR_LAB2RGB)
    if abs(gamma - 1.0) > 1e-6:
        out = gamma_correct(out, gamma)
    return out


def single_scale_retinex(
    rgb_u8: np.ndarray,
    sigma: float = 80.0,
    gain: float = 1.0,
    offset: float = 0.0,
    per_channel: bool = True,
    clip_percentile: float = 0.5,
) -> np.ndarray:
    """SSR：R = exp(log(I+ε) − log(I*G_σ + ε))，再线性拉伸。

    Args:
        sigma: 高斯核尺度。越大 → 低频估计越"平滑" → 压光照越彻底，但光晕越明显。
        gain/offset: 线性拉伸（先按 per-channel 或全局百分位归一化，再乘）。
        per_channel: True 时三通道各自 SSR（保色能力强但可能偏灰）；
                     False 时在亮度通道上 SSR 后按原色比例回投（更保真，但对强色偏无效）。
        clip_percentile: 拉伸时两端各裁掉的百分比，抑制极值像素把动态范围拉开。

    实操细节：log 前要 +ε（1.0 量级），否则暗部 log(0) = -inf 直接炸。
    """
    x = rgb_u8.astype(np.float32) + 1.0
    if per_channel:
        blur = cv2.GaussianBlur(x, (0, 0), sigmaX=float(sigma), sigmaY=float(sigma),
                                borderType=cv2.BORDER_REFLECT)
        r = np.log10(x) - np.log10(blur + 1e-6)
    else:
        g = cv2.cvtColor(rgb_u8, cv2.COLOR_RGB2GRAY).astype(np.float32) + 1.0
        gb = cv2.GaussianBlur(g, (0, 0), sigmaX=float(sigma), sigmaY=float(sigma),
                              borderType=cv2.BORDER_REFLECT)
        rg = np.log10(g) - np.log10(gb + 1e-6)
        # 亮度 Retinex 结果再按"原色比例"回投到三通道（保持色相）
        r = rg[..., None] * (x / np.maximum(g[..., None], 1e-6))

    # 百分位拉伸到 [0,1]，避免单个高光像素把整幅图压暗
    lo = np.percentile(r, clip_percentile)
    hi = np.percentile(r, 100.0 - clip_percentile)
    r = (r - lo) / max(hi - lo, 1e-6)
    r = np.clip(r * gain + offset, 0.0, 1.0)
    return np.round(r * 255.0).astype(np.uint8)


# --------------------------------------------------------------------------
# 工厂：名字 → 方法（便于配置驱动 + 消融表统一调用）
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class MethodSpec:
    name: str
    fn: object  # callable(rgb_u8) -> rgb_u8
    n_params: int  # "参数量"列用，经典方法是 0 或个位数


def make_method(name: str, **kw) -> MethodSpec:
    """按名字构造方法。未知名字直接报错（别静默 fallback 到 identity，那会让消融表骗人）。"""
    name = name.lower()
    if name == "identity":
        return MethodSpec("identity", lambda x: x.copy(), 0)
    if name == "gamma":
        g = kw.get("gamma", 2.2)
        return MethodSpec(f"gamma(g={g:g})", lambda x: gamma_correct(x, g), 0)
    if name == "clahe":
        cl = kw.get("clip_limit", 2.0)
        tl = kw.get("tile", 8)
        ga = kw.get("gamma", 1.0)
        return MethodSpec(
            f"clahe(clip={cl:g},tile={tl},g={ga:g})",
            lambda x: clahe_lab(x, cl, tl, ga),
            0,
        )
    if name == "ssr":
        s = kw.get("sigma", 80.0)
        ga = kw.get("gain", 1.0)
        return MethodSpec(f"ssr(sigma={s:g},gain={ga:g})",
                          lambda x: single_scale_retinex(x, s, ga), 0)
    raise ValueError(f"未知方法: {name}")


def method_grid() -> dict[str, list[dict]]:
    """M2 的**候选参数网格**（只在 val 上扫，test 一次性用）。

    刻意给每个方法一个合理的搜索范围，而不是手调一个好看的数：
    如果手调 test 上的参数，后面所有对比都失去意义。
    """
    return {
        "gamma": [{"gamma": g} for g in (1.5, 2.2, 3.0, 4.0)],
        "clahe": [
            {"clip_limit": cl, "tile": tl, "gamma": ga}
            for cl in (1.5, 2.5, 4.0)
            for tl in (8, 16)
            for ga in (1.0, 1.6)
        ],
        "ssr": [
            {"sigma": s, "gain": ga}
            for s in (30.0, 80.0, 150.0)
            for ga in (1.0, 1.3)
        ],
    }
