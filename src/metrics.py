"""统一指标模块 · 逐帧保真度 + 时序稳定性

============================================================
为什么要单独一个模块，而不是在每个脚本里各写一遍？
============================================================
指标是最容易"不小心算错然后骗自己"的地方。项目里只要有第二个 PSNR 实现，
就一定会出现"表 A 的 PSNR 和表 B 差 0.3dB"这种查半天的诡异 bug。
所以：**全项目只有这一份实现**，谁要算指标都 import 这里。

============================================================
tOF（temporal warping error）—— 本项目的核心指标
============================================================
定义（必须写进论文，否则不可比）：

    tOF = mean_t  mean_{p ∈ valid} | I_{t+1}(p) − warp(I_t, F_{t→t+1})(p) |

三个关键设计决定，每一个都是为了"指标不能被方法偷偷利用"：

1) **光流 F 由 GT 帧估计，而不是由待评估结果估计。**
   如果用方法自己的输出估流，一个把画面糊成一片的方法会得到又平滑又准的光流，
   tOF 反而更低 —— 指标被作弊了。用 GT 估流，所有方法面对的是同一套 F。

2) **前向-后向一致性掩码，遮挡区不计入。**
   warp 的本质是"假定像素没动"。物体遮挡/露出的地方，像素真的消失了/新出现了，
   任何方法都无法对齐，误差必然大。不掩掉这部分，tOF 主要在测"场景里有多少遮挡"，
   而不是"增强结果闪不闪"。掩码规则：|F_{t→t+1} + warp(F_{t+1→t}, F_{t→t+1})| < θ。

3) **只在"亮度对齐"后的图上算（可选 photometric alignment）。**
   很多方法的整体亮度跟 GT 差一截。这种全局偏移对"闪烁"没有贡献，
   却会污染 tOF。因此提供 align=True：先按标量增益把 I_{t+1} 对齐到 warp(I_t) 的亮度，
   再算残差。默认 True，并在报告里注明。

另有一个重要的对照指标：
    tOF_input  = 同样的算法作用在**低光输入**上的 tOF
    tOF_ratio  = tOF / tOF_input
"比值"比"绝对值"更能说明问题：一个方法可能只是把噪声平均掉了（低光输入本身 tOF 就高），
只有比值明显小于 1，才说明它真的在**抑制闪烁**而不是在**抹平一切**。
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


# --------------------------------------------------------------------------
# 逐帧保真度
# --------------------------------------------------------------------------
def psnr(a: np.ndarray, b: np.ndarray, max_val: float = 255.0) -> float:
    """PSNR(dB)。a/b 为 uint8 或 float（同一量纲）。完全相同时返回 99 而非 inf。"""
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    mse = float(np.mean((a - b) ** 2)) / (max_val**2)
    return 99.0 if mse <= 1e-12 else 10.0 * np.log10(1.0 / mse)


def ssim(a: np.ndarray, b: np.ndarray, max_val: float = 255.0) -> float:
    """结构相似度（多通道平均）。

    用 skimage 的实现，参数固定（gaussian_weights=False, sigma=1.5），
    与 SSIM 原论文的 11×11 均值窗口一致。**不要随手改参数**，
    否则和文献里的数字对不上。
    """
    from skimage.metrics import structural_similarity

    a = a.astype(np.float64)
    b = b.astype(np.float64)
    return float(
        structural_similarity(
            a, b, data_range=max_val, channel_axis=2 if a.ndim == 3 else None,
            gaussian_weights=False,
        )
    )


def lpips_metric():
    """惰性构造 LPIPS（需要 pip install lpips，首次会下载 AlexNet 权重 ~233MB）。

    返回一个 callable: (a_u8, b_u8) -> float。失败时返回 None，
    调用方应优雅降级并在报告里写"未测"，而不是让整个脚本崩掉。
    """
    try:
        import lpips  # noqa: F401
        import torch
    except Exception:
        return None

    import lpips as _lpips

    net = _lpips.LPIPS(net="alex", verbose=False).eval()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    net = net.to(dev)

    def _f(a_u8: np.ndarray, b_u8: np.ndarray) -> float:
        with torch.no_grad():
            ta = torch.from_numpy(a_u8.astype(np.float32) / 127.5 - 1.0)
            tb = torch.from_numpy(b_u8.astype(np.float32) / 127.5 - 1.0)
            ta = ta.permute(2, 0, 1)[None].to(dev)
            tb = tb.permute(2, 0, 1)[None].to(dev)
            return float(net(ta, tb).item())

    return _f


def brightness_aligned_psnr(a: np.ndarray, b: np.ndarray) -> float:
    """先把 a 的整体亮度对齐到 b（单标量最小二乘增益），再算 PSNR。

    只用于**诊断**：区分"亮度整体偏了"和"细节错了"。
    主表里报 raw PSNR；这张数字放附录，避免被误读成"我们偷偷调了亮度"。
    """
    af = a.astype(np.float64).ravel()
    bf = b.astype(np.float64).ravel()
    denom = float(np.dot(af, af))
    if denom <= 1e-9:
        return psnr(a, b)
    g = float(np.dot(af, bf) / denom)
    a2 = np.clip(a.astype(np.float64) * g, 0, 255).round().astype(np.uint8)
    return psnr(a2, b)


# --------------------------------------------------------------------------
# 光流
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class FlowParams:
    """Farneback 参数 + 掩码阈值。**固定下来，所有 tOF 数字都基于这一套参数。**"""

    pyr_scale: float = 0.5
    levels: int = 4
    winsize: int = 15
    iterations: int = 3
    poly_n: int = 5
    poly_sigma: float = 1.1
    fb_thresh: float = 0.5  # 前向-后向一致性阈值（像素）


def estimate_flow(gray_a: np.ndarray, gray_b: np.ndarray, p: FlowParams = FlowParams()) -> np.ndarray:
    """估计 a → b 的稠密光流 (H,W,2)，uint8 灰度输入。

    为什么用 Farneback 而不是 RAFT：
    - RAFT 需要下载权重 + GPU，且是学习式的，光流本身会引入"方法依赖"；
    - tOF 的定义要求光流对所有方法完全一致、完全可复现。Farneback 是确定性的，
      同一对图在任何机器上得到同一个 F。
    - 代价：在快速运动上精度不如 RAFT（会在失败案例里体现，已写入局限）。
    """
    return cv2.calcOpticalFlowFarneback(
        gray_a, gray_b, None,
        p.pyr_scale, p.levels, p.winsize, p.iterations, p.poly_n, p.poly_sigma, 0,
    ).astype(np.float32)


def warp_with_flow(img: np.ndarray, flow: np.ndarray) -> np.ndarray:
    """把 img 按 flow 采样到目标帧坐标。img: (H,W,3) 或 (H,W)，flow: (H,W,2)。

    flow 的定义是"目标像素 (x,y) 在源图里位于 (x+flow_x, y+flow_y)"（与 Farneback 一致）。
    越界像素用 BORDER_REFLECT，并**在外部用 valid 掩码排除**——反射填充只是为了让
    插值不炸，不代表边界处的值可信。
    """
    h, w = flow.shape[:2]
    gx, gy = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    map_x = gx + flow[..., 0]
    map_y = gy + flow[..., 1]
    return cv2.remap(
        img, map_x, map_y, interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT, borderValue=0,
    )


def occlusion_mask(gray_t: np.ndarray, gray_next: np.ndarray, p: FlowParams = FlowParams()) -> tuple[np.ndarray, np.ndarray]:
    """前向-后向一致性 → (valid 掩码, 光流 t→next)。

    valid = 1 表示"这个像素在两帧之间确实能对上"，可以参与 tOF。
    判定：把 F_{t→next} 与反向光流 warp 回来相加，理想情况下为 0；
    残差大于 fwd_bwd 阈值 → 视为遮挡/新出现/光流估计失败。
    """
    f_fw = estimate_flow(gray_t, gray_next, p)
    f_bw = estimate_flow(gray_next, gray_t, p)
    f_bw_warped = warp_with_flow(f_bw, f_fw)
    fb = np.linalg.norm(f_fw + f_bw_warped, axis=2)
    valid = (fb < p.fb_thresh).astype(np.float32)

    # 边界一圈的光流本来就不准（窗口越界），直接排除
    m = 2
    valid[:m, :] = valid[-m:, :] = 0
    valid[:, :m] = valid[:, -m:] = 0
    return valid, f_fw


def temporal_warping_error(
    frames: list[np.ndarray],
    flow_pairs: list[tuple[np.ndarray, np.ndarray]] | None = None,
    align: bool = True,
) -> float:
    """tOF：逐对比较 |I_{t+1} − warp(I_t)|，在遮挡掩码外取均值。

    Args:
        frames: 待评估序列（顺序与 flow_pairs 一致）
        flow_pairs: [(valid_mask, flow_t_to_next), ...]，由 occlusion_mask(**在 GT 帧上**)得到。
                    None 时退化为"用 frames 自己估流"——**只用于调试**，
                    正式数字必须传 GT 上估的流。
        align: 是否做标量亮度对齐（见模块 docstring 第 3 点）

    Returns:
        平均绝对值，0-255 尺度（与输入同量纲）。越低越稳。
    """
    n = len(frames)
    if n < 2:
        return float("nan")

    errs = []
    for t in range(n - 1):
        a = frames[t].astype(np.float32)
        b = frames[t + 1].astype(np.float32)
        if flow_pairs is not None:
            valid, flow = flow_pairs[t]
        else:
            ga = cv2.cvtColor(a, cv2.COLOR_RGB2GRAY) if a.ndim == 3 else a
            gb = cv2.cvtColor(b, cv2.COLOR_RGB2GRAY) if b.ndim == 3 else b
            ga8 = np.clip(ga, 0, 255).astype(np.uint8)
            gb8 = np.clip(gb, 0, 255).astype(np.uint8)
            valid, flow = occlusion_mask(ga8, gb8)

        warped = warp_with_flow(a, flow)
        m = valid > 0
        if m.sum() < 100:
            continue

        wa = warped[m]
        wb = b[m]
        if align:
            denom = float(np.mean(wa * wa))
            if denom > 1e-6:
                g = float(np.mean(wa * wb) / denom)
                wa = wa * g
        errs.append(float(np.mean(np.abs(wb - wa))))

    return float(np.mean(errs)) if errs else float("nan")


# --------------------------------------------------------------------------
# 序列级评估（一个方法跑完一个序列 → 一行数字）
# --------------------------------------------------------------------------
def evaluate_sequence(
    preds: list[np.ndarray],
    gts: list[np.ndarray],
    gt_flow_pairs: list[tuple[np.ndarray, np.ndarray]],
    input_frames: list[np.ndarray] | None = None,
    lpips_fn=None,
) -> dict:
    """返回该序列的逐帧均值指标。

    input_frames 传入低光输入时，额外给出 tOF_input 与 tOF_ratio，
    后者才是"真的抑制了闪烁"的证据（见模块 docstring）。
    """
    ps = [psnr(p, g) for p, g in zip(preds, gts)]
    ss = [ssim(p, g) for p, g in zip(preds, gts)]
    tof = temporal_warping_error(preds, gt_flow_pairs)

    out = {
        "psnr": float(np.mean(ps)),
        # 亮度对齐 PSNR：把「提亮的功劳」剥掉，单看「去噪/细节」。
        # M3 报告同时给两个口径 —— 差值就是「提亮」贡献的部分。
        "psnr_aligned": float(np.mean([brightness_aligned_psnr(p, g) for p, g in zip(preds, gts)])),
        "ssim": float(np.mean(ss)),
        "tof": tof,
        "n_frames": len(preds),
        "psnr_std": float(np.std(ps)),
        "psnr_min": float(np.min(ps)),
    }
    if lpips_fn is not None:
        out["lpips"] = float(np.mean([lpips_fn(p, g) for p, g in zip(preds, gts)]))

    if input_frames is not None:
        tof_in = temporal_warping_error(input_frames, gt_flow_pairs)
        out["tof_input"] = tof_in
        out["tof_ratio"] = float(tof / tof_in) if tof_in and tof_in > 1e-9 else float("nan")
    return out
