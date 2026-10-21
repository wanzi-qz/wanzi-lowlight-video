"""M3 · 逐帧 U-Net 去噪/提亮 baseline（主 baseline，时序模块的挂载点）

============================================================
设计目标（先想清楚再写网络）
============================================================
1. **够小**：~1.9M 参数，RTX 4060 8GB 上 batch 8@256² 训练 < 2s/iter。
   大模型会把"时序一致性"的收益淹没在"模型容量"里，反而不利于讲清楚问题。
2. **保留高频**：3 层下采样 + U-Net skip。skip 的意义是把编码器的高频细节
   直接送到解码器；没有 skip 的纯编码-解码会把噪点和细节一起平滑掉（画面"糊"）。
3. **残差输出 + 显式亮度增益**：网络不直接输出图像，而是输出
   (a) 一个残差 Δ（学到"该减掉多少噪声"）
   (b) 一个全局增益 g（学到"该提亮多少倍"）
   最终 out = clamp(input * g + Δ)。
   为什么拆开：低光任务里"亮度"和"噪声"是两个量纲完全不同的东西。
   让网络从零学一个 3 通道图，等于让它自己发现"整体乘 3 倍"这件事，收敛很慢、
   而且不同帧学到的 g 会抖动 —— **这本身就是闪烁的一个来源**。
   残差形式（Residual Learning, He et al. 2016）让网络只需学"扰动"，好训得多。

============================================================
为什么逐帧模型**必然**闪烁（M3 要验证的核心命题）
============================================================
训练目标 L = ||E(I_t) − J_t||₁ 对每帧 t **独立**求最小。
每帧的噪声实例不同 → 每帧的最优解不同 → 相邻帧的输出在"去噪强度"上有独立扰动，
表现出来就是**鼓包/呼吸/颗粒跳动**。
网络再准也逃不掉：因为 loss 里**根本没有跨帧项**。
这不是实现问题，是目标函数的结构性问题 —— M4 就是去改这个目标函数。

============================================================
输入归一化（重要）
============================================================
输入按 x/255，再除以 brightness_keep（≈0.35）把亮度拉回 GT 量级附近，
让网络看到"差不多正常亮度但很脏"的图。这不是作弊：brightness_keep 是合成时
**我们设定的相机增益**，真实相机也做类似的事。但它确实是一个**先验**，
所以必须在报告里写明「本 baseline 已知增益」。
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def conv_block(cin: int, cout: int, k: int = 3) -> nn.Sequential:
    """Conv → BN → ReLU ×2。BN 在这里是安全的（batch≥4，且推理时用 running stats）。"""
    return nn.Sequential(
        nn.Conv2d(cin, cout, k, padding=k // 2, bias=False),
        nn.BatchNorm2d(cout),
        nn.ReLU(inplace=True),
        nn.Conv2d(cout, cout, k, padding=k // 2, bias=False),
        nn.BatchNorm2d(cout),
        nn.ReLU(inplace=True),
    )


class Down(nn.Module):
    def __init__(self, cin: int, cout: int):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.conv = conv_block(cin, cout)

    def forward(self, x):
        return self.conv(self.pool(x))


class Up(nn.Module):
    """上采样后 concat skip。用 bilinear 而非转置卷积：
    转置卷积在低光数据上容易产生棋盘格伪影（checkerboard），而棋盘格会**逐帧抖动**，
    直接污染 tOF。这是一个"为了指标公平"的实现细节。"""

    def __init__(self, cin: int, cskip: int, cout: int):
        super().__init__()
        self.conv = conv_block(cin + cskip, cout)

    def forward(self, x, skip):
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return self.conv(torch.cat([x, skip], dim=1))


class UNetEnhance(nn.Module):
    """3 层下采样的 U-Net，输出 (gain, residual)。

    forward(x) 返回增强图（已 clamp 到 [0,1]）。
    forward_parts(x) 返回 (gain, residual, out)，训练/分析时用。
    """

    def __init__(self, base: int = 32, in_ch: int = 3, out_ch: int = 3):
        super().__init__()
        self.inc = conv_block(in_ch, base)
        self.d1 = Down(base, base * 2)
        self.d2 = Down(base * 2, base * 4)
        self.d3 = Down(base * 4, base * 8)
        self.u1 = Up(base * 8, base * 4, base * 4)
        self.u2 = Up(base * 4, base * 2, base * 2)
        self.u3 = Up(base * 2, base, base)

        # 残差头：小规模初始化而不是全零。
        # ⚠ 教学笔记（真实踩过的坑）：如果 res_head 和 gain_head 的最后一层**同时**
        # 零初始化，那么两者对主干特征 y 的梯度都经过零权重（d res/dy ∝ W = 0），
        # 整个 U-Net 主干的梯度恒为 0 —— 只有这两个头自己能动，网络永远学不会。
        # 表现是：loss 从 0.175 缓慢降到 0.16 就卡住（只有 gain 在动），
        # val PSNR 停在 21dB 附近（≈ input×2 的水平）。
        # 正确做法：残差头用小方差初始化（输出近似 0 但梯度通路畅通）。
        self.res_head = nn.Conv2d(base, out_ch, 1)
        nn.init.normal_(self.res_head.weight, std=1e-3)
        nn.init.zeros_(self.res_head.bias)

        # 增益头：全局一个标量（不是逐像素！）。逐像素增益 = 逐帧不一致的放大器。
        self.gain_head = nn.Sequential(
            nn.Linear(base, 32), nn.ReLU(inplace=True), nn.Linear(32, 1)
        )
        nn.init.normal_(self.gain_head[2].weight, std=1e-3)
        nn.init.zeros_(self.gain_head[2].bias)

    def forward_parts(self, x: torch.Tensor):
        x1 = self.inc(x)
        x2 = self.d1(x1)
        x3 = self.d2(x2)
        x4 = self.d3(x3)
        y = self.u1(x4, x3)
        y = self.u2(y, x2)
        y = self.u3(y, x1)

        res = torch.tanh(self.res_head(y)) * 0.5          # 残差限制在 [-0.5, 0.5]
        g = 1.0 + 2.0 * torch.sigmoid(self.gain_head(y.mean(dim=(2, 3))))  # gain ∈ (1, 3)
        g = g.view(-1, 1, 1, 1)
        out = torch.clamp(x * g + res, 0.0, 1.0)
        return g, res, out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_parts(x)[2]


def count_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


if __name__ == "__main__":  # 烟雾测试：形状对不对、参数量多少
    net = UNetEnhance()
    x = torch.rand(2, 3, 256, 256)
    g, r, out = net.forward_parts(x)
    print(f"params={count_params(net)/1e6:.2f}M  out={tuple(out.shape)}  "
          f"gain={g.flatten().tolist()}  res_abs_max={r.abs().max().item():.4f}")
    # 8 的倍数检查：U-Net 3 层下采样 → 输入必须能被 8 整除
    y = net(torch.rand(1, 3, 250, 254))
    print("non-multiple-of-8 input still works:", tuple(y.shape))
