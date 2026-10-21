"""M4-T3 · 循环 U-Net（ConvGRU 隐状态跨帧传递）

============================================================
T3 要解决什么（T1/T2 治不了的病）
============================================================
T1（推理期融合）和 T2（训练期 warp 约束）的消融结果指向同一个结论：
tOF 残差的大头是**内容相关的模型误差**——同一场景 U-Net 每帧"犯同样的错"，
warp 对齐后平均不掉。T3 的对症药：让网络**记住上一帧看过什么**，
当前帧的去噪/增强决策以历史为条件 → 相邻帧的输出天然绑定，
内容相关残差从"每帧独立采样"变成"条件于历史的推断"。

============================================================
为什么是 ConvGRU、插在瓶颈层
============================================================
- GRU vs LSTM：参数少 1/4，效果在该任务量级下无显著差异；ConvGRU（Ballas et al. 2016）
  是视频预测/增强的标准选择。
- 插在瓶颈层（base*8=256 通道，256² patch 下 32×32 空间分辨率）：
  隐状态的计算/显存开销最小，而语义信息最浓缩——时序记忆应该记"场景在发生什么"，
  不是记"像素长什么样"（像素级记忆交给 T1 的 warp 融合）。
- 隐状态 h 只进解码器、不动 skip：skip 通路保持逐帧独立（高频细节不该被历史模糊），
  时序信息只通过瓶颈注入。这是"记忆管语义、skip 管细节"的职责分离。

============================================================
评估的对照设计（防自己骗自己）
============================================================
T3 有两种推理方式，都必须报告：
- **stateful**：整条序列顺序推理，h 跨帧传递——T3 的真实形态
- **per-frame 重置**：每帧 h=0——等价于逐帧模型（+GRU 参数量）
如果 stateful 不明显好于 per-frame 重置，收益就不来自循环本身，
而只是"多了一些参数"——这个对照是 T3 结论合法性的底线。
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from src.models.unet import Down, Up, conv_block, count_params
except ImportError:  # 直接以脚本运行本文件时
    from unet import Down, Up, conv_block, count_params


class ConvGRU(nn.Module):
    """卷积 GRU：z/r 门与候选态都是 3×3 卷积，空间结构保留在隐状态里。

    h 首帧传 None → 内部置零。h 与 x 同通道数（256），同空间尺寸。
    """

    def __init__(self, ch: int, k: int = 3):
        super().__init__()
        self.ch = ch
        pad = k // 2

        def gate() -> nn.Conv2d:
            c = nn.Conv2d(ch * 2, ch, k, padding=pad)
            nn.init.orthogonal_(c.weight)
            nn.init.zeros_(c.bias)
            return c

        self.conv_z = gate()
        self.conv_r = gate()
        # 候选态：输入 h~（重置后的历史）与 x
        self.conv_h = gate()

    def forward(self, x: torch.Tensor, h: torch.Tensor | None) -> torch.Tensor:
        if h is None:
            h = torch.zeros(x.shape[0], self.ch, *x.shape[2:],
                            device=x.device, dtype=x.dtype)
        zh = torch.cat([x, h], dim=1)
        z = torch.sigmoid(self.conv_z(zh))          # 更新门：多少旧记忆保留
        r = torch.sigmoid(self.conv_r(zh))          # 重置门：写候选态时忘多少
        h_tilde = torch.tanh(self.conv_h(torch.cat([x, r * h], dim=1)))
        return (1 - z) * h + z * h_tilde


class UNetRecurrent(nn.Module):
    """UNetEnhance 主干 + 瓶颈 ConvGRU。

    forward(x)            → 逐帧形态（等价于 per-frame 重置），与 UNetEnhance 接口一致
    forward_state(x, h)   → (out, h_new)，循环形态
    """

    def __init__(self, base: int = 32, in_ch: int = 3, out_ch: int = 3):
        super().__init__()
        self.inc = conv_block(in_ch, base)
        self.d1 = Down(base, base * 2)
        self.d2 = Down(base * 2, base * 4)
        self.d3 = Down(base * 4, base * 8)
        self.gru = ConvGRU(base * 8)            # ← 与 UNetEnhance 唯一的结构差异
        self.u1 = Up(base * 8, base * 4, base * 4)
        self.u2 = Up(base * 4, base * 2, base * 2)
        self.u3 = Up(base * 2, base, base)

        self.res_head = nn.Conv2d(base, out_ch, 1)
        nn.init.normal_(self.res_head.weight, std=1e-3)
        nn.init.zeros_(self.res_head.bias)
        self.gain_head = nn.Sequential(
            nn.Linear(base, 32), nn.ReLU(inplace=True), nn.Linear(32, 1)
        )
        nn.init.normal_(self.gain_head[2].weight, std=1e-3)
        nn.init.zeros_(self.gain_head[2].bias)

    def _heads(self, x: torch.Tensor, y: torch.Tensor):
        res = torch.tanh(self.res_head(y)) * 0.5
        g = 1.0 + 2.0 * torch.sigmoid(self.gain_head(y.mean(dim=(2, 3))))
        g = g.view(-1, 1, 1, 1)
        out = torch.clamp(x * g + res, 0.0, 1.0)
        return g, res, out

    def forward_state(self, x: torch.Tensor, h: torch.Tensor | None):
        x1 = self.inc(x)
        x2 = self.d1(x1)
        x3 = self.d2(x2)
        x4 = self.d3(x3)
        h_new = self.gru(x4, h)                 # 时序记忆只在这一处注入
        y = self.u1(h_new, x3)
        y = self.u2(y, x2)
        y = self.u3(y, x1)
        g, res, out = self._heads(x, y)
        return out, h_new

    def forward_parts(self, x: torch.Tensor):
        raise NotImplementedError("循环模型请用 forward_state(x, h)；逐帧对照用 forward(x)")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_state(x, None)[0]


if __name__ == "__main__":
    net = UNetRecurrent()
    x = torch.rand(2, 3, 256, 256)
    out1, h = net.forward_state(x, None)
    out2, h2 = net.forward_state(x, h)
    print(f"params={count_params(net)/1e6:.2f}M  out={tuple(out1.shape)}  h={tuple(h.shape)}")
    print(f"per-frame forward consistent: {torch.allclose(out1, net(x))}")
