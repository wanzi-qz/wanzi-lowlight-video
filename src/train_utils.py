"""M3 · 训练/数据集/损失

============================================================
数据管道里三个容易做错、做了就毁掉实验的点
============================================================
1) **噪声是训练时在线合成的，不是预生成文件**。
   每个 epoch 用**新的 seed** 重新退化一遍 → 等效于无限张数的噪声数据增广，
   网络学的是"去噪"这个映射，而不是记住某一份噪声。
   测试集则必须用固定 seed（M1 已定），否则测试数字不可复现。

2) **patch 采样必须以"同一条序列的连续帧"为单位**。
   逐帧独立采样会破坏后续 T2 的相邻帧需求；这里直接返回相邻 2 帧的 pair，
   供 T2 的 warp 一致性损失使用（M3 阶段只用第 1 帧，M4 复用同一管道）。

3) **亮度增益不作为标签、不进入 loss**。它是模型内部的自由度，
   用 PSNR 评估时**包含**了增益的对错（这是真实任务：模型自己决定提多亮）。

损失：L1 + 0.2·(1−SSIM)。只用 L1 会偏"均值模糊"；SSIM 项保结构。
两个损失都在 **[0,1] 浮点域**上算（不是 uint8），这一点常被搞错 ——
uint8 上的 SSIM 会把量化误差当成结构差异。
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from src.data import DAVIS_ROOT, SPLIT, TRAIN_PATCH, list_frames
from src.synthesis import LowLightConfig, degrade_frame, frame_rng, load_rgb, srgb_to_linear, linear_to_srgb

# --------------------------------------------------------------------------
# 数据集
# --------------------------------------------------------------------------
class LowLightPairDataset(Dataset):
    """返回 (low_t, low_{t+1}, gt_t, gt_{t+1}) 的 uint8 (H,W,3) 张量。

    - 在线合成：__getitem__ 里实时退化，noise seed 每 epoch 换（由外部 set_epoch 触发）
    - crop：随机位置，尺寸 TRAIN_PATCH，宽高均需 8 的倍数（U-Net 3 层下采样）
    - 相邻帧取自同一条序列（保证 T2 有意义的帧对）
    """

    def __init__(self, seqs: tuple[str, ...], cfg: LowLightConfig,
                 patch: int = TRAIN_PATCH, base_seed: int = 1000,
                 return_next_low: bool = False,
                 flow_blur: float = 1.5):
        """
        return_next_low: True 时额外返回下一帧的低光图 + 帧间光流（T2 的 warp 损失需要）。
            代价：每样本多一次退化（~7ms）+ 一次 Farneback（256² ~10ms）。
            M3 用 False（快），M4-T2 用 True。
        flow_blur: 估流前对低光灰度图的高斯模糊 σ（抗噪声，与 T1 融合保持一致）。
        """
        self.cfg = cfg
        self.patch = patch
        self.base_seed = base_seed
        self.epoch = 0
        self.return_next_low = return_next_low
        self.flow_blur = flow_blur
        # 预加载全部 GT 帧到内存（DAVIS 13 条 × ~80 帧 × 480p ≈ 800MB，可承受）。
        # 不预加载的话，每个 iter 都要 cv2.imdecode ~16 张 jpg，数据加载占掉
        # 训练时间的 90%（实测 1.9s/batch vs GPU 前向反向 0.17s/batch）。
        self._cache: dict[tuple[str, int], np.ndarray] = {}
        self.items: list[tuple[str, int]] = []
        for s in seqs:
            files = list_frames(DAVIS_ROOT, s)
            for t, f in enumerate(files):
                self._cache[(s, t)] = load_rgb(f)
            # 相邻帧 pair，隔 1 帧取一次（覆盖整条序列）
            self.items += [(s, t) for t in range(0, len(files) - 1, 2)]
        rng = random.Random(42)
        rng.shuffle(self.items)

    def set_epoch(self, e: int) -> None:
        """换 epoch = 换噪声。**必须显式调用**，否则每个 epoch 噪声一样 → 白白浪费在线合成。"""
        self.epoch = e

    def __len__(self) -> int:
        return len(self.items)

    def _load_pair(self, seq: str, t: int, seed_offset: int):
        """只合成**当前 batch 会被用**的那一帧的低光版本。

        性能关键点：degrade_frame 里 Poisson 是逐像素的，256²×3 就要 ~30ms，
        一张 480p (~460k 像素) ~55ms。本 dataset 在 M3 只用第 1 帧，但返回 2 帧
        是给 M4 复用的 —— **这是过早优化**。改成惰性：合成放 __getitem__ 之外
        （M4 需要时再调用 full_pair），默认只合成 gt_t 的低光版。
        """
        gt_t = self._cache[(seq, t)]
        gt_t1 = self._cache[(seq, t + 1)]
        seed = self.base_seed + self.epoch * 10_000 + seed_offset
        lo_t = degrade_frame(gt_t, self.cfg, frame_rng(seed, seq, t))
        return lo_t, None, gt_t, gt_t1

    def low_pair(self, seq: str, t: int, seed_offset: int):
        """M4 用：同帧两个相邻帧的完整低光 pair。默认路径不走这里。"""
        gt_t = self._cache[(seq, t)]
        gt_t1 = self._cache[(seq, t + 1)]
        seed = self.base_seed + self.epoch * 10_000 + seed_offset
        return (degrade_frame(gt_t, self.cfg, frame_rng(seed, seq, t)),
                degrade_frame(gt_t1, self.cfg, frame_rng(seed, seq, t + 1)),
                gt_t, gt_t1)

    def __getitem__(self, idx: int):
        seq, t = self.items[idx]
        gt_t = self._cache[(seq, t)]
        gt_t1 = self._cache[(seq, t + 1)]

        # 随机 crop（帧对用同一个 crop 位置，保证对齐）
        h, w = gt_t.shape[:2]
        ph = pw = self.patch
        if h > ph:
            y = np.random.randint(0, h - ph + 1)
        else:
            y, ph = 0, h
        if w > pw:
            x = np.random.randint(0, w - pw + 1)
        else:
            x, pw = 0, w

        gt_crop = gt_t[y:y + ph, x:x + pw]
        gt1_crop = gt_t1[y:y + ph, x:x + pw]  # 帧对必须同 crop（M4 的 warp 损失依赖对齐）

        # ⚠ 性能 + 正确性双关键：**先 crop 再退化**。
        # 旧写法是先退化整张 480p 再 crop —— Poisson 是逐像素的，一张 480p ~55ms，
        # batch 8 就是 0.44s，纯 CPU，把 GPU（0.17s/batch）完全喂不饱（实测 1.0s/it）。
        # 先 crop 后只需退化 256²，~7ms，快 8 倍。
        # 噪声是逐像素独立的，crop 前后**统计分布完全相同**；唯一差别是 JPEG 的
        # 8×8 块边界随 crop 原点变化 —— 二阶效应，当作免费的数据增广即可。
        seed = self.base_seed + self.epoch * 10_000 + idx
        lo_crop = degrade_frame(gt_crop, self.cfg, frame_rng(seed, seq, t))

        def c(img: np.ndarray, normalize: bool) -> torch.Tensor:
            v = img.astype(np.float32) / 255.0
            if normalize:
                # ⚠ 训练/评估必须用同一套输入归一化！
                # 事故记录：第一版这里没除 brightness_keep（喂暗图），
                # 而 31_eval_unet.py 里除了 —— 模型学到的 gain≈2.86 套在已提亮的
                # 输入上，输出整体过曝剪裁，eval PSNR 6.8dB（比不处理还差 8dB），
                # 而训练日志里 val 明明是 31.7dB。数字对不上时，先查两边的前处理。
                v = v / self.cfg.brightness_keep
            return torch.from_numpy(v).permute(2, 0, 1)

        if not self.return_next_low:
            return c(lo_crop, True), c(gt_crop, False), c(gt1_crop, False)

        # ---- T2 路径：额外返回 lo_{t+1} 与帧间光流 ----
        lo1_crop = degrade_frame(gt1_crop, self.cfg, frame_rng(seed, seq, t + 1))
        # 光流在低光帧上估计（推理时可得，不用 GT —— 与 T1 融合同一口径）
        def gray_np(img_u8: np.ndarray) -> np.ndarray:
            g = cv2.cvtColor(img_u8, cv2.COLOR_RGB2GRAY).astype(np.float32)
            return cv2.GaussianBlur(g, (0, 0), self.flow_blur) if self.flow_blur > 0 else g
        g0, g1 = gray_np(lo_crop), gray_np(lo1_crop)
        # 目标帧 t+1 → 源帧 t 的反向流（warp E_t 到 t+1 坐标系用）
        flow = cv2.calcOpticalFlowFarneback(g1, g0, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        return (c(lo_crop, True), c(gt_crop, False), c(gt1_crop, False),
                c(lo1_crop, True), torch.from_numpy(flow.astype(np.float32)).permute(2, 0, 1))


# --------------------------------------------------------------------------
# 损失
# --------------------------------------------------------------------------
def ssim_loss(x: torch.Tensor, y: torch.Tensor, data_range: float = 1.0) -> torch.Tensor:
    """简化版 SSIM（高斯核固定 11×11, σ=1.5，与文献一致），返回 1 − SSIM。

    不引第三方包，保证训练/评估/复现三方一致。

    ⚠ AMP 注意：必须在 fp32 下算。fp16 里 C1/C2 (~1e-4 量级) 会被舍入，
    分母 s_x + s_y + C2 在平坦区域接近 0 → 梯度 NaN（实测 48/全部参数梯度 NaN）。
    强制 cast 到 float32 后问题消失，代价可忽略（SSIM 的 conv 只占总 FLOPs 的零头）。
    """
    x = x.float()
    y = y.float()
    C1 = (0.01 * data_range) ** 2
    C2 = (0.03 * data_range) ** 2
    kernel = _gaussian_kernel(11, 1.5).to(x.device).repeat(x.shape[1], 1, 1, 1)

    mu_x = F.conv2d(x, kernel, padding=5, groups=x.shape[1])
    mu_y = F.conv2d(y, kernel, padding=5, groups=y.shape[1])
    mu_x2, mu_y2, mu_xy = mu_x * mu_x, mu_y * mu_y, mu_x * mu_y
    s_x = F.conv2d(x * x, kernel, padding=5, groups=x.shape[1]) - mu_x2
    s_y = F.conv2d(y * y, kernel, padding=5, groups=y.shape[1]) - mu_y2
    s_xy = F.conv2d(x * y, kernel, padding=5, groups=x.shape[1]) - mu_xy

    ssim_map = ((2 * mu_xy + C1) * (2 * s_xy + C2)) / ((mu_x2 + mu_y2 + C1) * (s_x + s_y + C2))
    return 1.0 - ssim_map.mean()


def _gaussian_kernel(size: int, sigma: float) -> torch.Tensor:
    ax = torch.arange(size, dtype=torch.float32) - (size - 1) / 2
    g = torch.exp(-(ax ** 2) / (2 * sigma ** 2))
    k = torch.outer(g, g)
    return (k / k.sum()).view(1, 1, size, size)


def enhancement_loss(pred: torch.Tensor, gt: torch.Tensor,
                     l1_weight: float = 1.0, ssim_weight: float = 0.2) -> torch.Tensor:
    """L1 + 0.2·(1−SSIM)。都工作在 [0,1] 浮点域。

    ⚠ AMP 关键细节：SSIM 必须在 autocast **之外**算。
    autocast 是 thread-local 的：即使 ssim_loss 内部把输入 cast 成 fp32，
    其中的 F.conv2d 仍会被 autocast 拦截成 fp16 —— 而 SSIM 里的 C1/C2 (~1e-4)
    在 fp16 下直接被舍入没了，分母趋于 0 → **48 个参数梯度全部 NaN**（实测）。
    fp16 单独跑 L1 没问题、单独跑 SSIM 也"看起来没问题"（前向值正常），
    组合后 NaN —— 这种"单项正常组合崩"的问题只能在真实训练循环里暴露。
    """
    loss = l1_weight * F.l1_loss(pred, gt)
    if ssim_weight > 0:
        with torch.autocast("cuda", enabled=False):
            loss = loss + ssim_weight * ssim_loss(pred.float(), gt.float())
    return loss


# --------------------------------------------------------------------------
# 学习率调度（cosine + warmup）
# --------------------------------------------------------------------------
def lr_lambda(step: int, warmup: int, total: int) -> float:
    if step < warmup:
        return step / max(1, warmup)
    t = (step - warmup) / max(1, total - warmup)
    return 0.5 * (1 + np.cos(np.pi * min(1.0, t)))


# --------------------------------------------------------------------------
# T2 · warp 一致性损失
# --------------------------------------------------------------------------


def temporal_warp_loss(pred_t: torch.Tensor, pred_t1: torch.Tensor,
                       flow_bwd: torch.Tensor, phot_sigma: float = 0.1) -> torch.Tensor:
    """L_temp = mean( m ⊙ | warp(E_t, F) − E_{t+1} | )

    F：目标帧 t+1 → 源帧 t 的反向流（在低光输入上估计，数据集里算好传进来）。
    m：光度置信掩码 = exp(-|warp(E_t) − E_{t+1}|/σ)，**detach** ——
        掩码是"哪里能对齐"的判断，不能当梯度通道，否则网络会学会把
        对不齐的地方变得更对不齐来骗掩码。

    为什么不用前向-后向一致性掩码（T1 融合用的那种）：
    训练时每样本要多算 2 次 Farneback，batch 8 要 +0.4s/it；光度掩码在 GPU 上
    一次逐像素运算就够。T1 推理时用完整掩码（便宜且更准），训练用近似掩码 ——
    两边口径的差异在报告里注明。
    """
    b, _, h, w = pred_t.shape
    dev = pred_t.device
    # 像素坐标 → 归一化坐标 [-1,1]
    gx, gy = torch.meshgrid(
        torch.linspace(-1, 1, w, device=dev), torch.linspace(-1, 1, h, device=dev),
        indexing="xy")
    base = torch.stack([gx, gy], dim=0)[None]                       # (1,2,H,W)
    scale = torch.tensor([2.0 / (w - 1), 2.0 / (h - 1)], device=dev).view(1, 2, 1, 1)
    grid = (base + flow_bwd * scale).permute(0, 2, 3, 1)            # (B,H,W,2)
    warped = torch.nn.functional.grid_sample(
        pred_t, grid, mode="bilinear", padding_mode="border", align_corners=True)
    # ⚠ align_corners 必须为 True：上面 base 网格用 linspace(-1,1) 端点采样构建，
    # 是 align_corners=True 的坐标约定。混用 False 会引入最高半像素的系统错位
    # （诊断脚本实测：纯平移场下 MAE 0.0009 vs 0.0000），抬高 loss 地板、稀释梯度。

    diff = (warped - pred_t1).abs().mean(dim=1, keepdim=True)       # (B,1,H,W)
    mask = torch.exp(-diff / phot_sigma).detach()
    return (mask * diff).mean()


# --------------------------------------------------------------------------
# T3-long · K 帧片段数据集（BPTT-K 训练用）
# --------------------------------------------------------------------------
class LowLightClipDataset(Dataset):
    """返回连续 K 帧的低光/GT 片段 + 相邻帧对光流。

    与 LowLightPairDataset 的区别：片段长度可配（K=2 时等价于 pair）。
    目的：BPTT-K —— 让 h 在训练时累积 K-1 步，弥合「训练只有 1 步历史、
    测试有 23 步历史」的分布差（T3-BPTT2 负结果的首要嫌疑根因）。

    返回：
        lo:  (K,3,H,W) 低光帧（已归一化）
        gt:  (K,3,H,W) GT 帧
        flow:(K-1,2,H,W) 第 k+1→k 帧的反向流（低光帧上估计，temporal loss 用）
    """

    def __init__(self, seqs: tuple[str, ...], cfg: LowLightConfig,
                 clip_len: int = 6, patch: int = TRAIN_PATCH,
                 base_seed: int = 1000, flow_blur: float = 1.5):
        self.cfg = cfg
        self.patch = patch
        self.base_seed = base_seed
        self.clip_len = clip_len
        self.flow_blur = flow_blur
        self.epoch = 0
        self._cache: dict[tuple[str, int], np.ndarray] = {}
        self.items: list[tuple[str, int]] = []
        for s in seqs:
            files = list_frames(DAVIS_ROOT, s)
            for t, f in enumerate(files):
                self._cache[(s, t)] = load_rgb(f)
            # 片段起点：步长 = clip_len（不重叠，覆盖整条序列）
            self.items += [(s, t) for t in range(0, len(files) - clip_len + 1, clip_len)]
        rng = random.Random(42)
        rng.shuffle(self.items)

    def set_epoch(self, e: int) -> None:
        self.epoch = e

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int):
        seq, t0 = self.items[idx]
        K, (h, w) = self.clip_len, self._cache[(seq, t0)].shape[:2]
        ph = pw = self.patch
        y = np.random.randint(0, h - ph + 1) if h > ph else 0
        x = np.random.randint(0, w - pw + 1) if w > pw else 0
        if h <= ph:
            ph = h
        if w <= pw:
            pw = w

        seed = self.base_seed + self.epoch * 10_000 + idx
        los, gts = [], []
        for k in range(K):
            gt = self._cache[(seq, t0 + k)][y:y + ph, x:x + pw]
            lo = degrade_frame(gt, self.cfg, frame_rng(seed, seq, t0 + k))
            gts.append(gt)
            los.append(lo)

        # 相邻帧对光流（低光帧上估计，与 T1/T2 同口径；K-1 次 Farneback ~10ms/次）
        def gray_np(img_u8):
            g = cv2.cvtColor(img_u8, cv2.COLOR_RGB2GRAY).astype(np.float32)
            return cv2.GaussianBlur(g, (0, 0), self.flow_blur) if self.flow_blur > 0 else g

        flows = []
        for k in range(K - 1):
            g1, g0 = gray_np(los[k + 1]), gray_np(los[k])
            flows.append(cv2.calcOpticalFlowFarneback(
                g1, g0, None, 0.5, 3, 15, 3, 5, 1.2, 0))

        def stack_t(imgs, normalize):
            vs = []
            for img in imgs:
                v = img.astype(np.float32) / 255.0
                if normalize:
                    v = v / self.cfg.brightness_keep   # 与逐帧/pair 管道同一归一化
                vs.append(torch.from_numpy(v).permute(2, 0, 1))
            return torch.stack(vs)

        return (stack_t(los, True), stack_t(gts, False),
                torch.from_numpy(np.stack(flows).astype(np.float32)).permute(0, 3, 1, 2))
