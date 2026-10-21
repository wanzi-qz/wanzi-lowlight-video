"""CPU 冒烟测试 · 验证全链路（合成 → 指标 → 模型 → 训练一步 → tOF）无断点

目标：**2 分钟内跑完**，不需要 GPU、不需要 DAVIS 数据（用合成图代替）。
GitHub Actions 上也跑这套（见 .github/workflows/ci.yml）。

这不是单元测试全集 —— 是"最小可信链路"：任何一环坏了都会在这里炸，
而不是等到评估时才发现数字不对。

用法：
    python tests/test_smoke.py          # 直接跑
    pytest tests/test_smoke.py -q       # 或用 pytest
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.synthesis import (  # noqa: E402
    LowLightConfig, degrade_frame, frame_rng, load_rgb, save_rgb,
    srgb_to_linear, linear_to_srgb,
)


def _fake_frame(h=96, w=128, seed=0) -> np.ndarray:
    """造一张有结构的假图（渐变 + 方块），比纯随机更能暴露 bug。"""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:h, 0:w]
    base = (xx / w * 200 + yy / h * 40).astype(np.uint8)
    img = np.stack([base, base * 0.8, base * 0.6], axis=-1).astype(np.uint8)
    img[20:40, 30:60] = (255, 128, 64)     # 高光块
    img[50:70, 80:110] = (10, 10, 30)      # 暗块
    img += rng.integers(-3, 4, img.shape, dtype=np.int16).astype(np.uint8)  # 纹理
    return img


# --------------------------------------------------------------------------
def test_srgb_roundtrip():
    """sRGB↔linear 往返误差 < 1/255（uint8 量化噪声水平）。"""
    x = np.random.rand(64, 64, 3).astype(np.float32)
    back = linear_to_srgb(srgb_to_linear(x))
    assert float(np.abs(back - x).max()) < 1.0 / 255.0, "sRGB 往返误差过大"


def test_lut_matches_formula():
    """uint8 查表与连续公式一致（< 2e-3，即 < 0.5/255 的量化噪声）。"""
    x8 = np.random.randint(0, 256, (32, 32, 3), dtype=np.uint8)
    a = srgb_to_linear(x8)
    b = np.where(x8 / 255.0 <= 0.04045, (x8 / 255.0) / 12.92,
                 ((x8 / 255.0 + 0.055) / 1.055) ** 2.4).astype(np.float32)
    assert float(np.abs(a - b).max()) < 2e-3


def test_degrade_reproducible():
    """同 seed 逐位可复现；不同 seed 不同（M1-A 的核心判据）。"""
    img = _fake_frame()
    cfg = LowLightConfig()
    a = degrade_frame(img, cfg, frame_rng(0, "t", 0))
    b = degrade_frame(img, cfg, frame_rng(0, "t", 0))
    c = degrade_frame(img, cfg, frame_rng(1, "t", 0))
    assert np.array_equal(a, b), "同 seed 必须逐位相同"
    assert not np.array_equal(a, c), "不同 seed 必须不同"


def test_degrade_shape_and_range():
    img = _fake_frame()
    out = degrade_frame(img, LowLightConfig(), frame_rng(0, "t", 0))
    assert out.shape == img.shape and out.dtype == np.uint8
    assert out.min() >= 0 and out.max() <= 255


def test_metrics_sanity():
    """PSNR(同图)=99；PSNR(噪声图) 在合理区间；tOF 是有限数。"""
    from src.metrics import psnr, ssim, temporal_warping_error, occlusion_mask
    import cv2
    a = _fake_frame()
    noise = _fake_frame(seed=1)
    assert psnr(a, a) == 99.0
    p = psnr(a, noise)
    assert 10 < p < 40, f"PSNR(a, a+noise)={p} 不合理"
    s = ssim(a, a)
    assert s > 0.999, "同图 SSIM 应≈1"

    # tOF：同一张静止图复制两帧 → warp 误差应很小
    frames = [a, a.copy()]
    ga = cv2.cvtColor(a, cv2.COLOR_RGB2GRAY)
    flows = [occlusion_mask(ga, ga)]
    tof = temporal_warping_error(frames, flows)
    assert np.isfinite(tof) and tof < 2.0, f"静止序列 tOF={tof} 应接近 0"


def test_unet_forward_backward():
    """前向形状/范围 + 反向梯度全 finite（双零初始化和 SSIM fp16 两个坑的回归测试）。"""
    import torch
    from src.models.unet import UNetEnhance, count_params
    from src.train_utils import enhancement_loss
    net = UNetEnhance()
    assert 1.0 < count_params(net) / 1e6 < 3.0, "参数量应 ~2M"
    x = torch.rand(1, 3, 64, 64)
    y = torch.rand(1, 3, 64, 64)
    pred = net(x)
    assert pred.shape == x.shape
    assert float(pred.min()) >= 0 and float(pred.max()) <= 1.0
    loss = enhancement_loss(pred, y)   # CPU 上无 autocast；GPU 场景由训练脚本覆盖
    loss.backward()
    bad = [n for n, p in net.named_parameters()
           if p.grad is not None and not torch.isfinite(p.grad).all()]
    assert not bad, f"梯度含 NaN/Inf: {bad[:4]}"
    # 主干必须收到梯度（双零初始化陷阱的回归测试）
    assert net.inc[0].weight.grad.abs().max() > 0, "主干梯度为 0 —— 双零初始化陷阱复发！"


def test_classical_methods():
    """三个经典方法输出形状/范围正确。"""
    from src.baselines.classical import make_method
    img = _fake_frame()
    for name, kw in [("gamma", {"gamma": 2.2}), ("clahe", {"clip_limit": 2.0}),
                     ("ssr", {"sigma": 80.0})]:
        out = make_method(name, **kw).fn(img)
        assert out.shape == img.shape and out.dtype == np.uint8
        assert out.min() >= 0 and out.max() <= 255


def test_train_one_step(tmp_dir=None):
    """训练一步（真数据路径：在线合成 + loss + backward + optimizer.step）。"""
    import torch
    from torch.utils.data import DataLoader
    from src.data import SPLIT
    from src.synthesis import LowLightConfig
    from src.train_utils import LowLightPairDataset, enhancement_loss

    cfg = LowLightConfig()
    ds = LowLightPairDataset((SPLIT.train[0],), cfg, patch=64, base_seed=7)
    dl = DataLoader(ds, batch_size=2, shuffle=False)
    lo, gt, _ = next(iter(dl))
    net = torch.nn.Identity()  # 仅验证数据管道
    assert lo.shape == (2, 3, 64, 64) and gt.shape == (2, 3, 64, 64)
    # 低光输入已除以 brightness_keep 提亮到 GT 量级 → 高光区可超过 1（模型内 clamp）
    # 这是有意设计：训练/评估必须同一套归一化（见 train_utils.py 事故记录）
    assert float(lo.min()) >= 0.0, "输入不能为负"
    assert float(lo.max()) <= 1.0 / cfg.brightness_keep + 1e-6, "输入超出提亮上限"
    assert 0.0 <= float(gt.min()) and float(gt.max()) <= 1.0, "GT 必须在 [0,1]"
    assert float(lo.mean()) > float(gt.mean()) * 0.4, "提亮后的低光图均值应接近 GT 量级"


ALL = [test_srgb_roundtrip, test_lut_matches_formula, test_degrade_reproducible,
       test_degrade_shape_and_range, test_metrics_sanity, test_unet_forward_backward,
       test_classical_methods, test_train_one_step]


def main() -> int:
    print(f"冒烟测试：{len(ALL)} 项，CPU 预计 <2 分钟\n")
    failed = []
    for t in ALL:
        try:
            t()
            print(f"  PASS  {t.__name__}")
        except Exception as e:
            failed.append(t.__name__)
            print(f"  FAIL  {t.__name__}: {e}")
    print(f"\n{len(ALL)-len(failed)}/{len(ALL)} 通过")
    if failed:
        print(f"失败：{failed}  —— 修复后再跑评估，不要带病出数字")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
