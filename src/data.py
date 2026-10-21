"""数据划分 · 按序列（不是帧）划分 train / val / test

============================================================
为什么必须按序列划分？
============================================================
DAVIS 同一条视频里，相邻帧几乎是同一张图。如果按帧随机划分，
val 里某帧的"孪生兄弟"大概率在 train 里 → val 指标虚高，
你会在论文里写下一个自己都无法复现的数字。

序列级划分后，val 里的场景（狗、滑雪、快艇）在 train 里完全没见过。
这才是"泛化"该有的难度。

============================================================
划分方案（固定下来，写进论文）
============================================================
- DAVIS-2017 官方 train(60) / val(30) 两套，本项目取 13 条序列的子集：
  train 8 / val 2 / test 3（另加 1 条真实直方图校准用的序列，不参与指标汇总）
- 挑选原则：
  1) 运动速度分层（慢/中/快都有），否则 tOF 的绝对值只反映"平均运动量"
  2) train 覆盖 indoor/outdoor、动物/人/车/物体
  3) test 必须包含 ROADMAP 里点名的失败案例序列（bike-packing / bmx-trees）
- 硬约束：test 序列**绝不**出现在 train/val 里（脚本会自检）

⚠ 已知的诚实性隐患
DAVIS 的 13 条序列来自 90 条，它们的**外观分布**（1080p 手机/相机拍摄的白天视频）
和"真实夜间视频"仍不是一个分布 —— 见 reports 里"真实噪声校准"一节的说明。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

DAVIS_ROOT = Path("data/DAVIS/JPEGImages/480p")


@dataclass(frozen=True)
class Split:
    train: tuple[str, ...]
    val: tuple[str, ...]
    test: tuple[str, ...]
    calibrate: tuple[str, ...]  # 只用于统计真实传感器噪声，不参与指标汇总

    def all_metric_sequences(self) -> tuple[str, ...]:
        return self.val + self.test


# 运动速度标注（用于报告中"按运动量分组"的分析，不是官方标注，是本项目手动归的）
MOTION = {
    "bear": "slow",
    "cows": "slow",
    "classic-car": "slow",
    "flamingo": "slow",
    "koala": "slow",
    "color-run": "mid",
    "libby": "mid",
    "parkour": "mid",
    "car-turn": "mid",
    "soccerball": "fast",
    "bike-packing": "fast",
    "bmx-trees": "fast",
    "motocross-jump": "fast",
}


SPLIT = Split(
    # 8 条：4 慢 + 3 中 + 1 快。覆盖动物/人/车/静物
    train=("bear", "cows", "flamingo", "koala",
           "color-run", "libby", "parkour", "car-turn"),
    # 2 条：一条极慢（便于看清噪声本身），一条含手持抖动（贴近真实拍摄）
    val=("classic-car", "soccerball"),
    # 3 条：两条快速运动（失败案例）+ 一条中速作为"正常情况"
    test=("bike-packing", "bmx-trees", "motocross-jump"),
    # 真实传感器噪声统计用（本项目用一张真实夜景图，见 scripts/30_calibrate_noise.py）
    calibrate=(),
)


# DAVIS 每帧的分辨率不同（480p 高度固定为 480，宽度不一）。
# 训练时统一裁剪/缩放到 256×256，评估时**保持原分辨率**并在两侧裁掉不可整除的列，
# 这样报告里的 PSNR 是对全画幅算的，不是对某个随机 crop 算的（后者会方差巨大）。
TRAIN_PATCH = 256
CROP_ALIGN = 8  # 宽高都补/裁到 8 的倍数，避免 U-Net 下采样后尺寸不匹配


def list_frames(root: Path, seq: str) -> list[Path]:
    d = root / seq
    if not d.is_dir():
        raise FileNotFoundError(f"序列不存在: {d}")
    files = sorted(d.glob("*.jpg"))
    if not files:
        raise FileNotFoundError(f"序列里没有 jpg: {d}")
    return files


def check_split(root: Path = DAVIS_ROOT) -> dict:
    """自检：无重叠、序列存在、帧数统计。任何一项不过就抛错（不要静默继续）。"""
    groups = {"train": SPLIT.train, "val": SPLIT.val, "test": SPLIT.test}
    seen: dict[str, str] = {}
    for gname, seqs in groups.items():
        for s in seqs:
            if s in seen:
                raise ValueError(f"序列 {s} 同时出现在 {seen[s]} 和 {gname}")
            seen[s] = gname

    stats: dict = {}
    for gname, seqs in groups.items():
        stats[gname] = {}
        for s in seqs:
            n = len(list_frames(root, s))
            stats[gname][s] = {"frames": n, "motion": MOTION.get(s, "?")}
    stats["_total_frames"] = sum(stats[g][s]["frames"] for g in groups for s in stats[g])
    return stats
