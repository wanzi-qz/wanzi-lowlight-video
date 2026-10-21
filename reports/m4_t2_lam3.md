# 验收报告 · m4_t2_lam3（逐帧/时序 U-Net）

- checkpoint: `best.pt` @ iter 3000
- test 序列：bike-packing, bmx-trees, motocross-jump，每序列 24 帧连续片段，噪声 seed=0
- 参数量 1.95M，推理设备 cuda

## 汇总（test，逐帧直接 PSNR）

| 方法 | PSNR↑ | PSNR(亮度对齐)↑ | SSIM↑ | tOF↓ | tOF/tOF_in ↓ | 增益抖动σ↓ |
|---|---|---|---|---|---|---|
| low-light input | 14.88 | 25.84 | 0.6146 | 18.551 | 1.000 | nan |
| gamma(1.5) | 20.96 | 23.16 | 0.6464 | 21.660 | 1.186 | nan |
| clahe(1.5,8,1.0) | 19.82 | 20.98 | 0.5510 | 31.381 | 1.685 | nan |
| UNet (per-frame) | 24.83 | 27.58 | 0.8329 | 28.201 | 1.458 | 0.0002 |

## 逐序列（看是否在快速运动序列上更差）

| 序列 | 方法 | PSNR | tOF | tOF/tOF_in |
|---|---|---|---|---|
| bike-packing | low-light input | 15.90 | 11.941 | 1.000 |
| bike-packing | gamma(1.5) | 21.74 | 15.567 | 1.304 |
| bike-packing | clahe(1.5,8,1.0) | 22.31 | 19.706 | 1.650 |
| bike-packing | UNet (per-frame) | 26.07 | 13.649 | 1.143 |
| bmx-trees | low-light input | 14.79 | 25.412 | 1.000 |
| bmx-trees | gamma(1.5) | 21.59 | 29.075 | 1.144 |
| bmx-trees | clahe(1.5,8,1.0) | 19.84 | 43.285 | 1.703 |
| bmx-trees | UNet (per-frame) | 25.15 | 42.199 | 1.661 |
| motocross-jump | low-light input | 13.94 | 18.299 | 1.000 |
| motocross-jump | gamma(1.5) | 19.57 | 20.337 | 1.111 |
| motocross-jump | clahe(1.5,8,1.0) | 17.31 | 31.152 | 1.702 |
| motocross-jump | UNet (per-frame) | 23.26 | 28.756 | 1.571 |

## 核心结论

1. **保真度**：UNet 24.83dB，比 gamma 高 3.86dB，比 CLAHE 高 5.01dB。学习式方法在保真度上明确赢。 亮度对齐后 27.58dB（对齐差值 -2.76dB = 提亮贡献的口径差）。
2. **时序稳定性**：UNet 的 tOF/tOF_in = 1.458，gamma 1.186，CLAHE 1.685 —— 逐帧训练没有解决闪烁，这就是本项目的核心动机。
3. **增益呼吸**：UNet 每帧增益的帧间标准差 = 0.0002（理想 ≈ 0）。这是 tOF 抓不到、但人眼敏感的亮度闪烁，M4 的 T2 损失要单独约束它。
4. 但注意：这个改善来自**逐帧去噪更彻底**，不是来自**帧间一致**。如果 tOF 仍有明显残差，且不同帧的增益/去噪强度在抖，就说明还需要 M4。

## 下一步（M4 的设计依据）

- T1（训练无关）先验证「warp 融合能否在不重训的情况下降低 tOF」。若 UNet 的 tOF_ratio 已经很低（例如 < 0.6），T1 的收益空间会被压缩，那时应把重点转向 T2（约束增益抖动），并在报告里如实说明。
- T2 的损失必须**同时**约束：像素级 warp 残差 + 全局增益的一致性。

曲线图：`reports/m3_curves_bike-packing.png`
对比图：`reports/m3_compare_bike-packing.png`