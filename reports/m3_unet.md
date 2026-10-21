# M3 验收报告 · 逐帧 U-Net baseline

- checkpoint: `best.pt` @ iter 2750
- test 序列：bike-packing, bmx-trees, motocross-jump，每序列 24 帧连续片段，噪声 seed=0
- 参数量 1.95M，推理设备 cuda

## 汇总（test，逐帧直接 PSNR）

| 方法 | PSNR↑ | PSNR(亮度对齐)↑ | SSIM↑ | tOF↓ | tOF/tOF_in ↓ | 增益抖动σ↓ |
|---|---|---|---|---|---|---|
| low-light input | 14.88 | 25.84 | 0.6146 | 18.551 | 1.000 | nan |
| gamma(1.5) | 20.96 | 23.15 | 0.6464 | 21.662 | 1.187 | nan |
| clahe(1.5,8,1.0) | 19.82 | 20.98 | 0.5510 | 31.374 | 1.685 | nan |
| UNet (per-frame) | 25.13 | 27.78 | 0.8118 | 28.901 | 1.504 | 0.0002 |

## 逐序列（看是否在快速运动序列上更差）

| 序列 | 方法 | PSNR | tOF | tOF/tOF_in |
|---|---|---|---|---|
| bike-packing | low-light input | 15.90 | 11.942 | 1.000 |
| bike-packing | gamma(1.5) | 21.74 | 15.572 | 1.304 |
| bike-packing | clahe(1.5,8,1.0) | 22.31 | 19.702 | 1.650 |
| bike-packing | UNet (per-frame) | 26.34 | 14.644 | 1.226 |
| bmx-trees | low-light input | 14.79 | 25.412 | 1.000 |
| bmx-trees | gamma(1.5) | 21.59 | 29.073 | 1.144 |
| bmx-trees | clahe(1.5,8,1.0) | 19.84 | 43.284 | 1.703 |
| bmx-trees | UNet (per-frame) | 25.27 | 42.719 | 1.681 |
| motocross-jump | low-light input | 13.94 | 18.297 | 1.000 |
| motocross-jump | gamma(1.5) | 19.57 | 20.340 | 1.112 |
| motocross-jump | clahe(1.5,8,1.0) | 17.29 | 31.137 | 1.702 |
| motocross-jump | UNet (per-frame) | 23.78 | 29.339 | 1.603 |

## 核心结论

1. **保真度**：UNet 25.13dB，比 gamma 高 4.16dB，比 CLAHE 高 5.31dB。学习式方法在保真度上明确赢。 亮度对齐后 27.78dB（对齐差值 -2.65dB = 提亮贡献的口径差）。
2. **时序稳定性**：UNet 的 tOF/tOF_in = 1.504，gamma 1.187，CLAHE 1.685 —— 逐帧训练没有解决闪烁，这就是本项目的核心动机。
3. **增益呼吸**：UNet 每帧增益的帧间标准差 = 0.0002（理想 ≈ 0）。这是 tOF 抓不到、但人眼敏感的亮度闪烁，M4 的 T2 损失要单独约束它。
4. 但注意：这个改善来自**逐帧去噪更彻底**，不是来自**帧间一致**。如果 tOF 仍有明显残差，且不同帧的增益/去噪强度在抖，就说明还需要 M4。

## 下一步（M4 的设计依据）

- T1（训练无关）先验证「warp 融合能否在不重训的情况下降低 tOF」。若 UNet 的 tOF_ratio 已经很低（例如 < 0.6），T1 的收益空间会被压缩，那时应把重点转向 T2（约束增益抖动），并在报告里如实说明。
- T2 的损失必须**同时**约束：像素级 warp 残差 + 全局增益的一致性。

曲线图：`reports/m3_curves_bike-packing.png`
对比图：`reports/m3_compare_bike-packing.png`