# M2 验收报告 · 经典 baseline

- 低光配置：EV=2.5, Φ=800.0e-, s=0.35, seed=0
- 每序列帧数：8（连续片段，取序列中段）
- **参数只在 val 上选**（classic-car, soccerball），test 一次性使用：bike-packing, bmx-trees, motocross-jump

## val 选参结果

| 方法 | 最好参数 | val PSNR | val SSIM |
|---|---|---|---|
| gamma(g=1.5) | `{"gamma": 1.5}` | 21.04 | 0.6955 |
| clahe(clip=1.5,tile=8,g=1) | `{"clip_limit": 1.5, "tile": 8, "gamma": 1.0}` | 20.53 | 0.6549 |
| ssr(sigma=150,gain=1) | `{"sigma": 150.0, "gain": 1.0}` | 8.64 | 0.4555 |

## test 结果（3 条序列 × 8 帧）

| 方法 | PSNR↑ | SSIM↑ | LPIPS↓ | tOF↓ | tOF/tOF_in ↓ | ms/帧 |
|---|---|---|---|---|---|---|
| low-light input | 14.85 | 0.6255 | nan | 19.728 | 1.000 | 0 |
| gamma(g=1.5) | 21.07 | 0.6624 | nan | 22.667 | 1.188 | 175 |
| clahe(clip=1.5,tile=8,g=1) | 20.22 | 0.5750 | nan | 32.426 | 1.648 | 164 |
| ssr(sigma=150,gain=1) | 8.41 | 0.4754 | nan | 24.939 | 1.423 | 392 |

## 假设检验（开跑前写下的，事后照填）

- H1 「gamma 提亮后 PSNR 可能反而下降」：输入 14.85dB → 21.07dB，**不成立**
- H2 「CLAHE 的 SSIM 不一定最高」：最高 SSIM 是 **gamma(g=1.5)**（0.6624），CLAHE 0.5750，**成立**
- H3 「所有逐像素方法 tOF_ratio ≥ 0.8」：gamma(g=1.5)=1.188，clahe(clip=1.5,tile=8,g=1)=1.648，ssr(sigma=150,gain=1)=1.423 → **成立**

## 口径说明：本表的 PSNR 是「未做亮度对齐」的逐帧直接 PSNR

- 低光输入的亮度对齐 PSNR ≈ 25-26dB（见 M1 报告），但本表里只有 14.85dB ——
  这 ~10dB 的差距来自**整体变暗**，不是信息丢失。
- 因此「gamma 21dB」这类提升里，大部分是**把亮度拉回来**的功劳，去噪只占一小部分；
  M3（学习式 baseline）里会同时报告两种口径，把「提亮」和「去噪」拆开归因。

图：`reports/m2_classical_bike-packing.png`（上排：低光输入 → 各方法 → GT；下排：暗部 2× 放大）

> 注意：tOF/tOF_in 是本项目定义的归一化闪烁度。**≥1 表示「比不处理还抖」**，
> 因为它把噪声放大了。这个数字就是 M3/M4 要打败的对象。