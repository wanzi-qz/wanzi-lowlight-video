# 环境自检报告

- 生成时间：2026-10-21 10:21:16
- 状态：**READY (with TODO)**
- Python：3.12.3（`W:\Anaconda\python.exe`）
- 系统：Windows-11-10.0.26300-SP0

## PyTorch / GPU

- torch 版本：2.5.1
- CUDA 编译版本：cpu-only
- CUDA 可用：**False**
- 设备：—
- 显存：— GB
- CPU 线程数：8

```
NVIDIA GeForce RTX 4060 Laptop GPU, 616.92, 8188 MiB, 1829 MiB
```

## Python 包

| 包 | 状态 | 版本 | 必需 |
|---|---|---|---|
| numpy | OK | 1.26.4 | 是 |
| opencv-python | OK | 4.10.0 | 是 |
| torch | OK | 2.5.1 | 是 |
| torchvision | OK | 0.20.1 | 否 |
| scipy | OK | 1.13.1 | 否 |
| scikit-image | OK | 0.25.2 | 否 |
| lpips | MISSING | - | 否 |
| pyyaml | OK | 6.0.1 | 否 |
| tqdm | OK | 4.66.5 | 否 |

## 工具 / 磁盘 / 数据

- ffmpeg：`not found`
- git：`W:\WorkBuddy\resources\vendor\PortableGit\mingw64\bin\git.EXE`
- 磁盘：剩余 198.4 GB / 共 475.3 GB
- data/ 子目录：（空）

## 待办

1. torch 是 CPU 版（cuda_available=False），但机器有 GPU → 建议装 cu121 版，训练快 10-20 倍
2. ffmpeg 未找到（本方案用 DAVIS 的 JPEG 帧序列，可不需要；若要导出 mp4 对比视频再装）
3. data/ 为空 → M0 下一步下载 DAVIS-2017 480p（约 800MB）
