"""M1 可视化 · 把"逐帧独立噪声 = 闪烁"这件事做成能看的东西

产出：
  reports/preview_<seq>_strip.png   4 帧拼接：上排 GT，下排低光（肉眼可见噪点在跳）
  reports/preview_<seq>_cmp.mp4     上下分屏视频：GT / 低光（对齐亮度后），循环播放

为什么要做这个：
tOF 是个抽象数字，面试官/评审不会因为一个 0.007 的下降就相信"闪烁被抑制了"。
先让人**看见**闪烁，后面才有说服力。

用法：
    python scripts/11_make_preview_video.py --seq bike-packing --frames 40
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.synthesis import (  # noqa: E402
    LowLightConfig, degrade_frame, frame_rng, linear_to_srgb, load_rgb, save_rgb,
    srgb_to_linear,
)

REPORTS = ROOT / "reports"


def put_label(img: np.ndarray, text: str) -> np.ndarray:
    out = img.copy()
    cv2.rectangle(out, (0, 0), (len(text) * 11 + 16, 28), (0, 0, 0), -1)
    cv2.putText(out, text, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq", default="bike-packing")
    ap.add_argument("--frames", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    d = ROOT / "data" / "DAVIS" / "JPEGImages" / "480p" / args.seq
    files = sorted(d.glob("*.jpg"))[: args.frames]
    if not files:
        raise SystemExit(f"没有帧：{d}")

    cfg = LowLightConfig()
    gts, lls, aligned = [], [], []
    for t, f in enumerate(files):
        gt = load_rgb(f)
        ll = degrade_frame(gt, cfg, frame_rng(args.seed, args.seq, t))
        # 亮度对齐（乘 1/s），否则前后对比看不清楚细节，只能看到"暗"
        lin = srgb_to_linear(ll.astype(np.float32) / 255.0) / cfg.brightness_keep
        al = (linear_to_srgb(np.clip(lin, 0, 1)) * 255).astype(np.uint8)
        gts.append(gt)
        lls.append(ll)
        aligned.append(al)

    REPORTS.mkdir(exist_ok=True)
    h, w = gts[0].shape[:2]

    # 1) 4 帧拼接条：上 GT 下低光
    idx = [0, min(1, len(gts) - 1), min(2, len(gts) - 1), min(3, len(gts) - 1)]
    top = np.hstack([gts[i] for i in idx])
    bot = np.hstack([aligned[i] for i in idx])
    strip = np.vstack([put_label(top, "GT (ground truth)"), put_label(bot, "low-light, brightness-aligned")])
    strip_path = REPORTS / f"preview_{args.seq}_strip.png"
    save_rgb(strip_path, strip)  # 不能用 cv2.imwrite：中文路径会静默失败

    # 2) 分屏视频
    vid_path = REPORTS / f"preview_{args.seq}_cmp.mp4"
    vw = cv2.VideoWriter(str(vid_path), cv2.VideoWriter_fourcc(*"mp4v"), 10, (w, 2 * h))
    for i in range(len(gts)):
        a = put_label(gts[i], f"GT  frame {i}")
        b = put_label(aligned[i], f"degraded  frame {i}")
        vw.write(cv2.cvtColor(np.vstack([a, b]), cv2.COLOR_RGB2BGR))
    vw.release()

    # 3) 数字：相邻帧差（越大越闪）
    def frame_diff(seq_imgs):
        return float(np.mean([np.mean(np.abs(seq_imgs[i + 1].astype(np.float32) -
                                             seq_imgs[i].astype(np.float32)))
                              for i in range(len(seq_imgs) - 1)]))

    print(f"[seq] {args.seq}  帧数={len(gts)}  尺寸={w}x{h}")
    print(f"[flicker] 相邻帧差 GT={frame_diff(gts):.3f}  低光={frame_diff(aligned):.3f} "
          f"(0-255 尺度)")
    print(f"[out] {strip_path}")
    print(f"[out] {vid_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
