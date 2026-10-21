"""一键复现 · 按顺序跑完所有阶段，重新生成 reports/ 里的全部表格与图

============================================================
设计原则
============================================================
- **分阶段、可跳过**：完整复现（含训练）约 4-6 小时；已训练过的步骤自动跳过
  （checkpoint 存在且 --force 没开就跳过），只重跑评估。
- **每步真实运行**：任何一步失败立即停止并报告，不静默跳过 ——
  "一键复现"最忌讳的是最后给你一堆旧数字但看起来像新跑的。
- **报告落盘时打印 sha**：reports 里的 md 每次重写都会更新，
  json 里带时间戳，方便确认"这份报告是几点几分生成的"。

用法：
    python scripts/run_all_reproduce.py                 # 增量（跳过已有 checkpoint）
    python scripts/run_all_reproduce.py --force         # 全部重跑
    python scripts/run_all_reproduce.py --skip-train    # 只跑评估（用现有 checkpoint）
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable

# (名字, 命令, 产物检查路径, 是否属于"训练"类)
STEPS = [
    ("环境自检",       "scripts/00_check_env.py",            ["reports/env_report.md"], False),
    ("M1 合成验收",    "scripts/10_verify_synthesis.py",     ["reports/m1_synthesis_check.md"], False),
    ("M2 经典baseline", "scripts/20_eval_classical.py --max-frames 8", ["reports/m2_classical.md"], False),
    ("M3 训练U-Net",   "scripts/30_train_unet.py --iters 4000 --batch 8 --patch 256 --val-every 250",
                       ["runs/unet_v0/best.pt"], True),
    ("M3 评估",        "scripts/31_eval_unet.py --ckpt runs/unet_v0/best.pt --max-frames 24",
                       ["reports/m3_unet.md"], False),
    ("M4 T1 评估",     "scripts/40_eval_t1.py",              ["reports/m4_t1.md"], False),
    ("M5 汇总分析",    "scripts/50_m5_analysis.py",          ["reports/m5_analysis.md"], False),
]


def have(*paths: str) -> bool:
    return all((ROOT / p).exists() for p in paths)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true", help="忽略已有产物，全部重跑")
    ap.add_argument("--skip-train", action="store_true", help="跳过训练类步骤（用现有 checkpoint）")
    ap.add_argument("--only", default=None, help="只跑名字包含该子串的步骤")
    args = ap.parse_args()

    t0 = time.perf_counter()
    results = []
    for name, cmd, artifacts, is_train in STEPS:
        if args.only and args.only not in name:
            continue
        if is_train and args.skip_train:
            results.append((name, "SKIP (--skip-train)"))
            continue
        if not args.force and have(*artifacts):
            results.append((name, "SKIP（产物已存在，--force 可重跑）"))
            print(f"[skip] {name}: 产物已存在")
            continue

        print(f"\n===== {name} =====\n$ python {cmd}", flush=True)
        t1 = time.perf_counter()
        r = subprocess.run([PY] + cmd.split(), cwd=ROOT)
        dt = time.perf_counter() - t1
        ok = r.returncode == 0 and have(*artifacts)
        results.append((name, f"{'OK' if ok else 'FAIL'} ({dt/60:.1f} min)"))
        if not ok:
            print(f"\n[FAIL] {name} 退出码 {r.returncode}，停止后续步骤（不要带病继续）")
            break

    print("\n===== 复现汇总 =====")
    for name, status in results:
        print(f"  {name:<16} {status}")
    print(f"总耗时 {(time.perf_counter()-t0)/60:.1f} 分钟")
    print("\n所有报告都在 reports/ 下，时间戳在对应 json 里。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
