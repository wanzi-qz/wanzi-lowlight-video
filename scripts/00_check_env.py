"""M0 · 环境自检

用途：在任何代码之前，先搞清楚"这台机器能干什么"。
输出：reports/env_report.md（人读）+ reports/env_report.json（机读）

运行：
    python scripts/00_check_env.py

设计原则：只用标准库做必选检查，第三方库缺失只记 MISSING，不让脚本崩。
"""

from __future__ import annotations

import json
import platform
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
REPORTS.mkdir(exist_ok=True)

# 每项：(import 名, 显示名, 是否必需)
PACKAGES = [
    ("numpy", "numpy", True),
    ("cv2", "opencv-python", True),
    ("torch", "torch", True),
    ("torchvision", "torchvision", False),
    ("scipy", "scipy", False),
    ("skimage", "scikit-image", False),
    ("lpips", "lpips", False),      # M5 感知指标
    ("yaml", "pyyaml", False),      # 配置驱动
    ("tqdm", "tqdm", False),
]


def check_packages() -> list[dict]:
    rows = []
    for mod, name, required in PACKAGES:
        try:
            m = __import__(mod)
            ver = getattr(m, "__version__", "installed")
            rows.append({"name": name, "status": "OK", "version": ver, "required": required})
        except Exception:
            rows.append({"name": name, "status": "MISSING", "version": "-", "required": required})
    return rows


def check_torch() -> dict:
    info = {"installed": False, "version": None, "cuda_build": None,
            "cuda_available": False, "device": None, "vram_gb": None, "threads": None}
    try:
        import torch
    except Exception:
        return info
    info["installed"] = True
    info["version"] = torch.__version__
    info["cuda_build"] = torch.version.cuda or "cpu-only"
    info["cuda_available"] = bool(torch.cuda.is_available())
    info["threads"] = torch.get_num_threads()
    if info["cuda_available"]:
        info["device"] = torch.cuda.get_device_name(0)
        info["vram_gb"] = round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 1)
    return info


def check_nvidia_smi() -> str | None:
    if not shutil.which("nvidia-smi"):
        return None
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,driver_version,memory.total,memory.used",
             "--format=csv,noheader"],
            capture_output=True, text=True, timeout=15,
        )
        return out.stdout.strip() or None
    except Exception:
        return None


def check_tools() -> dict:
    tools = {}
    for t in ["ffmpeg", "git"]:
        tools[t] = shutil.which(t) or "not found"
    return tools


def check_disk() -> dict:
    total, used, free = shutil.disk_usage(ROOT)
    return {"total_gb": round(total / 1024**3, 1),
            "free_gb": round(free / 1024**3, 1)}


def check_data() -> dict:
    data = ROOT / "data"
    subdirs = [p.name for p in data.iterdir() if p.is_dir()] if data.exists() else []
    return {"data_dir_exists": data.exists(), "subdirs": subdirs}


def verdict(rep: dict) -> tuple[str, list[str]]:
    """给出人话结论 + 待办项。"""
    todo = []
    t = rep["torch"]
    if not t["installed"]:
        todo.append("torch 未安装 → 需要 `pip install torch --index-url https://download.pytorch.org/whl/cu121`")
    elif not t["cuda_available"]:
        todo.append("torch 是 CPU 版（cuda_available=False），但机器有 GPU → 建议装 cu121 版，训练快 10-20 倍")
    missing = [p["name"] for p in rep["packages"] if p["status"] == "MISSING" and p["required"]]
    if missing:
        todo.append(f"必需包缺失：{', '.join(missing)}")
    if rep["tools"]["ffmpeg"] == "not found":
        todo.append("ffmpeg 未找到（本方案用 DAVIS 的 JPEG 帧序列，可不需要；若要导出 mp4 对比视频再装）")
    if rep["disk"]["free_gb"] < 20:
        todo.append(f"磁盘剩余仅 {rep['disk']['free_gb']}GB，DAVIS 480p + 实验产物建议留 20GB+")
    if not rep["data"]["subdirs"]:
        todo.append("data/ 为空 → M0 下一步下载 DAVIS-2017 480p（约 800MB）")

    blocking = [x for x in todo if ("torch 未安装" in x or "必需包缺失" in x)]
    if blocking:
        return "BLOCKED", todo
    if todo:
        return "READY (with TODO)", todo
    return "READY", todo


def main() -> int:
    rep = {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "python": sys.version.split()[0],
        "executable": sys.executable,
        "platform": platform.platform(),
        "packages": check_packages(),
        "torch": check_torch(),
        "nvidia_smi": check_nvidia_smi(),
        "tools": check_tools(),
        "disk": check_disk(),
        "data": check_data(),
    }
    status, todo = verdict(rep)
    rep["status"] = status
    rep["todo"] = todo

    (REPORTS / "env_report.json").write_text(
        json.dumps(rep, indent=2, ensure_ascii=False), encoding="utf-8")

    t = rep["torch"]
    lines = [
        "# 环境自检报告",
        "",
        f"- 生成时间：{rep['generated_at']}",
        f"- 状态：**{status}**",
        f"- Python：{rep['python']}（`{rep['executable']}`）",
        f"- 系统：{rep['platform']}",
        "",
        "## PyTorch / GPU",
        "",
        f"- torch 版本：{t['version']}",
        f"- CUDA 编译版本：{t['cuda_build']}",
        f"- CUDA 可用：**{t['cuda_available']}**",
        f"- 设备：{t['device'] or '—'}",
        f"- 显存：{t['vram_gb'] or '—'} GB",
        f"- CPU 线程数：{t['threads']}",
        "",
        "```",
        rep["nvidia_smi"] or "nvidia-smi 不可用",
        "```",
        "",
        "## Python 包",
        "",
        "| 包 | 状态 | 版本 | 必需 |",
        "|---|---|---|---|",
    ]
    for p in rep["packages"]:
        lines.append(f"| {p['name']} | {p['status']} | {p['version']} | {'是' if p['required'] else '否'} |")
    lines += [
        "",
        "## 工具 / 磁盘 / 数据",
        "",
        f"- ffmpeg：`{rep['tools']['ffmpeg']}`",
        f"- git：`{rep['tools']['git']}`",
        f"- 磁盘：剩余 {rep['disk']['free_gb']} GB / 共 {rep['disk']['total_gb']} GB",
        f"- data/ 子目录：{rep['data']['subdirs'] or '（空）'}",
        "",
        "## 待办",
        "",
    ]
    lines += [f"{i}. {x}" for i, x in enumerate(todo, 1)] or ["（无）"]
    (REPORTS / "env_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"status={status}")
    print(f"torch={t['version']} cuda_build={t['cuda_build']} cuda_available={t['cuda_available']}")
    for x in todo:
        print("TODO:", x)
    print(f"report -> {REPORTS / 'env_report.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
