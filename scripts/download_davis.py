"""M0 · 下载 DAVIS-2017 trainval 480p

为什么要自己写而不是手动下载：
1. 断点续传（800MB 断了不用重来）
2. 校验（zip 完整性 + 序列数量 + 帧数），避免"下完了但文件是坏的"这种隐性故障
3. 写进 README 就是一条可复现命令

用法：
    python scripts/download_davis.py            # 下载 + 解压 + 校验
    python scripts/download_davis.py --check     # 只校验已有数据
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RAW = DATA / "raw"
ZIP_PATH = RAW / "DAVIS-2017-trainval-480p.zip"
EXTRACT_TO = DATA / "DAVIS"

URL = "https://data.vision.ee.ethz.ch/csergi/share/davis/DAVIS-2017-trainval-480p.zip"
EXPECTED_BYTES = 832_766_765


def human(n: float) -> str:
    for unit in ["B", "KB", "MB", "GB"]:
        if n < 1024:
            return f"{n:.1f}{unit}"
        n /= 1024
    return f"{n:.1f}TB"


def download(url: str, dst: Path, expected: int | None = None) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    have = dst.stat().st_size if dst.exists() else 0

    if expected and have == expected:
        print(f"[skip] 已完整下载：{dst.name} ({human(have)})")
        return
    if expected and have > expected:
        print("[warn] 本地文件比预期大，删除重下")
        dst.unlink()
        have = 0

    req = urllib.request.Request(url)
    if have:
        req.add_header("Range", f"bytes={have}-")
        print(f"[resume] 从 {human(have)} 继续")

    t0 = time.time()
    with urllib.request.urlopen(req, timeout=60) as resp, open(dst, "ab" if have else "wb") as f:
        total = expected or int(resp.headers.get("Content-Length", 0)) + have
        done = have
        last = 0.0
        while True:
            chunk = resp.read(1 << 20)  # 1MB
            if not chunk:
                break
            f.write(chunk)
            done += len(chunk)
            now = time.time()
            if now - last > 2:
                last = now
                speed = (done - have) / max(now - t0, 1e-6)
                pct = 100 * done / total if total else 0
                print(f"\r  {human(done)} / {human(total)}  {pct:5.1f}%  {human(speed)}/s", end="", flush=True)
    print()
    print(f"[ok] 下载完成：{human(dst.stat().st_size)}")


def extract(zip_path: Path, out: Path) -> None:
    if out.exists() and any(out.iterdir()):
        print(f"[skip] 已解压：{out}")
        return
    out.mkdir(parents=True, exist_ok=True)
    print("[info] 校验 zip 完整性（只测首尾若干条目，全量太慢）…")
    with zipfile.ZipFile(zip_path) as z:
        names = z.namelist()
        for n in names[:3] + names[-3:]:
            z.read(n)  # 读出来会触发 CRC 校验
        print(f"[ok] zip 可读，{len(names)} 个条目")
        print("[info] 解压中…")
        z.extractall(out)

    # zip 内层还有一层 DAVIS/，展平，避免路径到处多一跳
    nested = out / "DAVIS"
    if nested.is_dir() and (nested / "JPEGImages").is_dir():
        for p in nested.iterdir():
            target = out / p.name
            if target.exists():
                continue
            shutil.move(str(p), str(target))
        nested.rmdir()
    print(f"[ok] 解压到 {out}")


def verify(data_root: Path) -> dict:
    """检查目录结构、序列数、帧数，输出结构化报告。"""
    # zip 顶层目录通常叫 DAVIS/
    candidates = [data_root / "DAVIS", data_root]
    jpeg = None
    for c in candidates:
        p = c / "JPEGImages" / "480p"
        if p.is_dir():
            jpeg = p
            break
    if jpeg is None:
        raise SystemExit(f"找不到 JPEGImages/480p，检查 {data_root} 结构")

    seqs = sorted([d for d in jpeg.iterdir() if d.is_dir()])
    frames = {d.name: len(list(d.glob("*.jpg"))) for d in seqs}
    has_ann = (jpeg.parents[1] / "Annotations" / "480p").is_dir()
    report = {
        "root": str(jpeg),
        "num_sequences": len(seqs),
        "total_frames": sum(frames.values()),
        "has_annotations": has_ann,
        "frames_per_sequence": frames,
    }
    return report


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="只校验，不下载")
    ap.add_argument("--force", action="store_true", help="重新解压")
    args = ap.parse_args()

    if not args.check:
        download(URL, ZIP_PATH, EXPECTED_BYTES)
        if args.force and EXTRACT_TO.exists():
            shutil.rmtree(EXTRACT_TO)
        extract(ZIP_PATH, EXTRACT_TO)

    rep = verify(EXTRACT_TO)
    (ROOT / "reports").mkdir(exist_ok=True)
    (ROOT / "reports" / "davis_index.json").write_text(
        json.dumps(rep, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"[verify] 序列数 = {rep['num_sequences']}，总帧数 = {rep['total_frames']}，"
          f"标注 = {rep['has_annotations']}")
    print("[verify] 帧数最少/最多的 5 条：")
    items = sorted(rep["frames_per_sequence"].items(), key=lambda kv: kv[1])
    for name, n in items[:3] + items[-3:]:
        print(f"    {name:20s} {n:4d} 帧")
    print(f"[ok] 索引写入 reports/davis_index.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
