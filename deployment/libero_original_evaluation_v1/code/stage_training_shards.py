"""把移动硬盘上的全量 OXE 数据暂存为内置盘上的小型训练工作集。

完整数据仍保留在原位置；本脚本只复制每个数据集最前面的若干个 train
TFRecord 分片，不删除源文件，也不会覆盖尺寸相同的已完成副本。
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


DEFAULT_SOURCES = (
    ("bridge_v2", Path(r"E:\dataset\bridge_v2_0.0.1\0.0.1")),
    ("language_table_real", Path(r"E:\dataset\language_table_0.1.0\0.1.0")),
    (
        "bc_z",
        Path(r"D:\ntu_related\dissertation\dataset\OpenX\bc_z_1.0.0\1.0.0"),
    ),
    (
        "rt1_fractal",
        Path(
            r"D:\ntu_related\dissertation\dataset\OpenX\fractal20220817_0.1.0\0.1.0"
        ),
    ),
)


def train_shards(source: Path) -> list[Path]:
    files = sorted(source.glob("*train*.tfrecord*"))
    if not files:
        files = sorted(source.glob("*.tfrecord*"))
    return files


def copy_atomic(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and destination.stat().st_size == source.stat().st_size:
        print(f"[缓存] 已存在，跳过：{destination.name}")
        return
    temporary = destination.with_suffix(destination.suffix + ".part")
    if temporary.exists():
        temporary.unlink()
    print(
        f"[缓存] 正在复制 {source.name} "
        f"({source.stat().st_size / 1024**2:.1f} MB)"
    )
    shutil.copy2(source, temporary)
    temporary.replace(destination)


def stage(args: argparse.Namespace) -> None:
    output = Path(args.output_dir).resolve()
    selections: list[tuple[str, Path, list[Path]]] = []
    total_bytes = 0
    for label, source in DEFAULT_SOURCES:
        if not source.is_dir():
            raise FileNotFoundError(f"找不到 {label} 数据目录：{source}")
        selected = train_shards(source)[: args.shards_per_dataset]
        if not selected:
            raise FileNotFoundError(f"{source} 下没有找到 TFRecord 分片")
        selections.append((label, source, selected))
        total_bytes += sum(path.stat().st_size for path in selected)

    usage_path = output
    while not usage_path.exists():
        usage_path = usage_path.parent
    free_bytes = shutil.disk_usage(usage_path).free
    print("=" * 68)
    print(f"[缓存] 输出目录：{output}")
    print(f"[缓存] 每个数据集分片数：{args.shards_per_dataset}")
    print(f"[缓存] 预计复制：{total_bytes / 1024**3:.2f} GB")
    print(f"[缓存] 当前可用空间：{free_bytes / 1024**3:.2f} GB")
    if free_bytes < total_bytes + args.reserve_gb * 1024**3:
        raise RuntimeError(
            f"空间不足：复制后必须至少保留 {args.reserve_gb:.1f} GB。"
        )
    if args.dry_run:
        for label, source, selected in selections:
            print(f"[预览] {label}: {source} -> {len(selected)} 个分片")
        return

    output.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, object] = {
        "format": "oxe_local_shard_cache_v1",
        "shards_per_dataset": args.shards_per_dataset,
        "datasets": {},
    }
    for label, source, selected in selections:
        destination_dir = output / label
        destination_dir.mkdir(parents=True, exist_ok=True)
        for metadata_name in ("dataset_info.json", "features.json"):
            metadata = source / metadata_name
            if metadata.exists():
                copy_atomic(metadata, destination_dir / metadata.name)
        for shard in selected:
            copy_atomic(shard, destination_dir / shard.name)
        manifest["datasets"][label] = {
            "source": str(source),
            "files": [path.name for path in selected],
            "bytes": sum(path.stat().st_size for path in selected),
        }

    (output / "cache_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("[缓存] 完成。之后训练只读取内置盘缓存，不访问移动硬盘。")
    print(f"[缓存] 训练数据根目录：{output}")


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="建立 OXE 内置盘训练分片缓存")
    parser.add_argument(
        "--output-dir", default=str(project_root / "training_cache" / "oxe_core")
    )
    parser.add_argument(
        "--shards-per-dataset",
        type=int,
        default=2,
        help="每个核心数据集复制多少个 train 分片；试跑建议 2，正式实验可改 8。",
    )
    parser.add_argument("--reserve-gb", type=float, default=20.0)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.shards_per_dataset <= 0:
        parser.error("--shards-per-dataset 必须大于 0")
    return args


if __name__ == "__main__":
    stage(parse_args())
