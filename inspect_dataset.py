#!/usr/bin/env python3
"""Diagnostic script to inspect Kaggle dataset structure before training.

Run once before training to understand what's being fed to build_memmap.
No side effects — read-only, prints to stdout only.

Usage:
    python inspect_dataset.py
    python inspect_dataset.py --input-dir /kaggle/input/other-dataset
"""

import argparse

from collections import Counter
from pathlib import Path


def inspect(input_dir: Path):
    print(f"{'=' * 70}")
    print(f"  DATASET INSPECTOR: {input_dir}")
    print(f"{'=' * 70}\n")

    if not input_dir.exists():
        print(f"ERROR: {input_dir} does not exist")
        return

    # ── 1. Directory tree ──
    print("── 1. DIRECTORY TREE ──\n")
    all_files: list[Path] = []
    dir_counts: Counter[str] = Counter()
    for p in sorted(input_dir.rglob("*")):
        if p.is_file():
            all_files.append(p)
            rel = p.relative_to(input_dir)
            dir_counts[str(rel.parent)] += 1

    for d, count in sorted(dir_counts.items()):
        print(f"  {d + '/':.<50} {count:>5} files")
    print(f"\n  TOTAL: {len(all_files)} files\n")

    if not all_files:
        print("  No files found. Exiting.")
        return

    # ── 2. Extension histogram ──
    print("── 2. FILE EXTENSIONS ──\n")
    ext_counts: Counter[str] = Counter()
    for f in all_files:
        ext = f.suffix.lower() if f.suffix else "(no extension)"
        ext_counts[ext] += 1

    for ext, count in ext_counts.most_common():
        print(f"  {ext:.<40} {count:>6} files")
    print()

    # ── 3. File sizes per extension ──
    print("── 3. FILE SIZES BY EXTENSION ──\n")
    ext_sizes: dict[str, list[int]] = {}
    for f in all_files:
        ext = f.suffix.lower() if f.suffix else "(no extension)"
        ext_sizes.setdefault(ext, []).append(f.stat().st_size)

    for ext in sorted(ext_sizes):
        sizes = ext_sizes[ext]
        total = sum(sizes)
        median = sorted(sizes)[len(sizes) // 2]
        print(f"  {ext}:")
        print(f"    count:  {len(sizes)}")
        print(f"    min:    {min(sizes):>12,} bytes")
        print(f"    median: {median:>12,} bytes")
        print(f"    max:    {max(sizes):>12,} bytes")
        print(f"    total:  {total:>12,} bytes ({total / 1024 / 1024:.1f} MB)")
    print()

    # ── 4. Sample reads ──
    print("── 4. SAMPLE READS (first 500 chars) ──\n")
    by_ext: dict[str, list[Path]] = {}
    for f in all_files:
        ext = f.suffix.lower() if f.suffix else "(no extension)"
        by_ext.setdefault(ext, []).append(f)

    for ext in sorted(by_ext):
        samples = by_ext[ext][:3]
        print(f"  [{ext}]")
        for s in samples:
            print(f"    --- {s.name} ({s.stat().st_size:,} bytes) ---")
            try:
                text = s.read_text(encoding="utf-8", errors="replace")[:500]
                print(f"    {repr(text[:200])}")
            except Exception as e:
                print(f"    ERROR reading: {e}")
        print()

    # ── 5. allowed_ext filter match ──
    print("── 5. FILTER MATCH (current allowed_ext) ──\n")
    allowed_ext = (".txt", ".md", ".gz", ".csv")
    matched = [f for f in all_files if f.suffix.lower() in allowed_ext]
    skipped = [f for f in all_files if f.suffix.lower() not in allowed_ext]

    print(f"  allowed_ext: {allowed_ext}")
    print(f"  MATCHED:     {len(matched)} files ({len(matched) / len(all_files) * 100:.1f}%)")
    print(f"  SKIPPED:     {len(skipped)} files ({len(skipped) / len(all_files) * 100:.1f}%)")

    if skipped:
        skipped_exts = Counter(f.suffix.lower() or "(no extension)" for f in skipped)
        print("\n  Skipped extensions:")
        for ext, count in skipped_exts.most_common():
            print(f"    {ext:.<30} {count:>5}")
    print()

    # ── 6. Total raw text estimate ──
    print("── 6. TOKEN ESTIMATE ──\n")
    total_bytes = sum(f.stat().st_size for f in all_files)

    est_tokens = total_bytes // 4
    print(f"  Total file bytes:     {total_bytes:>15,} ({total_bytes / 1024 / 1024:.1f} MB)")
    print(f"  Est. tokens (÷4):     {est_tokens:>15,} ({est_tokens / 1_000_000:.1f}M)")

    print("\n  With current filter (matched only):")
    print()

    print(f"{'=' * 70}")
    print("  CONCLUSION")
    print(f"{'=' * 70}")
    if len(skipped) > len(matched):
        print(f"  WARNING: {len(skipped)} files are SKIPPED by the current filter.")
        print("  This is likely why data_meta.json shows few tokens.")
        print("  Fix: add extensionless files to allowed_ext in build_memmap().")
    else:
        print("  Filter looks OK — most files are matched.")
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inspect dataset structure")
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=None,
        help="Dataset directory (default: auto-detect Kaggle or local text/)",
    )
    args = parser.parse_args()

    if args.input_dir:
        input_dir = args.input_dir
    elif Path("/kaggle/input").exists():
        # List available datasets and pick the first one
        input_dir = Path("/kaggle/input")
        print("Kaggle detected. Listing /kaggle/input contents:\n")
        for p in sorted(input_dir.iterdir()):
            print(f"  {p.name}/")
        print("\nUse --input-dir to target a specific dataset, e.g.:")
        print("  python inspect_dataset.py --input-dir /kaggle/input/llm-training-data")
    else:
        input_dir = Path(__file__).resolve().parent / "text"

    inspect(input_dir)
