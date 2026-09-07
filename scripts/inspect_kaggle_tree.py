import os
from pathlib import Path
from collections import Counter
import pickle


def print_tree(root: Path, prefix: str = "", max_entries: int = 100) -> list[Path]:
    root = Path(root)
    if not root.exists():
        print(f"ERROR: {root} does not exist")
        return []
    if root.is_file():
        print(f"{prefix}└──{root.name}")
        return [root]
    if not root.is_dir():
        raise TypeError("Invalid Argument type for 'dir'.'")

    entries = sorted(root.iterdir())
    len_entries = len(entries)

    if len(entries) > max_entries:
        print(f"{prefix}└── {root.name} [{len(entries)} entries exceeds filelimit, not opening dir]")
        return entries

    for i, file in enumerate(entries):
        is_last: bool = i == len_entries - 1
        connector: str = "└── " if is_last else "├── "
        print(f"{prefix}{connector}{file.name}")

        if file.is_dir():
            extension: str = "    " if is_last else "│   "
            entries += print_tree(file, prefix + extension)
            # print_tree(file, prefix + extension)

    return entries


def get_filestats(files: list[Path]) -> dict:
    print("── 1. DIRECTORY STATS ──\n")
    all_files: list[Path] = []
    dir_counts: Counter[str] = Counter()
    for p in files:
        if p.is_file():
            all_files.append(p)
            dir_counts[str(p.parent)] += 1

    for d, count in sorted(dir_counts.items()):
        print(f"  {d + '/':.<50} {count:>5} files")
    print(f"\n  TOTAL: {len(files)} files\n")

    if not files:
        print("  No files found. Exiting.")
        return {}

    # ── 2. Extension histogram ──
    print("── 2. FILE EXTENSIONS ──\n")
    ext_counts: Counter[str] = Counter()
    for f in files:
        ext = f.suffix.lower() if f.suffix else "(no extension)"
        ext_counts[ext] += 1

    for ext, count in ext_counts.most_common():
        print(f"  {ext:.<40} {count:>6} files")
    print()

    # ── 3. File sizes per extension ──
    print("── 3. FILE SIZES BY EXTENSION ──\n")
    ext_sizes: dict[str, list[int]] = {}
    for f in files:
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

    return {"extensions": ext_counts, "tailles": ext_sizes, "dir": dir_counts}


def main() -> None:

    _BASE_DIR = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
    TARGET = "kaggle"
    IS_KAGGLE = bool(os.getenv("KAGGLE_KERNEL_RUN_TYPE"))
    root_dir = _BASE_DIR

    if IS_KAGGLE:
        try:
            while root_dir.name != TARGET:
                root_dir = root_dir.parent
        except Exception as e:
            print(f"Target '{TARGET}' is not reachable")
            print(repr(e))
            root_dir = _BASE_DIR

    entries = print_tree(root_dir)
    result = get_filestats(entries)

    if IS_KAGGLE:
        out_path = Path(TARGET) / "working" / "result.pkl"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("wb") as out:
            pickle.dump(result, out)


if __name__ == "__main__":
    main()
