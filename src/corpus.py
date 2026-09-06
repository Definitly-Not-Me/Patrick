"""Corpus building: file parsing, text cleaning, memmap generation."""

import gzip
import json
from pathlib import Path

import numpy as np
from tiktoken import Encoding


def parse_file(path: Path) -> str:
    """Read a file by extension. Returns "" for CSV or on error."""
    suffix = path.suffix.lower()

    if suffix == ".csv":
        return ""

    if suffix == ".gz":
        try:
            with gzip.open(path, "rt", encoding="utf-8", errors="ignore") as f:
                return f.read()
        except Exception:
            return ""

    try:
        return path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return ""


def clean_text(text: str) -> str:
    """Clip Gutenberg headers/footers between START/END markers."""
    start_marker = "*** START OF"
    end_marker = "*** END OF"

    start_idx = text.find(start_marker)
    end_idx = text.find(end_marker)

    if start_idx != -1 and end_idx != -1:
        start_line_end = text.find("\n", start_idx)
        if start_line_end != -1:
            start_idx = start_line_end + 1
        text = text[start_idx:end_idx]
    elif start_idx != -1:
        start_line_end = text.find("\n", start_idx)
        if start_line_end != -1:
            text = text[start_line_end + 1:]

    return text.strip()


def build_memmap(input_dir: Path, bin_path: Path, meta_path: Path, tokenizer: Encoding, allowed_ext: tuple[str] = (".txt", ".md", ".gz", ".csv")) -> int:
    """
    Tokenise all files under input_dir into a flat uint16 .bin with EOF separators.
    Returns total number of tokens written.
    """
    assert input_dir.is_dir(), "Input must be directory"

    eof_id = tokenizer._special_tokens["<|endoftext|>"]
    all_tokens: list[int] = []

    for file_path in sorted(input_dir.rglob("*")):
        if not file_path.is_file():
            continue
        if not file_path.name.endswith(allowed_ext):
            continue

        raw = parse_file(file_path)
        if not raw.strip():
            continue
        cleaned = clean_text(raw)
        if not cleaned.strip():
            continue
        ids = tokenizer.encode(cleaned, allowed_special={"<|endoftext|>"})
        all_tokens.extend(ids)
        all_tokens.append(eof_id)

    if not all_tokens:
        raise ValueError(f"No tokens produced from {input_dir}")

    bin_path.parent.mkdir(parents=True, exist_ok=True)
    np.array(all_tokens, dtype=np.uint16).tofile(str(bin_path))

    meta = {
        "num_tokens": len(all_tokens),
        "dtype": "uint16",
        "vocab_size": tokenizer.n_vocab,
        "eof_id": eof_id,
    }
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.write_text(json.dumps(meta), encoding="utf-8")

    return len(all_tokens)
