"""Corpus building: file parsing, text cleaning, memmap generation."""

import os
import sys
from pathlib import Path
import gzip
import json
import random
import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from tiktoken import Encoding
import tiktoken
import numpy as np
from pydantic import Field, BaseModel, model_validator


_BASE_DIR = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()

# Add src/ to path for imports (corpus, data modules)
_src_dir = str(_BASE_DIR / "src")
if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)

SCRIPT_DIR = _BASE_DIR
IS_KAGGLE = bool(os.getenv("KAGGLE_KERNEL_RUN_TYPE"))
WORK_DIR = Path("/kaggle/working") if IS_KAGGLE else SCRIPT_DIR
_KAGGLE_INPUT = Path("/kaggle/input")


class GPTDatasetV2(Dataset):
    """
    Memory maped token dataset.
    Assigne un dataset Pytorch a un fichier .bin produit par build_memmap.
    Assure que les données ne sont jamais entiérement chargés en mémoire.
    """

    def __init__(
        self,
        bin_path: str | Path,
        split_idx: tuple[int, int] = (0, -1),
        max_length: int = 1024,
        stride: int = 512,
    ) -> None:
        self.context_length = max_length
        self.stride = stride if stride else max_length

        data = np.memmap(str(bin_path), dtype=np.uint16, mode="r")
        assert data is not None, f"File at '{bin_path}' is empty"

        self.data = data[split_idx[0] : split_idx[1]]
        self.length = max(0, (len(self.data) - self.context_length - 1) // stride + 1)

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index) -> tuple[Tensor, Tensor]:
        start = index * self.stride
        end = start + self.context_length
        x = torch.from_numpy(np.array(self.data[start:end])).long()
        y = torch.from_numpy(np.array(self.data[start + 1 : end + 1])).long()

        return x, y


class GPTConfig(BaseModel):
    """
    Configuration du modèle. Par défaut ceux de GPT2-small
    """

    num_layers: int = Field(default=12)
    num_heads: int = Field(default=12, gt=1)
    context_length: int = Field(default=1024, gt=0, multiple_of=2)
    embeddings_dim: int = Field(default=768, gt=0)
    drop_rate: float = Field(default=0.1, lt=1)
    qvk_bias: bool = Field(default=False)
    vocab_size: int = Field(default=50257)
    temperature: float = Field(default=0.8, gt=0)
    top_k: int = Field(default=40)

    @model_validator(mode="after")
    def validate_dimensions(self):
        if self.embeddings_dim % self.num_heads != 0:
            raise ValueError("embeddings_dim must be divisible by num_heads")
        return self


GPT_medium = GPTConfig(embeddings_dim=1024, num_layers=24, num_heads=16)

GPT_large = GPTConfig(embeddings_dim=1280, num_layers=36, num_heads=20)


def parse_file(path: Path) -> str:
    suffix = path.suffix.lower()

    if suffix in (".csv", ".json", ".xml", ".html", ".parquet"):
        return ""

    if suffix == ".gz":
        try:
            with gzip.open(path, "rt", encoding="utf-8") as f:
                return f.read()
        except Exception:
            return ""

    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        # Skip les fichiers non binaires
        return ""
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
            text = text[start_line_end + 1 :]

    return text.strip()


def build_memmap(
    input_dir: Path,
    bin_path: Path,
    meta_path: Path,
    tokenizer: Encoding,
    allowed_ext: tuple = (".txt", ".md", ".gz", ""),
    max_bytes: int = 14 * 1024**3,
) -> int:
    assert input_dir.is_dir(), "Input must be directory"

    eof_id = tokenizer._special_tokens["<|endoftext|>"]

    total_tokens = 0
    files_processed = 0
    written = 0

    bin_path.parent.mkdir(parents=True, exist_ok=True)

    with open(bin_path, "wb") as bin_f:
        path = sorted(input_dir.rglob("*"))
        for file_path in path:
            if written >= max_bytes:
                print(f"Reached {written // (1024**3):.1f}GB limit, stopping.")
                break

            if not file_path.is_file():
                continue

            if file_path.suffix not in allowed_ext:
                continue

            raw = parse_file(file_path)
            if not raw.strip():
                continue
            cleaned = clean_text(raw)
            if not cleaned.strip():
                continue

            ids = tokenizer.encode(cleaned, allowed_special={""})
            ids.append(eof_id)

            # Write this file's tokens directly to disk (uint16: vocab < 65536 fits losslessly)
            np.array(ids, dtype=np.uint16).tofile(bin_f)

            total_tokens += len(ids)
            files_processed += 1
            written += len(ids) * 2

            if files_processed % 500 == 0:
                print(
                    f"  [{files_processed}] {file_path.name}: {len(ids):,} tokens (total: {total_tokens:,})"
                )

    if total_tokens == 0:
        raise ValueError(f"No tokens produced from {input_dir}")

    meta = {
        "num_tokens": total_tokens,
        "files_processed": files_processed,
        "dtype": "uint16",
        "vocab_size": tokenizer.n_vocab,
        "eof_id": int(eof_id),
    }

    meta_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    return total_tokens


def create_dataloader_v2(
    input_dir: Path,
    batch_size: int,
    context_length: int,
    stride: int,
    tokenizer: Encoding,
    split: float = 0.9,
    num_workers: int = 0,
    shuffle: bool = True,
    drop_last: bool = True,
    cache_dir: Path | None = None,
) -> tuple[DataLoader, DataLoader]:
    """Crée memmap si nécessaire, crée dataset, return DataLoader."""

    cache_dir = cache_dir if cache_dir else input_dir
    bin_path = cache_dir / "data.bin"
    meta_path = cache_dir / "data_meta.json"
    total_tokens = 0

    cache_dir.mkdir(parents=True, exist_ok=True)

    if not bin_path.exists():
        total_tokens = build_memmap(input_dir, bin_path, meta_path, tokenizer)
    else:
        total_tokens = os.path.getsize(bin_path) // np.dtype(np.uint16).itemsize

    split_idx = int(total_tokens * split)

    train_ds = GPTDatasetV2(
        bin_path, max_length=context_length, split_idx=(0, split_idx), stride=stride
    )
    val_ds = GPTDatasetV2(
        bin_path, max_length=context_length, split_idx=(split_idx, -1), stride=stride
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        drop_last=drop_last,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        drop_last=drop_last,
    )

    return train_loader, val_loader


def _detect_input_dir() -> Path:
    """On Kaggle, find the directory containing the text corpus (not openwebtext bins)."""
    if not _KAGGLE_INPUT.exists():
        return SCRIPT_DIR / "text"
    for p in sorted(_KAGGLE_INPUT.rglob("*")):
        if p.is_file() and p.suffix.lower() in (".txt", ".md", ".gz", ".csv"):
            return p.parent
    return SCRIPT_DIR / "text"


class TrainingConfig(BaseModel):
    """Hyperparamètres d'entraînement et de validation."""

    model: GPTConfig = Field(
        default_factory=GPTConfig, description="Architecture du modèle"
    )

    # Data
    batch_size: int = Field(default=4, gt=0, description="Taille des mini-batches")
    tokenizer_encoding: str = Field(
        default="gpt2", description="Encodeur tiktoken à utiliser"
    )
    stride_ratio: float = Field(
        default=0.5,
        gt=0,
        le=1,
        description="Ratio de chevauchement des fenêtres glissantes",
    )

    # Training loop
    num_epochs: int = Field(default=10, gt=0, description="Nombre total d'époques")
    lr: float = Field(
        default=1e-4, gt=0, description="Taux d'apprentissage maximal (OneCycleLR)"
    )
    weight_decay: float = Field(
        default=0.1, ge=0, description="Régularisation L2 du optimiseur AdamW"
    )
    warmup_steps: int = Field(
        default=500, ge=0, description="Nombre d'étapes de warmup linéaire"
    )
    grad_accum_steps: int = Field(
        default=1, gt=0, description="Nombre d'étapes d'accumulation de gradients"
    )
    max_grad_norm: float = Field(
        default=1.0, gt=0, description="Valeur maximale du gradient pour le clipping"
    )
    final_div_factor: float = Field(
        default=10.0, gt=0, description="Facteur de division finale du OneCycleLR"
    )

    # Evaluation
    eval_freq: int = Field(
        default=100, gt=0, description="Fréquence d'évaluation (en nombre de steps)"
    )
    num_batches: int = Field(
        default=-1, description="Nombre max de batches pour l'évaluation (-1 = tous)"
    )
    sample_length: int | None = Field(
        default=None,
        description="Longueur du texte généré pour l'exemple (None = context_length)",
    )
    start_context: list[str] = Field(
        default=[
            "Bonjour, je suis",
            "Hello, I am",
            "Every effort moves you",
            "Alors ",
            "mitochondrias are",
            "Chaque enfant qu'on enseigne,",
            "\n###",
            ".",
            "1 + 1 = ?",
        ],
        description="Textes de départ pour la génération d'exemples. Choisi au hasard",
    )

    # Misc
    seed: int = Field(
        default=8, description="Graine aléatoire pour la reproductibilité"
    )

    @property
    def example(self) -> str:
        return random.choice(self.start_context)


TRAINING_PRESET_TEST = dict(
    model=GPTConfig(
        num_layers=2,
        num_heads=4,
        context_length=256,
        embeddings_dim=64,
        drop_rate=0.2,
        vocab_size=50257,
        temperature=0.8,
        top_k=20,
    ),
    batch_size=2,
    num_epochs=10,
    eval_freq=500,
    num_batches=-1,
    warmup_steps=100,
    lr=3e-3,
)

TRAINING_PRESET_PROD = dict(
    model=GPTConfig(vocab_size=50257, drop_rate=0.1),
    batch_size=2,
    grad_accum_steps=8,
    num_epochs=5,  # Large dataset
    eval_freq=2000,
    num_batches=100,
    warmup_steps=2000,
    lr=3e-4,
)


def build_config(testing: bool) -> TrainingConfig:
    if testing:
        return TrainingConfig(**TRAINING_PRESET_TEST)
    return TrainingConfig(**TRAINING_PRESET_PROD)


def main() -> None:
    INPUT_DIR = _detect_input_dir()
    tc = build_config(False)
    max_length = tc.model.context_length
    stride = int(max_length * tc.stride_ratio)
    tokenizer = tiktoken.get_encoding("gpt2")

    train_loader, val_loader = create_dataloader_v2(
        input_dir=INPUT_DIR,
        cache_dir=WORK_DIR,
        batch_size=tc.batch_size,
        context_length=max_length,
        stride=stride,
        tokenizer=tokenizer,
    )
    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}\n")
