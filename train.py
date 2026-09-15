import os
import sys
import csv
import psutil
from datetime import datetime
from pathlib import Path
import gzip
import json
import random
import time
from time import perf_counter
import torch
from torch import Tensor, device, nn
from torch.amp import GradScaler, autocast
from torch.utils.data import ConcatDataset, DataLoader, Dataset, IterableDataset
from tiktoken import Encoding, get_encoding
import tiktoken
import hashlib
import pickle
import numpy as np
from datasets import load_dataset, interleave_datasets
from dotenv import load_dotenv
from requests.exceptions import ConnectionError, ChunkedEncodingError, HTTPError
import requests
from memory_profiler import profile
from argparse import ArgumentParser, Namespace, RawTextHelpFormatter
from collections.abc import Iterator


# Base dir: notebook kernels don't define __file__, and src/ is absent there
# anyway (train.py is fully self-contained), so this is only meaningful when
# run as a script or imported from a local fs.
_BASE_DIR = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()

# Add src/ to path for imports (corpus, data modules)
_src_dir = str(_BASE_DIR / "src")
if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)

SCRIPT_DIR = _BASE_DIR
IS_KAGGLE = bool(os.getenv("KAGGLE_KERNEL_RUN_TYPE"))
WORK_DIR = Path("/kaggle/working") if IS_KAGGLE else SCRIPT_DIR
_KAGGLE_INPUT = Path("/kaggle/input")
ENV = (
    Path("../input/datasets/definitlynotme/llm-training-data/.env")
    if Path("../input/datasets/definitlynotme/llm-training-data/.env").exists()
    else Path().cwd()
)

ISTTY = sys.stdout.isatty()

SOURCES = {
    "train": [
        {
            "path": "HuggingFaceFW/finewiki",
            "name": "fr",
            "weight": (50 / 100),
        },
        {
            "path": "HuggingFaceFW/fineweb-2",
            "name": "fon_Latn",
            "weight": (10 / 100),
        },
        {
            "path": "HuggingFaceFW/finewiki",
            "name": "en",
            "weight":  (40 / 100) ,
        },
    ],
    "val": [
        {
            "path": "HuggingFaceFW/fineweb-2",
            "name": "fra_Latn",
            "weight": 0.8,
        },
        {
            "path": "HuggingFaceFW/fineweb-edu",
            "name": "default",
            "weight": 0.2,
        },
    ],
}


load_dotenv(ENV, verbose=True)


def _detect_input_dir() -> Path:
    """On Kaggle, find the directory containing the text corpus (not openwebtext bins)."""
    if not _KAGGLE_INPUT.exists():
        return SCRIPT_DIR / "text"
    for p in sorted(_KAGGLE_INPUT.rglob("*")):
        if p.is_file() and p.suffix.lower() in (".txt", ".md", ".gz", ".csv"):
            return p.parent
    return SCRIPT_DIR / "text"


INPUT_DIR = _detect_input_dir()

# -------------------- Debugging --------------------------------------


MB = 1024**2


def _rss_mb() -> int:
    return psutil.Process(os.getpid()).memory_info().rss / MB


def _perf(label: str, start: float):
    """Print elapsed time since `start` with a label."""
    print(f"[perf] {label}: {perf_counter() - start:.4f}s")


# ------------------------ Data --------------------------------------

# == Helper functions for Dataset ==


def load_tokens(txt: str, tokenizer: Encoding) -> list[int]:
    """
    Sauvegarde les tokens d'un fichier texte pour éviter de retokenizer à chaque
    fois
    """
    if not txt:
        return []
    text_hash = hashlib.sha256(txt.encode()).hexdigest()[:16]

    cache = Path(f".cache/tokens_{text_hash}.pkl")
    cache.parent.mkdir(parents=True, exist_ok=True)

    if cache.exists():
        with cache.open("rb") as file:
            tokens_ids: list[int] = pickle.load(file)
    else:
        with cache.open("wb") as file:
            tokens_ids = tokenizer.encode(txt, allowed_special={"<|endoftext|>"})
            pickle.dump(tokens_ids, file)
    return tokens_ids


def safe_next(it, max_retries=3, backoff=2.0) -> dict | None:
    """Retry on transient network errors, raise on permanent ones."""
    for attempt in range(max_retries):
        try:
            return next(it)
        except (ConnectionError, ChunkedEncodingError) as e:
            print(
                f"Failed to connect to hugging face. Retrying. [{attempt}/{max_retries}]"
            )
            if attempt == max_retries - 1:
                raise e
            time.sleep(backoff * (attempt + 1))


def _get_num_rows(src_path: str, name: str, split: str, retry: int = 3) -> int:
    """
    Utilise l'api hugging face pour approximer la taille
    du dataset
    """
    url = (
        f"https://datasets-server.huggingface.co/size?dataset={src_path}&config={name}"
    )

    for i in range(retry):
        try:
            response = requests.get(url).json()
        except (ConnectionError, HTTPError) as e:
            print("Error while trying to query dataset size :", repr(e))
            print(f"Retrying [{i}/{retry}]")

            if i + 1 == retry:
                return random.randint(1000, int(1e6))
            else:
                continue

    split_info = response["size"]["splits"]
    for entry in split_info:
        if entry["split"] == split:
            return entry["num_rows"]

    return response["size"]["dataset"]["num_bytes_memory"] // 4


def get_tokens(
    it,
    tokenizer,
    suffix: int,
) -> list[int] | None:
    """
    Retourne une liste des tokens renvoyés par next(it).
    """
    try:
        example = safe_next(it)
    except StopIteration:
        return None

    if "input_ids" in example:
        # Pre-tokenized
        ids = example["input_ids"]

        return ids if isinstance(ids, list) else [ids]

    elif "text" in example:
        # Raw text
        return tokenizer.encode_ordinary(example["text"]) + [suffix]

    else:
        print("Unexpected response:", example)
        return None


# == Dataset ==


class GPTDatasetV1(Dataset[tuple[Tensor, Tensor]]):
    def __init__(self, txt, tokenizer, max_length, stride):

        token_ids = load_tokens(txt, tokenizer)

        self.input_ids = []
        self.target_ids = []
        # Use a sliding window to chunk the book into overlapping sequences of max_length
        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i : i + max_length]
            target_chunk = token_ids[i + 1 : i + max_length + 1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx) -> tuple[Tensor, Tensor]:
        return self.input_ids[idx], self.target_ids[idx]


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



class GPTDatasetV3(IterableDataset):
    """
    Version du dataset tirant ses sources de hugging face.
    Par souci d'optimisation du stockage, cette version
    stream les fichiers et les tokenize ad hoc si besoin
    """

    def __init__(
        self,
        sources: list[dict],
        tokenizer: Encoding,
        context_length: int,
        stride: int,
        seed: int,
        split: str,
    ) -> None:
        """
        Args:
            sources: list de dict de format [{"path": "chemin/dataset", "name": "nom_dossier", "weight": 0.8}, ...]
            weight: Combien de documents prendre de 'nom_dossier' par round
        """
        self.sources = sources
        self.tokenizer = tokenizer
        self.ctx = context_length

        self.stride = stride
        self.seed = seed
        self.split = split
        self._length = 0

        assert 0 < self.stride <= self.ctx, f"stride must be in (0, ctx], got {self.stride}"

        # Estimation du nombre de tokens
        for src in self.sources:
            approx_length = _get_num_rows(src["path"], src["name"], split)
            self._length += approx_length


    def __iter__(self)-> Iterator[tuple[Tensor, Tensor]]:
        buffer = []
        data = []
        eof = self.tokenizer._special_tokens["<|endoftext|>"]

        for src in self.sources:
            print("Querying Dataset ", src["name"])
            ds = load_dataset(
                src["path"],
                name=src["name"],
                streaming=True,
                split=self.split
            )
            data.append(ds)

        mixed_dataset = interleave_datasets(
            data,
            probabilities=[src["weight"] for src in self.sources],
            seed = self.seed,
        )
        mixed_dataset.shuffle(buffer_size=10_000)

        for sample in mixed_dataset:
            if sample.get("text"):
                tokens = self.tokenizer.encode(sample["text"])
                tokens.append(eof)
            elif sample.get("input_ids"):
                ids =  sample.get("inputs_ids")
                tokens = ids if isinstance(ids, list) else [ids]

            else:
                print("Unknown response received", sample)
                tokens = None
                continue

            if tokens is None:
                continue

            buffer.extend(tokens)

            while len(buffer) >= self.ctx + 1:
                x = torch.tensor(
                    buffer[: self.ctx],
                    dtype=torch.long,
                )
                y = torch.tensor(
                    buffer[1 : self.ctx + 1],
                    dtype=torch.long,
                )

                yield x, y
                buffer = buffer[self.stride :]

    def __len__(self)-> int:
        return self._length

def create_dataloader_v1(
    txt, batch_size, max_length, stride, shuffle=True, drop_last=True, num_workers=0
):
    # Initialize the tokenizer
    tokenizer = tiktoken.get_encoding("gpt2")

    # Create dataset
    dataset = GPTDatasetV1(txt, tokenizer, max_length, stride)

    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers,
    )

    return dataloader


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
        shuffle=True,
        num_workers=num_workers,
        drop_last=drop_last,
    )

    return train_loader, val_loader


def create_dataloader_v3(
    sources: list[dict],
    batch_size: int,
    context_length: int,
    stride: int,
    tokenizer: Encoding,
    split: str,
    num_workers: int = 0,
    prefetch_factor: int = 2,
    seed: int = 6,
) -> DataLoader:

    ds = GPTDatasetV3(
        sources=sources,
        tokenizer=tokenizer,
        stride=stride,
        context_length=context_length,
        split=split,
        seed=seed,
    )

    dataloader = DataLoader(
        ds,
        batch_size=batch_size,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor,
    )

    return dataloader


# ------------------------ Config --------------------------------------

from pydantic import Field, BaseModel, model_validator


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
    top_k: int = Field(default=64)
    q_latent_dim: int = Field(default=384, gt=0)
    kv_latent_dim: int = Field(default=128, gt=0)
    rope_dim: int = Field(default=32, gt=0, multiple_of=2)
    table_size: int = Field(default=2, gt=0)

    @model_validator(mode="after")
    def validate_dimensions(self):
        if self.embeddings_dim % self.num_heads != 0:
            raise ValueError("embeddings_dim must be divisible by num_heads")
        return self


GPT_medium = GPTConfig(embeddings_dim=1024, num_layers=24, num_heads=16)

GPT_large = GPTConfig(embeddings_dim=1280, num_layers=36, num_heads=20)

GPT_XL = GPTConfig(embeddings_dim=1600, num_layers=48, num_heads=25)


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

# Pour l'environement de Kaggle
TRAINING_PRESET_PROD = dict(
    model=GPTConfig(vocab_size=50257, drop_rate=0.1, context_length=512),
    batch_size=4,
    grad_accum_steps=16,
    num_epochs=3,  # Large dataset
    eval_freq=200,
    num_batches=50,
    warmup_steps=2000,
    lr=6e-4,
)
# ------------------------ Monitor ------------------------------------------


class Monitor:
    """Plain-text training monitor.

    Replaces the rich TUI: rich's Live/Console recursion under Kaggle's
    papermill execution, so we print one summary line per eval instead.
    """

    def __init__(self, max_history: int = 300, refresh_per_second: int = 4):
        # max_history / refresh_per_second kept for call-site compatibility.
        self.start_time = time.time()

    def log(
        self,
        *,
        step: int,
        epoch: int | None = None,
        train_loss: float | None = None,
        val_loss: float | None = None,
        lr: float | None = None,
        grad_norm: float | None = None,
        tokens_per_sec: float | None = None,
        tokens_seen: int | None = None,
    ):
        parts = [f"step {step}"]
        if epoch is not None:
            parts.append(f"epoch {epoch}")
        if train_loss is not None:
            parts.append(f"train {train_loss:.4f}")
        if val_loss is not None:
            parts.append(f"val {val_loss:.4f}")
        if lr is not None:
            parts.append(f"lr {lr:.3e}")
        if grad_norm is not None:
            parts.append(f"grad {grad_norm:.3f}")
        if tokens_per_sec is not None:
            parts.append(f"{tokens_per_sec:,.0f} tok/s")
        if tokens_seen is not None:
            parts.append(f"tokens {tokens_seen:,}")
        print("[monitor] " + " | ".join(parts), flush=True)

    def close(self):
        print(f"[monitor] done in {time.time() - self.start_time:.1f}s", flush=True)


# --------------------- CSV Logger -------------------------------------------

CSV_HEADER = [
    "step",
    "epoch",
    "train_loss",
    "val_loss",
    "lr",
    "grad_norm",
    "tokens_seen",
    "best_val_loss",
    "best_train_loss",
]


def make_csv_path(work_dir: Path) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return work_dir / f"training_log_{timestamp}.csv"


def open_csv(path: Path):
    f = open(path, "w", newline="")
    writer = csv.writer(f)
    writer.writerow(CSV_HEADER)
    return writer, f


def log_eval_row(
    writer,
    *,
    step: int,
    epoch: int,
    train_loss: float,
    val_loss: float,
    lr: float,
    grad_norm: float,
    tokens_seen: int,
    best_val_loss: float,
    best_train_loss: float,
):
    writer.writerow(
        [
            step,
            epoch,
            f"{train_loss:.6f}",
            f"{val_loss:.6f}",
            f"{lr:.2e}",
            f"{grad_norm:.4f}",
            tokens_seen,
            f"{best_val_loss:.6f}",
            f"{best_train_loss:.6f}",
        ]
    )


# --------------------- Plots ------------------------------------------------


def generate_plots(csv_path: Path, output_dir: Path, hyperparams: dict):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    data = {col: [] for col in CSV_HEADER}
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            for col in CSV_HEADER:
                data[col].append(float(row[col]))

    if not data["step"]:
        print("No eval data to plot.")
        return

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(
        data["step"], data["train_loss"], label="Train Loss", marker="o", markersize=3
    )
    ax.plot(data["step"], data["val_loss"], label="Val Loss", marker="s", markersize=3)
    ax.set_xlabel("Step")
    ax.set_ylabel("Loss")
    ax.set_title("Training Progress")
    ax.legend()
    ax.grid(True, alpha=0.3)

    hp_text = "\n".join(f"{k}: {v}" for k, v in hyperparams.items())
    ax.text(
        0.98,
        0.98,
        f"{hp_text:.20}",
        transform=ax.transAxes,
        fontsize=8,
        verticalalignment="top",
        horizontalalignment="right",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
        fontfamily="monospace",
    )

    plt.tight_layout()
    plot_path = output_dir / "loss_curve.png"
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"Graph saved to {plot_path}")


# ----------------------------- Parse data -----------------------------------

"""Corpus building: file parsing, text cleaning, memmap generation."""


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


# --------------------------------- Model -----------------------------------


def get_bigram_hash(x: Tensor, table_size: int) -> Tensor:
    """
    Calcul le hash de chaque bigram d'une sequence
    Args:
        x: Tenseur (B, T)
    Returns: (B, T) table des indices
    """
    rand_int_1 = int(2**32 - 1)
    rand_int_2 = 2147483623

    mod = table_size - 1

    x = x.to(torch.int64).clone()
    x[:, 0] = mod
    x[:, 1:] = torch.bitwise_xor(
        rand_int_1 * x[:, 1:],
        rand_int_2 * x[:, :-1]
    ) % mod

    return x


class GPTModelV2(nn.Module):
    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.tok_emb = nn.Embedding(config.vocab_size, config.embeddings_dim)

        # Empile les transformers blocks
        self.trans_blocks = nn.Sequential(
            *[TransformerBlockV2(config) for _ in range(config.num_layers)]
        )

        # Engram
        self.bigram_embed = nn.Embedding(config.table_size * config.vocab_size,config.embeddings_dim // 2, dtype=torch.float32)
        self.bigram_proj = nn.Linear(config.embeddings_dim // 2, config.embeddings_dim)
        nn.init.zeros_(self.bigram_embed.weight)
        nn.init.zeros_(self.bigram_proj.weight)
        self.x0_lambdas = nn.Parameter(torch.zeros(config.num_layers))       # unigram re-injection gate
        self.bigram_lambdas = nn.Parameter(0.1 * torch.ones(config.num_layers))  # bigram gate


        self.final_norm = RMSNorm(config.embeddings_dim)
        self.output_head = nn.Linear(
            config.embeddings_dim, config.vocab_size, bias=False
        )
        self.drop_emb = nn.Dropout(config.drop_rate)
        self.top_k = config.top_k
        self.temp = config.temperature
        self.cfg = config

    def forward(self, input_idx: Tensor) -> Tensor:
        batch_size, seq_len = input_idx.shape
        x0 = self.tok_emb(input_idx)
        x0 = nn.functional.rms_norm(x0, (self.cfg.embeddings_dim,))
        x = self.drop_emb(x0)
        bigram_idx = get_bigram_hash(input_idx, self.bigram_embed.num_embeddings)
        x0_bigram = self.bigram_proj(self.bigram_embed(bigram_idx))

        for i, block in enumerate(self.trans_blocks):
            x = x + self.x0_lambdas[i] * x0 + self.bigram_lambdas[i] * x0_bigram
            x = block(x)

        x = self.trans_blocks(x)
        x = self.final_norm(x)
        logits = self.output_head(x)

        return logits * torch.tanh(logits / 15.0)

    def _size(self) -> float:  # based on chapter code

        total_params = sum(p.numel() for p in self.parameters())
        print(f"Nombre total de parametre: {total_params:,}")

        total_params_gpt2 = total_params - sum(
            p.numel() for p in self.output_head.parameters()
        )
        # print(
        #     f"Nombre de paramètres entrainables en considérant le weight tying: {total_params_gpt2:,}"
        # )

        # Calculate the total size in bytes (assuming float32, 4 bytes per parameter)
        total_size_bytes = total_params * 4

        # Convert to megabytes
        total_size_mb = total_size_bytes / (1024 * 1024)

        print(f"Taille totale du modele: {total_size_mb:.2f} MB")

        return total_size_mb

    def generate(
        self,
        input: Tensor,
        max_new_tokens: int,
        context_size: int,
        EOF_id: int | None = None,
    ) -> Iterator[Tensor]:
        """
        Genere le prochain token d'une sequence
        Args:
           input: le Tenseur (batch_size, tokens_id)
           EOF_id: l'id du token EOF
        Yield:
           next_id: l'id du prochain token
        """
        assert input.size(0) == 1, "Inference needs batch size = 1."

        prompt = input[0].tolist()
        prompt_len = len(prompt)

        assert prompt_len != 0, "Prompt should not be empty"

        max_new_tokens = min(max_new_tokens, context_size - prompt_len)
        total = prompt_len + max_new_tokens
        device = next(self.parameters()).device

        caches = [
            KVCache(total, self.cfg.kv_latent_dim, self.cfg.rope_dim, device)
            for _ in range(self.cfg.num_layers)
        ]

        emb = self.tok_emb(input)
        pos = 0

        # Cache KV pour token du prompt
        for t in range(prompt_len):
            x = emb[:, t : t + 1]
            for block, cache in zip(self.trans_blocks, caches):
                x = block.step(x, cache, pos=t)

        pos = prompt_len

        logits = self.output_head(x.squeeze(0))

        for _ in range(max_new_tokens):

            ## Garde seulement les top_k plus probables tokens
            if self.top_k > 1:
                top_logits = torch.topk(logits, self.top_k)
                min_val = top_logits.values[:, -1]
                logits = torch.where(
                    logits < min_val,
                    float("-inf"),
                    logits,
                )

            if self.temp > 0.0:
                probas = torch.softmax(logits / self.temp, dim=-1)
                next_id = torch.multinomial(probas, num_samples=1)
            else:
                next_id = torch.argmax(logits, dim=-1, keepdim=True)

            if EOF_id is not None and (next_id == EOF_id).any():
                break

            # Effet typewritter
            yield next_id

            # Nouveau token va dans le cache
            x = self.tok_emb(next_id)
            for block, cache in zip(self.trans_blocks, caches):
                x = block.step(x, cache, pos)
            pos += 1
            logits = self.output_head(x.squeeze(0))

    def compile_xpath(self, all: bool = False):
        """
        Compile une partie/la totalité des couches les plus lourdes
        du model, ce qui optimise la vitesse des appels forward()
        """
        for block in self.trans_blocks:
            print("Compiling attention blocks ...")
            block.attention.compile()

        if all:
            for name, module in self.named_modules():
                if isinstance(module, nn.Module) and callable(module.forward):
                    print(f"Compiling module {name} of model ...")
                    module.compile()

class TransformerBlockV2(nn.Module):
    """
    Transformer block. Combine toutes les autres couches en un bloc cohérant
    """

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.attention = MLAV1(GPTConfig())

        self.ffwd: SwiGLU = SwiGLU(config.embeddings_dim)
        self.norm1: RMSNorm = RMSNorm(config.embeddings_dim)
        self.norm2: RMSNorm = RMSNorm(config.embeddings_dim)
        self.dropout: nn.Dropout = nn.Dropout(config.drop_rate)

    def forward(self, x: Tensor) -> Tensor:
        x_attn = self.attention(self.norm1(x))
        x_ffwd = self.ffwd(self.norm2(x))

        return x + self.dropout(x_attn) + self.dropout(x_ffwd)

    def step(self, x: Tensor, cache: "KVCache", pos: int) -> Tensor:
        """
        Equivalent de forward() optimisé pour l'inférence
        """
        x_attn = self.attention.step(self.norm1(x), cache, pos)
        x_ffwd = +self.ffwd(self.norm2(x))

        return x + self.dropout(x_attn) + self.dropout(x_ffwd)


class KVCache:
    """
    KV caching pour optimiser la vitesse
    d'inférence
    """

    def __init__(
        self,
        max_len: int,
        dim_c: int,
        dim_R: int,
        dev: device,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        self.C_buf = torch.zeros(max_len, dim_c, device=dev, dtype=dtype)

        self.R_buf = torch.zeros(max_len, dim_R, device=dev, dtype=dtype)
        self.len = 0

    def append(self, c, r):
        self.C_buf[self.len] = c
        self.R_buf[self.len] = r
        self.len += 1

    def C(self) -> Tensor:
        return self.C_buf[: self.len]

    def R(self) -> Tensor:
        return self.R_buf[: self.len]

    def __len__(self) -> int:
        return self.len




class MLAV1(nn.Module):
    """
    Multi-Head Latent attention avec RoPE, XSA
    """

    def __init__(self, cfg: GPTConfig) -> None:
        super().__init__()
        self.dim_out = cfg.embeddings_dim
        self.num_heads = cfg.num_heads
        self.dim_heads = self.dim_out // self.num_heads
        self.dim_c = cfg.kv_latent_dim
        self.dim_r = cfg.rope_dim

        self.rope = RoPE(dim=cfg.rope_dim, max_seq_len=cfg.context_length)

        # Query
        self.WD_Q = nn.Linear(cfg.embeddings_dim, cfg.q_latent_dim, bias=cfg.qvk_bias)

        # Absorbed Query
        self.W_QK = nn.Linear(
            cfg.q_latent_dim, self.num_heads * cfg.kv_latent_dim, bias=cfg.qvk_bias
        )

        # Latent
        self.WD_KV = nn.Linear(cfg.embeddings_dim, self.dim_c, bias=cfg.qvk_bias)

        # values
        self.WU_V = nn.Linear(
            self.dim_c, self.num_heads * self.dim_heads, bias=cfg.qvk_bias
        )

        # Rope decoupling
        self.W_QR = nn.Linear(
            cfg.q_latent_dim, self.num_heads * cfg.rope_dim, bias=cfg.qvk_bias
        )  # q^R per head (384→384)

        self.W_KR = nn.Linear(cfg.embeddings_dim, cfg.rope_dim, bias=cfg.qvk_bias)

        self.Wo = nn.Linear(
            self.num_heads * self.dim_heads, cfg.embeddings_dim, bias=cfg.qvk_bias
        )

        self.scale = cfg.kv_latent_dim**-0.5
        self.attn_dropout = nn.Dropout(cfg.drop_rate)

    def forward(self, x, offset: int = 0, causal: bool = True) -> None:
        batch_size, num_tokens, _ = x.shape

        # Latents
        C_q = self.WD_Q(x)
        C_kv = self.WD_KV(x)
        k_R = self.rope(self.W_KR(x), offset)

        # Absorbed Query and summary
        q_abs = self.W_QK(C_q).view(batch_size, num_tokens, self.num_heads, self.dim_c)

        # Normalisation
        nn.functional.rms_norm(q_abs, (self.dim_c,))
        nn.functional.rms_norm(C_kv, (self.dim_c,))

        scores = torch.matmul(q_abs.transpose(1, 2), C_kv.transpose(-2, -1)[:, None])

        # Postional scores
        q_R = self.rope(
            self.W_QR(C_q).view(batch_size, num_tokens, self.num_heads, self.dim_r),
            offset,
        )
        # Normalisation
        nn.functional.rms_norm(k_R, (self.dim_r,))
        nn.functional.rms_norm(q_R, (self.dim_r,))

        scores = scores + torch.matmul(
            q_R.transpose(1, 2), k_R.transpose(-2, -1)[:, None]
        )

        scores = scores * self.scale

        if causal:
            mask = torch.ones(
                num_tokens, num_tokens, dtype=torch.bool, device=x.device
            ).tril()[None]
        else:
            mask = None

        # `~` signifie  ici negation
        if mask is not None:
            scores = scores.masked_fill(~mask[:, None], float("-inf"))

        attn = scores.softmax(dim=-1)
        attn = self.attn_dropout(attn)

        values = self.WU_V(C_kv).view(
            batch_size, num_tokens, self.num_heads, self.dim_heads
                )

        # Orthogonal projection (XSA)

        output = torch.matmul(attn, values.transpose(1, 2)).transpose(1, 2).contiguous()

        v_n = nn.functional.normalize(values, dim=-1)
        output = output - (output * v_n).sum(dim=-1, keepdim=True) * v_n

        return self.Wo(
            output.view(batch_size, num_tokens, self.num_heads * self.dim_heads)
        )

    def step(self, x: Tensor, cache: "KVCache", pos: int) -> Tensor:
        """
        Version de forward() utilisant les resultats precedements générés (dans le cache).

        Optimise la vitesse d'inférence
        Args:
             x: tenseur representant le tout dernier token generé. shape: (1, 1, model dim)


             cache: l'instance Cache a utiliser
             pos: position absolue de x dans la sequence à générer
        Returns:
            out: hidden state
        """

        C_q = self.WD_Q(x)
        q_abs = self.W_QK(C_q).view(self.num_heads, self.dim_c)
        q_R = self.rope(
            self.W_QR(C_q).view(1, 1, self.num_heads, self.dim_r), pos
        ).view(self.num_heads, self.dim_r)


        # Normalisation
        q_abs = nn.functional.rms_norm(q_abs, (self.dim_c,))
        q_R = nn.functional.rms_norm(q_R, (self.dim_r,))
        k_c = nn.functional.rms_norm(self.WD_KV(X)[0, 0], (self.dim_c,))
        k_r =  nn.functional.rms_norm(self.rope(self.W_KR(x), pos)[0, 0], (self.dim_r,))

        # Caching Mechanism
        cache.append(k_c, k_r)

        C_all, R_all = cache.C(), cache.R()

        attn = (q_abs @ C_all.T + q_R @ R_all.T).mul(self.scale)

        # XSA
        attn[:, -1] = float("-inf")
        attn = attn.softmax(dim=-1)

        C_bar = attn @ C_all
        out = torch.bmm(
            self.WU_V.weight.view(self.num_heads, self.dim_heads, -1),
            C_bar.unsqueeze(-1),
        )

        return self.Wo(out.view(1, 1, self.dim_heads * self.num_heads))


class RoPE(nn.Module):
    """Rotational Positional Embedding"""

    def __init__(self, dim: int, max_seq_len: int, base: float = 10_000.0) -> None:
        super().__init__()
        assert dim % 2 == 0
        inv = base ** (-torch.arange(0, dim, 2, dtype=torch.float32) / dim)
        angle = torch.outer(torch.arange(max_seq_len, dtype=torch.float32), inv)
        embeddings = torch.cat([angle, angle], dim=-1)
        self.register_buffer(
            "cos", embeddings.cos(), persistent=False
        )  # not learned -> out of state_dict
        self.register_buffer("sin", embeddings.sin(), persistent=False)

    def forward(self, x: Tensor, offset: int = 0) -> Tensor:
        if x.dim() == 3:
            return self.forward(x[:, :, None, :], offset).squeeze(2)
        T = x.size(1)
        cos = self.cos[offset : offset + T][None, :, None, :]  # (1, T, 1, D)
        sin = self.sin[offset : offset + T][None, :, None, :]
        x1, x2 = x.chunk(2, dim=-1)

        return x * cos + torch.cat([-x2, x1], dim=-1) * sin



class Pytorch_MHA(nn.Module):
    """
    Mécanisme d'attention optimisé avec Pytorch
    Multi-Head et produit scalaire
    """

    def __init__(
        self,
        dim_inputs: int,
        dim_outputs: int,
        num_heads: int,
        context_length: int,
        dropout: float = 0.1,
        qvk_bias: bool = False,
        need_weights: bool = False,
    ) -> None:
        super().__init__()
        self.multihead_attention = nn.MultiheadAttention(
            embed_dim=dim_outputs,
            num_heads=num_heads,
            dropout=dropout,
            bias=qvk_bias,
            add_bias_kv=qvk_bias,
            batch_first=True,
        )
        self.context_length: int = context_length
        self.need_weights: bool = need_weights

        self.register_buffer(
            "mask",
            torch.triu(torch.ones(context_length, context_length), diagonal=1).bool(),
        )

    def forward(self, x: Tensor) -> Tensor:
        batch_size, num_tokens, _ = x.shape

        if self.context_length >= num_tokens:
            attn_mask = self.mask[:num_tokens, :num_tokens]
        else:
            attn_mask = self.mask[: self.context_length, : self.context_length]

        heads_output, _ = self.multihead_attention(
            x, x, x, attn_mask=attn_mask, need_weights=self.need_weights
        )

        return heads_output


class LayerNorm(nn.Module):
    """
    Couche de normalisation linéaire.

    Stabilise l'entraînement en facilitant la convergence
    """

    def __init__(self, embeddings_dim: int) -> None:

        super().__init__()

        # Constante pour eviter les divisions par zéro
        self.eps = 1e-5

        # Paramètres additionnels que le modèles peut modifier
        # afin d'améliorer les performances
        self.scale = nn.Parameter(torch.ones(embeddings_dim))
        self.shift = nn.Parameter(torch.zeros(embeddings_dim))

    def forward(self, x: Tensor) -> Tensor:
        moyenne = x.mean(dim=-1, keepdim=True)
        variance = x.var(dim=-1, keepdim=True, unbiased=False)
        norm_x = (x - moyenne) / torch.sqrt(variance + self.eps)

        return self.scale * norm_x + self.shift


class RMSNorm(nn.Module):
    """
    Couche de normalisation semblable a LayerNorm a la difference
    qu'elle moins coûteuse puisqu'elle stabilise seulement la
    variance
    """

    def __init__(self, dim: int, eps: float = 1e-8) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: Tensor) -> Tensor:
        root_mean_squared = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)

        return self.weight * (x * root_mean_squared)


class GELU(nn.Module):
    """
    Approximation de la fonction d'activation GELU
    """

    def __init__(self):
        super().__init__()

    def forward(self, x: Tensor) -> Tensor:
        return (
            0.5
            * x
            * (
                1
                + torch.tanh(
                    torch.sqrt(torch.tensor(2.0 / torch.pi))
                    * (x + 0.044715 * torch.pow(x, 3))
                )
            )
        )


class SwiGLU(nn.Module):
    """
    fonction d'activation Sigmoid with Gated Linear Unit.
    Plus effective que GELU dans le domaine des LLMs
    """

    def __init__(self, dim) -> None:
        super().__init__()
        self.up = nn.Linear(dim, dim * 4, bias=False)
        self.gate = nn.Linear(dim, dim * 4, bias=False)
        self.down = nn.Linear(dim * 4, dim, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        """
                     ┌─ Linear → SiLU ─┐
        x ───────────┤                 × → Linear → y
                     └─ Linear ────────┘
        """
        return self.down(self.up(x) * nn.functional.silu(self.gate(x)))


class FeedForward(nn.Module):
    """
    Module du réseau de neurones orchestrant certains composants répétitifs
    """

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(config.embeddings_dim, 4 * config.embeddings_dim),
            GELU(),
            nn.Linear(4 * config.embeddings_dim, config.embeddings_dim),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.layers(x)


class TransformerBlock(nn.Module):
    """
    Transformer block. Combine toutes les autres couches en un bloc cohérant
    """

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.attention: Pytorch_MHA = Pytorch_MHA(
            dim_inputs=config.embeddings_dim,
            dim_outputs=config.embeddings_dim,
            dropout=config.drop_rate,
            qvk_bias=config.qvk_bias,
            num_heads=config.num_heads,
            context_length=config.context_length,
        )

        self.ffwd: SwiGLU = SwiGLU(config.embeddings_dim)
        self.norm1: RMSNorm = RMSNorm(config.embeddings_dim)
        self.norm2: RMSNorm = RMSNorm(config.embeddings_dim)
        self.dropout: nn.Dropout = nn.Dropout(config.drop_rate)

    def forward(self, x: Tensor) -> Tensor:
        x_attn = self.attention(self.norm1(x))
        x_ffwd = self.ffwd(self.norm2(x))

        return x + self.dropout(x_attn) + self.dropout(x_ffwd)


class GPTModel(nn.Module):
    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.tok_emb = nn.Embedding(config.vocab_size, config.embeddings_dim)

        # Embedding de Position
        self.pos_emb = nn.Embedding(config.context_length, config.embeddings_dim)

        # Empile les transformers blocks
        self.trans_blocks = nn.Sequential(
            *[TransformerBlock(config) for _ in range(config.num_layers)]
        )

        self.final_norm = RMSNorm(config.embeddings_dim)
        self.output_head = nn.Linear(
            config.embeddings_dim, config.vocab_size, bias=False
        )
        self.drop_emb = nn.Dropout(config.drop_rate)
        self.top_k = config.top_k
        self.temp = config.temperature

    def forward(self, input_idx: Tensor) -> Tensor:
        batch_size, seq_len = input_idx.shape
        tok_embeds = self.tok_emb(input_idx)
        pos_embeds = self.pos_emb(torch.arange(end=seq_len, device=input_idx.device))

        x = tok_embeds + pos_embeds
        x = self.drop_emb(x)
        x = self.trans_blocks(x)
        x = self.final_norm(x)
        logits = self.output_head(x)

        return logits

    def _size(self) -> None:  # based on chapter code

        total_params = sum(p.numel() for p in self.parameters())
        print(f"Nombre total de parametre: {total_params:,}")

        total_params_gpt2 = total_params - sum(
            p.numel() for p in self.output_head.parameters()
        )
        # print(
        #     f"Nombre de paramètres entrainables en considérant le weight tying: {total_params_gpt2:,}"
        # )

        # Calculate the total size in bytes (assuming float32, 4 bytes per parameter)
        total_size_bytes = total_params * 4

        # Convert to megabytes
        total_size_mb = total_size_bytes / (1024 * 1024)

        print(f"Taille totale du modele: {total_size_mb:.2f} MB")

    def generate(
        self,
        input: Tensor,
        max_new_tokens: int,
        context_size: int,
        EOF_id: int | None = None,
    ) -> Tensor:
        """
        Genere le prochain token d'une sequence
        Args:
           input: le Tenseur (batch_size, tokens_id)
           temperature: introduit un charactère aléatoire dans la selection du
           prochain token
           top_k: parmi les top_k tokens les plus propables sera choisi le prochain
           EOF_id: l'id du token EOF
        Returns:
           output: le Tenseur avec le token généré ajouter à l'input
        """

        for _ in range(max_new_tokens):
            current_input = input[:, -context_size:]
            with torch.no_grad():
                logits = self.forward(current_input)

            logits = logits[:, -1, :]

            ## Garde seulement les top_k plus probables tokens
            if self.top_k > 1:
                top_logits = torch.topk(logits, self.top_k)
                min_val = top_logits.values[:, -1]
                logits = torch.where(
                    logits < min_val,
                    float("-inf"),
                    logits,
                )

            if self.temp > 0.0:
                probas = torch.softmax(logits / self.temp, dim=-1)
                next_id = torch.multinomial(probas, num_samples=1)
            else:
                next_id = torch.argmax(logits, dim=-1, keepdim=True)

            if EOF_id is not None and (next_id == EOF_id).any():
                break

            input = torch.cat((input, next_id), dim=1)

        return input


# -------------- Fonctions auxiliaire ----------------------------------------
def text_to_tokens(text: str, tokenizer: Encoding) -> Tensor:
    """
    Transforme du text en une représentation vectorielle i.e embedding
    """
    tokens_ids = tokenizer.encode(text, allowed_special={"<|EOF|>"})
    encoded_tensor = torch.tensor(tokens_ids).unsqueeze(0)

    return encoded_tensor


def tokensIds_to_text(tokens_ids: Tensor, tokenizer: Encoding) -> str:
    """
    Transform un représentation vectorielle (embedding) en text
    """
    flat = tokens_ids.squeeze(0)

    return tokenizer.decode(flat.tolist())


def calc_loss_batch(
    input_batch: Tensor, target_batch: Tensor, model: GPTModel, dev: device
) -> Tensor:
    """
    Calcule la fonction de perte pour une configuration  model
    """
    input_batch, target_batch = input_batch.to(dev), target_batch.to(dev)
    logits = model(input_batch)
    loss = torch.nn.functional.cross_entropy(
        logits.flatten(0, 1), target_batch.flatten()
    )
    return loss


def calc_loss_loader(
    data_loader: DataLoader, model: GPTModel, dev: device, num_batches: int = -1
) -> float:
    """
    Evalue le model sur un certain nombre d'examples du dataset
    Args:
         data_loader: Un objet de la class DataLoader contenant le dataser
         model: Le model a évalué
         dev : l'unité de calcul (cpu vs gpu) où éffectué le calcul
         num_batch  : le nombre d'examples à considére
    Returns:
        total_loss: la somme de toutes les pertes sur l'ensemble des examples
        pris en compte
    """
    total_loss: float = 0.0
    data_size = len(data_loader)

    if data_size == 0:
        raise ValueError(f"DataLoader {data_loader} refers to an empty Dataset")
    elif num_batches < 0:
        num_batches = data_size
    else:
        ## Plafonne le nombre d'examples
        num_batches = min(num_batches, data_size)

    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i < num_batches:
            with autocast(device_type=dev.type, enabled=dev.type == "cuda"):
                loss = calc_loss_batch(input_batch, target_batch, model, dev)
            total_loss += loss.item()
        else:
            break

    return total_loss / num_batches


def evaluate_model(
    model: GPTModel,
    train_loader: DataLoader,
    val_loader: DataLoader,
    dev: device,
    num_batches: int,
) -> float:
    """
    Compare la performance du model sur les données d'entraînement et ceux de
    validation
    Args:
        num_batches: la taille de l'échantillon de chaque dataset a considérer
        train_loader: le dataloader correspondant au datset d'entraînement
        val_loader: le dataloader correspondant au dataset de validation
    Returns:
        train_loss, val_loss: perf sur les données d'entraînement, perf sur ceux
        de validation
    """
    model.eval()
    with torch.no_grad():
        # Cause souvent OUT OF MEMORY sur des gros dataset
        # train_loss = calc_loss_loader(train_loader, model, dev, num_batches)
        val_loss = calc_loss_loader(val_loader, model, dev, num_batches)

    model.train()
    return val_loss


def generate_and_print_sample(model, tokenizer, start_context, context_size, dev):
    model.eval()
    eof = tokenizer._special_tokens.get("<|endoftext|>", None)
    encoded = text_to_tokens(start_context, tokenizer).to(dev)
    with autocast(device_type=dev.type, enabled=dev.type == "cuda"):
        out = model.generate(
            encoded, max_new_tokens=20, context_size=context_size, EOF_id=eof
        )
    print(tokensIds_to_text(out, tokenizer))
    model.train()


def generate_and_print_sampleV2(model, tokenizer, start_context, context_size, dev):
    model.eval()
    eof = tokenizer._special_tokens.get("<|endoftext|>", None)
    encoded = text_to_tokens(start_context, tokenizer).to(dev)
    with autocast(device_type=dev.type, enabled=dev.type == "cuda"):
        for tok in model.generate(
            encoded, max_new_tokens=20, context_size=context_size, EOF_id=eof
        ):
            print(tokensIds_to_text(tok, tokenizer), end="", flush=True)
        print()
    model.train()


def load_checkpoint(filepath: Path, dev: device, config: GPTConfig) -> dict:
    with torch.device("meta"):
        model = GPTModel(config)
        scaler = GradScaler(enabled=dev.type == "cuda")

    checkpoint = torch.load(filepath, map_location=dev, weights_only=False)
    model.load_state_dict(checkpoint["model"], assign=True, strict=True)

    optimizer = torch.optim.AdamW(params=model.parameters())
    optimizer.load_state_dict(checkpoint["optimizer"])

    scaler.load_state_dict(checkpoint["scaler"])

    print("Successfully Loaded checkpoint from ", str(filepath))

    checkpoint["model"] = model
    checkpoint["optimizer"] = optimizer
    checkpoint["scaler"] = scaler

    return checkpoint


def get_device(testing: bool) -> torch.device:
    if testing:
        dev = torch.device("cpu")
    elif torch.cuda.is_available():
        try:
            free, total = torch.cuda.mem_get_info()
            print(f"Cuda info: {free / MB} MB free / {total / MB} MB")
            needed = 4 * 4096 * 768 * 4  #

            if free < needed:
                raise OSError("SIGSEGV, Insufficient memory")

            # Test si le gpu peux supporter un tensor moyen
            torch.zeros(4, 4096, 768, device="cuda")
            dev = torch.device("cuda")
        except Exception as e:
            print(e)
            print("\n\033[31mGPU indisponible. Fallback sur cpu\033[0m")
            dev = torch.device("cpu")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        dev = torch.device("mps")
    else:
        dev = torch.device("cpu")
    print(f"Device: {dev}\n")
    return dev


def build_config(testing: bool, args: Namespace | None = None) -> TrainingConfig:
    preset = TRAINING_PRESET_TEST if testing else TRAINING_PRESET_PROD
    config = TrainingConfig(**preset)

    if not args:
        return config

    for name, value in vars(args).items():
        if value is None:
            continue
        if hasattr(config.model, name) and value != getattr(config.model, name):
            setattr(config.model, name, value)
        elif hasattr(config, name) and value != getattr(config, name):
            setattr(config, name, value)

    return config


# -----------------------------------------------------------------------------


def train_model(
    model: GPTModel | GPTModelV2,
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: TrainingConfig,
    tokenizer: Encoding,
    dev: device,
    monitor=None,
    resume_from: Path | None = None,
    save_dir: str | Path = ".",
) -> tuple[list, list, list, list]:
    """
    Entraînement avec warmup, cosine decay, gradient clipping.
    Sauvegarde automatiquement le modèle en cas d'interruption (Ctrl+C).
    Log les métriques d'évaluation dans un CSV horodaté.
    """
    # TRACKERS
    train_losses, val_losses, track_tokens, track_lrs = [], [], [], []
    tokens_seen, global_step = 0, -1
    best_val_loss = float("inf")
    best_train_loss = float("inf")
    loader_len = len(train_loader)
    running_train_loss = 0.0
    accum_loss = 0.0

    total_training_steps = loader_len * config.num_epochs

    csv_path = make_csv_path(Path(save_dir))
    csv_writer, csv_file = open_csv(csv_path) if csv_path else (None, None)

    if resume_from:
        checkpoint = load_checkpoint(resume_from, dev, config.model)
        model = checkpoint["model"]
        optimizer = checkpoint["optimizer"]
        scaler = checkpoint["scaler"]
        global_step = checkpoint["global_step"]
        start_epoch = checkpoint["epoch"]
        best_train_loss = checkpoint["best_train_loss"]
        best_val_loss = checkpoint["best_val_loss"]
        model.to(dev)

        print(f"Resumed from {resume_from} (epoch {start_epoch}, step {global_step})\n")

    else:
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=config.lr, weight_decay=config.weight_decay
        )
        scaler = GradScaler(enabled=dev.type == "cuda")
        start_epoch = 0

    # Scheduler doit etre neuf pour chaque nouvelle entrainement
    # Sans quoi le learning rate se comporte de maniere imprevisible
    # eg: rester constant car total_step a ete atteint entrainement passe
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer=optimizer,
        max_lr=optimizer.param_groups[0]["lr"],
        total_steps=total_training_steps // config.grad_accum_steps,
        epochs=config.num_epochs,
        final_div_factor=config.final_div_factor,
        pct_start=config.warmup_steps
        / max(total_training_steps // config.grad_accum_steps, 1),
    )

    # Nécessaire de le définir à l'intérieur pour simplifier la sauvegarde

    def _save_checkpoint(tag: str):
        path = Path(save_dir) / "model_checkpoint.pt"
        path = path.with_name(f"{path.stem}_{tag}{path.suffix}")
        path.parent.mkdir(exist_ok=True, parents=True)

        torch.save(
            {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "global_step": global_step,
                "epoch": epoch,
                "best_val_loss": best_val_loss,
                "best_train_loss": best_train_loss,
                "scaler": scaler.state_dict(),
            },
            path,
        )
        print(f"\nCheckpoint sauvegardé: {path}")

    try:
        model.train()
        for epoch in range(start_epoch, config.num_epochs):
            optimizer.zero_grad()

            for i, (input_batch, target_batch) in enumerate(train_loader):
                should_step = (i + 1) % config.grad_accum_steps == 0 or (
                    i + 1
                ) == loader_len
                with autocast(device_type=dev.type, enabled=dev.type == "cuda"):
                    accumulation = min(
                        config.grad_accum_steps,
                        loader_len
                        - (i // config.grad_accum_steps) * config.grad_accum_steps,
                    )
                    loss = (
                        calc_loss_batch(input_batch, target_batch, model, dev)
                        / accumulation
                    )

                scaler.scale(loss).backward()
                tokens_seen += input_batch.numel()
                accum_loss += loss.item()

                if should_step:
                    ## Debugging
                    if i < 1000:
                        print("Stepping ... ", global_step)

                    global_step += 1

                    scaler.unscale_(optimizer)

                    grad_norm = (
                        torch.nn.utils.clip_grad_norm_(
                            model.parameters(), max_norm=config.max_grad_norm
                        )
                        if global_step > config.warmup_steps
                        else 0.0
                    )

                    scaler.step(optimizer)
                    scaler.update()

                    optimizer.zero_grad()

                    scheduler.step()

                    lr_now = float(scheduler.get_last_lr()[-1])

                    track_lrs.append(lr_now)

                    running = accum_loss
                    accum_loss = 0.0
                    running_train_loss = (
                        running
                        if running_train_loss > 0
                        else 0.95 * running_train_loss + 0.05 * running
                    )

                    if global_step % config.eval_freq == 0:
                        example = config.example

                        torch.cuda.empty_cache()

                        train_loss = running_train_loss

                        val_loss = evaluate_model(
                            model, train_loader, val_loader, dev, config.num_batches
                        )
                        train_losses.append(train_loss)
                        val_losses.append(val_loss)
                        track_tokens.append(tokens_seen)

                        if val_loss < best_val_loss:
                            best_val_loss = val_loss
                            _save_checkpoint("best_model")

                        if train_loss < best_train_loss:
                            best_train_loss = train_loss

                        if monitor:
                            monitor.log(
                                step=global_step,
                                epoch=epoch + 1,
                                train_loss=train_loss,
                                val_loss=val_loss,
                                lr=lr_now,
                                grad_norm=float(grad_norm),
                                tokens_per_sec=tokens_seen
                                / (time.time() - monitor.start_time),
                                tokens_seen=tokens_seen,
                            )

                        if csv_writer:
                            log_eval_row(
                                csv_writer,
                                step=global_step,
                                epoch=epoch + 1,
                                train_loss=train_loss,
                                val_loss=val_loss,
                                lr=lr_now,
                                grad_norm=float(grad_norm),
                                tokens_seen=tokens_seen,
                                best_val_loss=best_val_loss,
                                best_train_loss=best_train_loss,
                            )
                            csv_file.flush()

                        print(
                            f"\n{'=' * 15} SAMPLE (step {global_step}, epoch {epoch + 1}) {'=' * 15}"
                        )
                        print(f"Input:  {example}\nOutput: ", end="")

                        generate_and_print_sampleV2(
                            model,
                            tokenizer,
                            example,
                            config.sample_length or config.model.context_length,
                            dev,
                        )
                        print("=" * 60)

                        print(
                            f"  Train Loss: {train_loss:.4f} (best: {best_train_loss:.4f})"
                        )
                        print(
                            f"  Val Loss:   {val_loss:.4f} (best: {best_val_loss:.4f})"
                        )
                        print(
                            f"  Epoch:      {epoch + 1}/{config.num_epochs} (step {global_step:,})"
                        )
                        print(f" Learning rate: {lr_now}")

            if len(train_loader) % config.grad_accum_steps != 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), max_norm=config.max_grad_norm
                )
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scheduler.step()

    except KeyboardInterrupt:
        _save_checkpoint("interrupted")
        raise
    finally:
        if csv_file:
            csv_file.close()

    return train_losses, val_losses, track_tokens, track_lrs


# --------------------- __main__ helpers -------------------------------------


def _append_openwebtext(
    train_loader: DataLoader, val_loader: DataLoader
) -> tuple[DataLoader, DataLoader]:
    """
    Petit Hack pour charger les données de openwebtext en plus de celles de
    présentes
    """
    ds = train_loader.dataset  # commun aux deux

    candidates = list(Path("/kaggle/input").rglob("train.bin"))
    if not candidates:
        raise FileNotFoundError("No openwebtext train.bin found under /kaggle/input")
    owt_dir = candidates[0].parent
    owt_train = GPTDatasetV2(
        owt_dir / "train.bin", max_length=ds.context_length, stride=ds.stride
    )
    owt_val = GPTDatasetV2(
        owt_dir / "val.bin", max_length=ds.context_length, stride=ds.stride
    )

    new_train_ds = ConcatDataset([train_loader.dataset, owt_train])
    new_val_ds = ConcatDataset([val_loader.dataset, owt_val])

    new_train_loader = DataLoader(
        dataset=new_train_ds,
        shuffle=True,  # Par hypothèse
        batch_size=train_loader.batch_size,
        num_workers=train_loader.num_workers,
        drop_last=train_loader.drop_last,
        pin_memory=True,
    )

    new_val_loader = DataLoader(
        dataset=new_val_ds,
        shuffle=True,
        batch_size=train_loader.batch_size,
        num_workers=train_loader.num_workers,
        drop_last=train_loader.drop_last,
        pin_memory=True,
    )

    return new_train_loader, new_val_loader


def parse_args():
    parser = ArgumentParser(
        prog="train",
        description="""
        Script pour l'entraînement d'un Modèle de Language
        semblable à GPT2.
        """,
        epilog="@Definitly-Not-Me, 2026\n**Ce programme est fourni sans garantie**",
        formatter_class=RawTextHelpFormatter
    )

    parser.add_argument(
        "--resume-from",
        type=Path,
        help="""
        Poursuivre l'entraînement d'un model à partir d'un fichier de sauvegarde
        '.pt'. Attention à ce que L'architecture du model corresponde sans quoi
        la sauveguarde sera considéré invalide\n
        """,
    )

    parser.add_argument(
        "--input",
        type=Path,
        default="online",
        help="""
        Chemin du dossier contenant le dataset pour l'entraînement.
        Stream le dataset depuis Hugging Face si la valeur est 'online'.
        Il est recommandé que les fichiers du dataset soit aux formats:
        .txt, .gzip, .md ou .bin\n
        """,
        required=True,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        help="""
        Le nombre d'examples que le model rencontre par lot\n
        """,
    )

    parser.add_argument(
        "--context-length",
        type=int,
        help="""
        La longueur de la fenêtre de contexte du model.
        Correspond aussi à la longueur des sequences que le model aperçoit.
        Doit être divisible par 2.\n
        """,
    )

    parser.add_argument(
        "--num-epoch",
        type=int,
        help="""
        Le nombre d'époch (itération) que le model doit effectuer sur
        le dataset\n
        """,
    )

    parser.add_argument(
        "--eval-freq",
        type=int,
        help="""
        Le nombre de pas (examples) que le modèle doit rencontrer avant chaque
        evaluation\n
        """,
    )

    parser.add_argument(
        "--lr",
        type=float,
        help="""
        Taux d'apprentissage\n
        """,
    )

    parser.add_argument(
        "--num-heads", type=int, help="Nombre de tête d'attention par block\n"
    )

    parser.add_argument("--num-layers", type=int, help="Nombre de couches du model\n")

    parser.add_argument("--drop-rate", type=float, help="Probabilité de dropout\n")

    parser.add_argument(
        "--out",
        type=Path,
        default=".",
        help="Le dossier ou écrire les fichiers qui seront générés\n",
    )

    parser.add_argument(
        "-v",
        type=int,
        default=2,
        choices=[1, 2],
        help="""
        Quelle version du modèle utilisé.
        La version 1 utilise l'architecture Multihead attention + positionnal embeddings
        originale de GPT2 tandis que la version 2 utilise Multihead latent attention + RoPE
        inspiré de DeepSeek.\n
        """,
    )

    parser.add_argument(
        "--num_workers",
        type=int,
        default=1,
        help="""
        Le nombre de threads reservable du chargement du dataset\n
        """,
    )

    parser.add_argument(
        "--test",
        default=False,
        action="store_true",
        help="""
        Entraîner le modèle sur le CPU avec une configuration minimale et un
        dataset déliberément réduit.\n
        """,
    )

    parser.add_argument(
        "--seed",
        type=int,
        help="""
        La valeur de la seed du générateur de Pytorch.\n
        """,
    )

    parser.add_argument(
        "--taille",
        type=str,
        default="small",
        choices=["small", "medium", "large", "xl"],
        help="""
        Variante du model.
        Small correspond à GPT-2 small (~ 124 M)
        et XL à GPT-2 XL (~ 1.5 B)\n
        """,
    )

    parser.add_argument(
        "--compile",
        default=False,
        action="store_true",
        help="""Appeler torch.compile() sur le modele pour optimiser les calculs"""
    )

    return parser.parse_args()


def entrypoint():
    args = parse_args()
    testing_mode = bool(os.getenv("testing")) or args.test
    tc = build_config(testing_mode, args)
    tokenizer = tiktoken.get_encoding(tc.tokenizer_encoding)
    dev = get_device(testing_mode)
    max_length = args.context_length or tc.model.context_length
    stride = int(max_length * tc.stride_ratio)
    num_workers = args.num_workers

    torch.manual_seed(args.seed or tc.seed)

    ## DatLoading
    print("Chargement du dataset depuis", args.input)

    if str(args.input) == "online":
        train_sources = SOURCES["train"]
        val_sources = SOURCES["val"]

        train_loader = create_dataloader_v3(
            sources=train_sources,
            batch_size=tc.batch_size,
            context_length=max_length,
            stride=stride,
            num_workers=num_workers,
            split="train",
            tokenizer=tokenizer,
        )
        val_loader = create_dataloader_v3(
            sources=val_sources,
            batch_size=tc.batch_size,
            context_length=max_length,
            stride=stride,
            num_workers=num_workers,
            split="train",  # Intentionnel, rares split "train" sur Hugging
            tokenizer=tokenizer,
        )

    else:
        sources = Path(args.input)
        train_loader, val_loader = create_dataloader_v2(
            input_dir=sources,
            cache_dir=args.out or WORK_DIR,
            batch_size=tc.batch_size,
            context_length=max_length,
            stride=stride,
            tokenizer=tokenizer,
        )

    print(f"Train batches: {len(train_loader)}, Validation batches: {len(val_loader)}\n")

    ## Model Loading

    print(f"Chargement de la Configuration '{args.taille}'")

    if args.resume_from:
        load_path = Path(args.resume_from)
    else:
        load_path = None

    match args.taille:
        case "small":
            gptconf = tc.model
        case "medium":
            gptconf = GPT_medium
        case "large":
            gptconf = GPT_large
        case "xl":
            gptconf = GPT_XL
        case _:
            gptconf = GPTConfig()

    match args.v:
        case 1:
            model = GPTModel(gptconf)
        case 2:
            model = GPTModelV2(gptconf)
        case _:
            print("Unknown Model version", args.v)
            exit(1)

    model.to(dev)
    model._size()

    if args.compile and not testing_mode:
        model.compile()
        if callable(model.compile_xpath):
           model.compile_xpath()

    monitor = Monitor()
    train_losses, val_losses, track_tokens, track_lrs = train_model(
        train_loader=train_loader,
        val_loader=val_loader,
        save_dir=args.out or WORK_DIR,
        config=tc,
        dev=dev,
        tokenizer=tokenizer,
        model=model,
        resume_from=load_path,
    )

    monitor.close()

    final_path = WORK_DIR / "model_final.pt"
    torch.save(model.state_dict(), final_path)
    print(f"\nModel saved to {final_path}")


def resume() -> None:
    testing = os.getenv("testing") == "1"
    tc = build_config(testing)
    tokenizer = tiktoken.get_encoding(tc.tokenizer_encoding)
    dev = get_device(testing)
    max_length = tc.model.context_length
    stride = int(max_length * tc.stride_ratio)
    num_workers = 1

    train_sources = SOURCES["train"]
    val_sources = SOURCES["val"]

    torch.manual_seed(tc.seed)

    train_loader = create_dataloader_v3(
        sources=train_sources,
        batch_size=tc.batch_size,
        context_length=max_length,
        stride=stride,
        num_workers=num_workers,
        split="train",
        tokenizer=tokenizer,
    )
    val_loader = create_dataloader_v3(
        sources=val_sources,
        batch_size=tc.batch_size,
        context_length=max_length,
        stride=stride,
        num_workers=num_workers,
        split="train",  # Intentionnel, rares split "train" sur Hugging
        tokenizer=tokenizer,
    )

    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}\n")

    if testing or not IS_KAGGLE:
        inp = input(">>Spécifiez le chemin du fichier de sauvegarde: ")
        load_path = Path(inp)
    else:
        load_path = Path(
            "../input/models/definitlynotme/patrick-gpt2/pytorch/default/1/model_checkpoint_best_model.pt"
        )

    model = GPTModel(tc.model).to(dev)
    model._size()

    # ── Train ──

    monitor = Monitor()
    train_losses, val_losses, track_tokens, track_lrs = train_model(
        train_loader=train_loader,
        val_loader=val_loader,
        save_dir=WORK_DIR,
        config=tc,
        dev=dev,
        tokenizer=tokenizer,
        model=model,
        resume_from=load_path,
    )
    monitor.close()

    final_path = WORK_DIR / "model_final.pt"
    torch.save(model.state_dict(), final_path)
    print(f"\nModel saved to {final_path}")


def main():

    testing = os.getenv("testing") == "1"

    tc = build_config(testing)

    tokenizer = get_encoding(tc.tokenizer_encoding)

    # ── Device ──

    dev = get_device(testing)

    torch.manual_seed(tc.seed)

    # ── Data (memmap) ──
    max_length = tc.model.context_length
    stride = int(max_length * tc.stride_ratio)

    train_loader, val_loader = create_dataloader_v2(
        input_dir=INPUT_DIR,
        cache_dir=WORK_DIR,
        batch_size=tc.batch_size,
        context_length=max_length,
        stride=stride,
        tokenizer=tokenizer,
    )

    # Append pretokenized OpenWebText (tokenizer must be gpt-2)
    # if IS_KAGGLE:
    #     train_loader, val_loader = _append_openwebtext(train_loader, val_loader)

    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}\n")

    # ── Model + Optimizer ──

    model = GPTModel(tc.model).to(dev)

    model._size()

    # ── Train ──

    monitor = Monitor()
    train_losses, val_losses, track_tokens, track_lrs = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=tc,
        tokenizer=tokenizer,
        dev=dev,
        save_dir=WORK_DIR,
        monitor=monitor,
    )
    monitor.close()

    # ── Save ──
    final_path = WORK_DIR / "model_final.pt"
    torch.save(model.state_dict(), final_path)
    print(f"\nModel saved to {final_path}")


if __name__ == "__main__":

    entrypoint()
