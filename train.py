import os
import sys
import csv
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
from torch.optim.optimizer import Optimizer
from torch.utils.data import ConcatDataset, DataLoader, Dataset
from tiktoken import Encoding
import tiktoken
import hashlib
import pickle
import numpy as np


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


def _detect_input_dir() -> Path:
    """On Kaggle, find the directory containing the text corpus (not openwebtext bins)."""
    if not _KAGGLE_INPUT.exists():
        return SCRIPT_DIR / "text"
    for p in sorted(_KAGGLE_INPUT.rglob("*")):
        if p.is_file() and p.suffix.lower() in (".txt", ".md", ".gz", ".csv"):
            return p.parent
    return SCRIPT_DIR / "text"


INPUT_DIR = _detect_input_dir()


def _perf(label: str, start: float):
    """Print elapsed time since `start` with a label."""
    print(f"[perf] {label}: {perf_counter() - start:.4f}s")


# ------------------------ Data --------------------------------------


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
        shuffle=False,
        num_workers=num_workers,
        drop_last=drop_last,
    )

    return train_loader, val_loader


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
    top_k: int = Field(default=40)

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
# ------------------------ Monitor ------------------------------------------



# ── monitor ──────────────────────────────────────────────────────────────────


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


class RoPE(nn.Module):
    """
    Rotationnary Positional Encoding
    """

    def __init__(self, max_seq_len: int, out_dim: int, base: float = 10_000.0) -> None:
        super().__init__()

        inverse_freq = 1.0 / (base ** (torch.arange(0, out_dim, 2).float() / out_dim))

        positions = torch.arange(max_seq_len).float()

        angles = positions[:, None] * inverse_freq[None, :]

        self.register_buffer("cos", angles.cos())
        self.register_buffer("sin", angles.sin())

    def forward(self, x: Tensor) -> Tensor:
        # x: [batch, heads, num_tokens, head_dim]
        seq_len = x.shape[-2]

        cos = self.cos[:seq_len]
        sin = self.sin[:seq_len]

        x_even = x[..., 0::2]
        x_odd = x[..., 1::2]

        rotated_even = x_even * cos - sin * x_odd
        rotated_odd = x_odd * cos + sin * x_even
        # Interleave dimensions again
        x_rotated = torch.empty_like(x)

        x_rotated[..., 0::2] = rotated_even
        x_rotated[..., 1::2] = rotated_odd

        return x_rotated


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
        print(
            f"Nombre de paramètres entrainables en considérant le weight tying: {total_params_gpt2:,}"
        )

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
) -> tuple[float, float]:
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
        train_loss = calc_loss_loader(train_loader, model, dev, num_batches)
        val_loss = calc_loss_loader(val_loader, model, dev, num_batches)

    model.train()
    return train_loss, val_loss


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


# -----------------------------------------------------------------------------


def train_model(
    model: GPTModel,
    train_loader: DataLoader,
    val_loader: DataLoader,
    optimizer: Optimizer,
    tc: TrainingConfig,
    tokenizer: Encoding,
    dev: device,
    save_path: str | Path = ".",
    start_epoch: int = 0,
    global_step_offset: int = 0,
    csv_path: Path | None = None,
    monitor: "Monitor | None" = None,
) -> tuple[list, list, list, list]:
    """
    Entraînement avec warmup, cosine decay, gradient clipping.
    Sauvegarde automatiquement le modèle en cas d'interruption (Ctrl+C).
    Log les métriques d'évaluation dans un CSV horodaté.
    """
    train_losses, val_losses, track_tokens, track_lrs = [], [], [], []
    tokens_seen, global_step = 0, global_step_offset - 1
    best_val_loss = float("inf")
    best_train_loss = float("inf")
    loader_len = len(train_loader)
    total_training_steps = loader_len * tc.num_epochs

    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer=optimizer,
        max_lr=optimizer.param_groups[0]["lr"],
        total_steps=total_training_steps // tc.grad_accum_steps,
        epochs=tc.num_epochs,
        final_div_factor=tc.final_div_factor,
        pct_start=tc.warmup_steps / max(total_training_steps // tc.grad_accum_steps, 1),
    )

    csv_writer, csv_file = open_csv(csv_path) if csv_path else (None, None)

    scaler = GradScaler(enabled=dev.type == "cuda")

    def _save_checkpoint(tag: str):
        if save_path is None:
            return
        path = Path(save_path)
        path = path.with_name(f"{path.stem}_{tag}{path.suffix}")
        path.parent.mkdir(exist_ok=True, parents=True)
        torch.save(
            {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "global_step": global_step,
                "epoch": epoch + 1,
                "best_val_loss": best_val_loss,
                "best_train_loss": best_train_loss,
                "scheduler": scheduler.state_dict(),
                "scaler": scaler.state_dict(),
            },
            path,
        )
        print(f"\nCheckpoint sauvegardé: {path}")

    try:
        model.train()
        for epoch in range(start_epoch, tc.num_epochs):
            optimizer.zero_grad()

            for i, (input_batch, target_batch) in enumerate(train_loader):
                should_step = (i + 1) % tc.grad_accum_steps == 0 or (
                    i + 1
                ) == loader_len
                with autocast(device_type=dev.type, enabled=dev.type == "cuda"):
                    accumulation = min(
                        tc.grad_accum_steps,
                        loader_len - (i // tc.grad_accum_steps) * tc.grad_accum_steps,
                    )
                    loss = (
                        calc_loss_batch(input_batch, target_batch, model, dev)
                        / accumulation
                    )

                scaler.scale(loss).backward()

                if should_step:
                    global_step += 1

                    scaler.unscale_(optimizer)

                    grad_norm = (
                        torch.nn.utils.clip_grad_norm_(
                            model.parameters(), max_norm=tc.max_grad_norm
                        )
                        if global_step > tc.warmup_steps
                        else 0.0
                    )

                    scaler.step(optimizer)
                    scaler.update()

                    optimizer.zero_grad()

                    scheduler.step()

                    lr_now = float(scheduler.get_last_lr()[0])

                    tokens_seen += input_batch.numel() * tc.grad_accum_steps
                    track_lrs.append(lr_now)

                    if global_step % tc.eval_freq == 0:
                        example = tc.example

                        torch.cuda.empty_cache()

                        train_loss, val_loss = evaluate_model(
                            model, train_loader, val_loader, dev, tc.num_batches
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

                        generate_and_print_sample(
                            model,
                            tokenizer,
                            example,
                            tc.sample_length or tc.model.context_length,
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
                            f"  Epoch:      {epoch + 1}/{tc.num_epochs} (step {global_step:,})"
                        )

            if len(train_loader) % tc.grad_accum_steps != 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), max_norm=tc.max_grad_norm
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


def split_corpus(corpus: str, train_ratio: float = 0.9) -> tuple[str, str]:
    split = int(len(corpus) * train_ratio)
    return corpus[:split], corpus[split:]


def get_device(testing: bool) -> torch.device:
    if testing:
        dev = torch.device("cpu")
    elif torch.cuda.is_available():
        try:
            torch.zeros(1, device="cuda")
            dev = torch.device("cuda")
        except Exception:
            dev = torch.device("cpu")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        dev = torch.device("mps")
    else:
        dev = torch.device("cpu")
    print(f"Device: {dev}\n")
    return dev


def build_config(testing: bool) -> TrainingConfig:
    if testing:
        return TrainingConfig(**TRAINING_PRESET_TEST)
    return TrainingConfig(**TRAINING_PRESET_PROD)


def main():
    import argparse
    from tiktoken import get_encoding

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--test", action="store_true", help="Quick sanity check with small config"
    )
    parser.add_argument(
        "--resume", type=str, default=None, help="Path to checkpoint to resume from"
    )
    # Notebook kernels don't have our CLI args in sys.argv (they carry the
    # kernel launcher's own args), so fall back to defaults there.
    if "__file__" in globals():
        args = parser.parse_args()
    else:
        args = parser.parse_args([])

    testing = os.getenv("testing") == "1" or args.test

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
    if IS_KAGGLE:
        train_loader, val_loader = _append_openwebtext(train_loader, val_loader)

    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}\n")

    # ── Model + Optimizer ──

    model = GPTModel(tc.model).to(dev)

    model._size()

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=tc.lr, weight_decay=tc.weight_decay
    )

    # ── Resume ──
    start_epoch = 0
    global_step_offset = 0
    if args.resume:
        ckpt = torch.load(args.resume, map_location=dev, weights_only=False)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        global_step_offset = ckpt["global_step"]
        start_epoch = ckpt["epoch"]
        print(
            f"Resumed from {args.resume} (epoch {start_epoch}, step {global_step_offset})\n"
        )

    # ── Save paths ──
    save_path = WORK_DIR / "model_checkpoint.pt"
    csv_path = make_csv_path(WORK_DIR)

    # ── Train ──

    monitor = Monitor()
    train_losses, val_losses, track_tokens, track_lrs = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        tc=tc,
        tokenizer=tokenizer,
        dev=dev,
        save_path=save_path,
        start_epoch=start_epoch,
        global_step_offset=global_step_offset,
        csv_path=csv_path,
        monitor=monitor,
    )
    monitor.close()

    # ── Save ──
    final_path = WORK_DIR / "model_final.pt"
    torch.save(model.state_dict(), final_path)
    print(f"\nModel saved to {final_path}")

    # ── Plots ──
    generate_plots(csv_path, WORK_DIR, tc.model_dump())


if __name__ == "__main__":
    main()
