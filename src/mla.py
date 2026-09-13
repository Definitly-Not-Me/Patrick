from collections.abc import Iterator
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
from tiktoken import Encoding
import tiktoken
import hashlib
import pickle
import numpy as np
from datasets import load_dataset
from dotenv import load_dotenv
from requests.exceptions import ConnectionError, ChunkedEncodingError, HTTPError
import requests
from memory_profiler import profile
import torch
from torch import Tensor, device, nn
from pydantic import Field, BaseModel, model_validator


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


class GPTConfig(BaseModel):
    """
    Configuration du modèle. Par défaut ceux de GPT2-small
    """

    num_layers: int = Field(default=12)
    num_heads: int = Field(default=12, gt=1)
    context_length: int = Field(default=4096, gt=0, multiple_of=2)
    embeddings_dim: int = Field(default=768, gt=0)
    drop_rate: float = Field(default=0.1, lt=1)
    qvk_bias: bool = Field(default=False)
    vocab_size: int = Field(default=50257)
    temperature: float = Field(default=0.8, gt=0)
    top_k: int = Field(default=40)
    q_latent_dim: int = Field(default=512, gt=0)
    kv_latent_dim: int = Field(default=256, gt=0)
    rope_dim: int = Field(default=32, gt=0, multiple_of=2)

    @model_validator(mode="after")
    def validate_dimensions(self):
        if self.embeddings_dim % self.num_heads != 0:
            raise ValueError("embeddings_dim must be divisible by num_heads")
        return self


class TransformerBlock(nn.Module):
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


class GPTModel(nn.Module):
    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.tok_emb = nn.Embedding(config.vocab_size, config.embeddings_dim)

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
        self.cfg = config

    def forward(self, input_idx: Tensor) -> Tensor:
        batch_size, seq_len = input_idx.shape
        x = self.tok_emb(input_idx)

        x = self.drop_emb(x)
        x = self.trans_blocks(x)
        x = self.final_norm(x)
        logits = self.output_head(x)

        return logits

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


class RoPE(nn.Module):
    """Rotational Positional Embedding"""

    def __init__(self, dim: int, num_tokens: int, base: float = 10_000.0) -> None:
        super().__init__()
        assert dim % 2 == 0
        inv = base ** (-torch.arange(0, dim, 2, dtype=torch.float32) / dim)
        angle = torch.outer(torch.arange(num_tokens, dtype=torch.float32), inv)
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


class MLAV1(nn.Module):
    """
    Multi-Head Latent attention avec RoPE
    """

    def __init__(self, cfg: GPTConfig) -> None:
        super().__init__()
        self.dim_out = cfg.embeddings_dim
        self.num_heads = cfg.num_heads
        self.dim_heads = self.dim_out // self.num_heads
        self.dim_c = cfg.kv_latent_dim
        self.dim_r = cfg.rope_dim

        self.rope = RoPE(dim=cfg.rope_dim, num_tokens=cfg.context_length)

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
        scores = torch.matmul(q_abs.transpose(1, 2), C_kv.transpose(-2, -1)[:, None])

        # Postional scores
        q_R = self.rope(
            self.W_QR(C_q).view(batch_size, num_tokens, self.num_heads, self.dim_r),
            offset,
        )
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
        am = attn.mean(1).detach()
        attn = self.attn_dropout(attn)

        values = self.WU_V(C_kv).view(
            batch_size, num_tokens, self.num_heads, self.dim_heads
        )
        output = torch.matmul(attn, values.transpose(1, 2)).transpose(1, 2).contiguous()

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

        # Caching mechanism
        cache.append(self.WD_KV(x)[0, 0], self.rope(self.W_KR(x), pos)[0, 0])

        C_all, R_all = cache.C(), cache.R()
        attn = (q_abs @ C_all.T + q_R @ R_all.T).mul(self.scale).softmax(-1)

        C_bar = attn @ C_all
        out = torch.bmm(
            self.WU_V.weight.view(self.num_heads, self.dim_heads, -1),
            C_bar.unsqueeze(-1),
        )

        return self.Wo(out.view(1, 1, self.dim_heads * self.num_heads))


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


def _build_mask(self, idx_matrix) -> Tensor:
    batch_size, num_tokens, _ = idx_matrix.shape
    causal = torch.ones(
        num_tokens, num_tokens, dtype=torch.bool, device=idx_matrix.device
    ).tril()
    idx = (
        idx_matrix.masked_fill(~causal, float("-inf"))
        .topk(min(self.dsa_top_k, num_tokens), -1)
        .indices
    )
    m = torch.zeros(
        batch_size, num_tokens, num_tokens, dtype=torch.bool, device=idx_matrix.device
    )
    m.scatter_(-1, idx, True)

    return (m & causal[None]) | torch.eye(
        num_tokens, dtype=torch.bool, device=idx_matrix.device
    )[None]


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

def generate_and_print_sample(model, tokenizer, start_context, context_size, dev):
    model.eval()
    eof = tokenizer._special_tokens.get("<|endoftext|>", None)
    encoded = text_to_tokens(start_context, tokenizer).to(dev)
    with autocast(device_type=dev.type, enabled=dev.type == "cuda"):
        for tok in model.generate(encoded, max_new_tokens=20, context_size=context_size, EOF_id=eof):
            print(tokensIds_to_text(out, tokenizer), end="")
        print()
    model.train()

class LightningIndexer(nn.Module):
    """Inspiré par architecture de deepseek"""

    def __init__(self, cfg: GPTConfig, rope: RoPE) -> None:
        super().__init__()
        self.num_heads = cfg.indexer_heads
        self.dim_I = cfg.indexer_head_dim
        self.W_QI = nn.Linear(
            cfg.embeddings_dim, self.num_heads * self.dim_I, bias=cfg.qvk_bias
        )
        self.W_KI = nn.Linear(cfg.embeddings_dim, self.dim_I, bias=cfg.qvk_bias)
        self.w = nn.Linear(self.dim_I, 1, bias=False)
        self.rope = rope

    def forward(self, x, offset=0):
        batch_size, num_tokens, _ = x.shape

        q = self.rope(
            self.W_QI(x).view(batch_size, num_tokens, self.num_heads, self.dim_I),
            offset,
        )
        k = self.rope(self.W_KI(x)[:, :, None, :], offset).squeeze(2)
        grades = torch.relu(
            torch.matmul(q.transpose(1, 2), k[:, None].transpose(-2, -1))
        )
        w = self.w(q)

        return (w.transpose(1, 2) * grades).sum(1)


def shape_contract(m, cfg):
    d, H = cfg.embeddings_dim, cfg.num_heads
    dh, dq = d // H, cfg.q_latent_dim
    dc, dR = cfg.kv_latent_dim, cfg.rope_dim
    expect = {  # (out, in) — nn.Linear convention
        "WD_Q.weight": (dq, d),
        "W_QK.weight": (H * dc, dq),
        "WD_KV.weight": (dc, d),
        "WU_V.weight": (H * dh, dc),
        "W_QR.weight": (H * dR, dq),
        "W_KR.weight": (dR, d),
        "Wo.weight": (d, H * dh),
    }
    params = dict(m.named_parameters())
    for name, want in expect.items():
        got = tuple(params[name].shape)
        assert got == want, f"{name}: {got} != {want}"
    print("shape contract OK")


if __name__ == "__main__":
    cfg = GPTConfig()
    block = MLAV1(cfg)
    print("Total parameters", sum(p.numel() for p in block.parameters()))
    shape_contract(block, cfg)
    print(block(torch.randn(2, 16, 768)).shape)
    block.eval()
    X = torch.randn(1, 8, 768)
    full = block(X)
    cache = KVCache(
        dim_c=cfg.kv_latent_dim,
        max_len=cfg.context_length,
        dim_R=cfg.rope_dim,
        dev=torch.device("cpu"),
    )
    stepped = torch.cat([block.step(X[:, t : t + 1], cache, t) for t in range(8)], 1)
    print(torch.allclose(stepped, full, atol=1e-4))  # True → delete this script
    scout = LightningIndexer(cfg, rope=block.rope)
    print(scout(torch.randn(2, 16, 768)).shape)  # expect (2, 16, 16)
    print("Scout :", sum(p.numel() for p in scout.parameters()))
    out = block(X)
    (out.sum() + block.last_aux).backward()
    block.eval()  # dropout off — equality tests need determinism
