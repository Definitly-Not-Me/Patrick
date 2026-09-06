from torch import nn, Tensor
import torch
from attention import Pytorch_MHA
from config import GPTConfig


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
        self.drop_shortcut: nn.Dropout = nn.Dropout(config.drop_rate)

    def forward(self, x: Tensor) -> Tensor:
        normalized_1 = self.norm1(x)
        normalized_2 = self.norm2(x)

        return x + (self.attention(normalized_1) + self.ffwd(normalized_2))


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


if __name__ == "__main__":
    from tiktoken import get_encoding

    config = GPTConfig()
    model = GPTModel(config)
    model.eval()
    model._size()
    start_context = "Hello world"
    encoder = get_encoding("gpt2")
    encoded = encoder.encode(start_context)
    print("encoded:", encoded)

    encoded_tensor = torch.tensor(encoded).unsqueeze(0)
    print("encoded_tensor.shape:", encoded_tensor.shape)

    out = model.generate(
        encoded_tensor, max_new_tokens=6, context_size=config.context_length
    )

    print("Out: ", out)
    print("Output length: ", len(out[0]))
    decoded_text = encoder.decode(out.squeeze(0).tolist())

    print(decoded_text)
