import torch
from torch import nn, Tensor


class SelfAttention_v4(nn.Module):
    """Encapsulation pour tout ce qui est relatif à l'attention"""

    def __init__(
        self,
        dim_inputs: int,
        dim_outputs: int,
        context_length: int,
        dropout: float,
        num_heads: int,
        qvk_bias: bool = False,
    ) -> None:
        """
        :param dim_inputs: dimension du tenseur d'entrée
        :param dim_outputs: dimension du tenseur de sortie
        :param qvk_bias: Si oui ou non il faut ajouter b
        :param dropout: Taux suivant lequel les connexions des vecteurs
        attentions sont masquées
        :param num_heads: nombre de têtes pour l'attention
        """
        super().__init__()

        assert (
            dim_outputs % num_heads == 0
        ), "dim_outputs doit être divisible par le\
        nombre de tête"

        # Nombre de composants du vecteur de sortie
        self.dim_outputs = dim_outputs

        self.num_heads = num_heads

        # Chaque tête est responsable d'une partie du résultat
        self.heads_dim = dim_outputs // num_heads
        # Paramètre *appris* pondérant les query q_i,
        # Il est initialisé aléatoirement
        self.W_query = nn.Linear(dim_inputs, dim_outputs, bias=qvk_bias)

        # Paramètre appris pondérant les clés k_i
        self.W_key = nn.Linear(dim_inputs, dim_outputs, bias=qvk_bias)

        # Pareille pour les valeurs
        self.W_value = nn.Linear(dim_inputs, dim_outputs, bias=qvk_bias)

        # Le synthétiseur de toute les informations récoltées par chaque tête
        self.output_projection = nn.Linear(dim_outputs, dim_outputs)

        # dropout
        self.dropout = nn.Dropout(dropout)

        # Pour optimiser les calculs avec Pytorch
        self.register_buffer(
            "mask", torch.triu(torch.ones(context_length, context_length), diagonal=1)
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        Calcul le vecteur context de chaque token
        :param x: Lot de Tenseurs en entrée
        :return context_vec: matrix/tenseur de tous les vecteurs context
        """
        batch_size, num_tokens, dim_inputs = x.shape

        # Obtenir les query pour chaque token
        queries: Tensor = self.W_query(x)

        # Obtenir les clés pour chaque token
        keys: Tensor = self.W_key(x)

        # Obtenir les clés pour chaque token
        values: Tensor = self.W_value(x)

        # Redisign les tenseurs par soucis d'optimisation des calculs
        # à venir.
        queries = queries.view(batch_size, num_tokens, self.num_heads, self.heads_dim)
        keys = keys.view(batch_size, num_tokens, self.num_heads, self.heads_dim)
        values = values.view(batch_size, num_tokens, self.num_heads, self.heads_dim)

        keys = keys.transpose(1, 2)
        queries = queries.transpose(1, 2)
        values = values.transpose(1, 2)

        # Calcul le score d'attention pour chaque token
        # On transpose par rapport au dimension 1 et 2 puisque
        # 0 est représente juste le nombre d'entrée dans le lot
        attention_scores: Tensor = queries @ keys.transpose(2, 3)

        # Pour chaque token, on masque le score de tous ceux qui viennent
        # après lui.
        mask_bool: Tensor = self.mask.bool()[:num_tokens, :num_tokens]
        attention_scores.masked_fill_(mask_bool, -torch.inf)

        attention_weight: Tensor = torch.softmax(
            attention_scores / keys.shape[-1] ** 0.5, dim=-1
        )
        context_vec: Tensor = attention_weight @ values

        # Fusionner les reponses
        context_vec = context_vec.transpose(
            1, 2
        )  # ->(batch_size, num_token, num_heads, heads:dim)

        # Ecrase num_head et head_dim en un
        context_vec = context_vec.contiguous().view(
            batch_size, num_tokens, self.dim_outputs
        )

        return self.output_projection(context_vec)


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
        self.output_proj = nn.Linear(dim_outputs, dim_outputs)

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
        output = self.output_proj(heads_output)

        return output


if __name__ == "__main__":
    import torch

    inputs = torch.tensor(
        [
            [0.43, 0.15, 0.89],  # Your     (x^1)
            [0.55, 0.87, 0.66],  # journey  (x^2)
            [0.57, 0.85, 0.64],  # starts   (x^3)
            [0.22, 0.58, 0.33],  # with     (x^4)
            [0.77, 0.25, 0.10],  # one      (x^5)
            [0.05, 0.80, 0.55],
        ]  # step     (x^6)
    )
