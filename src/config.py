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
    vocab_size: int = Field(default=100277)
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
