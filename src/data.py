import re
import pickle
import hashlib
from pathlib import Path

import numpy as np
import tiktoken
import torch
from tiktoken import Encoding
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from corpus import build_memmap

DELIMITERS = re.compile(r"([;,.?:_!()\'\"]|[^\S\r\n]|--)")
PONCTUATION = re.compile(r"\s+([,.:?!\'()\"_;])")


## ------------- Optimisation -----------------------
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


## ----------------------------------------------------


class SimpleTokenizerV1:
    def __init__(self, vocab):
        self.txt_to_num: dict[str, int] = vocab
        self.num_to_text: dict[int, str] = {i: s for s, i in vocab.items()}

    def encode(self, text: str) -> list[str]:
        input: list[str] = re.split(DELIMITERS, text)
        # print("[encode()]input:", input)
        preprocessed: list[str] = [item.strip(" ") for item in input if item.strip(" ")]
        # print("[encode()]preprocessed:", preprocessed)
        token: list = [self.txt_to_num.get(word, "<|unk|>") for word in preprocessed]
        # print("[encode()]token", token)

        print("[EMBEDINGS]: \n*", token)
        return token

    def decode(self, ids: list[int]) -> str:
        intermed = " ".join(self.num_to_text[i] for i in ids)
        output = re.sub(PONCTUATION, r"\1", intermed)

        print("[OUPUT]:\n--->", output)
        return output


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
        max_length: int = 1024,
        stride: int = 512,
        split: str = "train",
        split_ratio: float = 0.9,
    ) -> None:
        self.context_length = max_length
        self.stride = stride if stride is not None else max_length

        data = np.memmap(str(bin_path), dtype=np.uint16, mode="r")
        assert data is not None, f"File at '{bin_path}' is empty"

        split_idx = int(len(data) * split_ratio)

        if split == "train":
            data = data[:split_idx]
        else:
            data = data[split_idx:]

        self.data = data
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
    split: str = "train",
    num_workers: int = 0,
    shuffle: bool = True,
    drop_last: bool = True,
    cache_dir: Path | None = None,
) -> DataLoader:
    """Crée memmap si nécessaire, crée dataset, return DataLoader.

    cache_dir: dossier d'écriture pour data.bin / data_meta.json.
               Par défaut = input_dir. Utiliser WORK_DIR sur Kaggle
               car /kaggle/input est read-only.
    """

    input_dir = Path(input_dir)
    if cache_dir is None:
        cache_dir = input_dir
    else:
        cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    bin_path = cache_dir / "data.bin"
    meta_path = cache_dir / "data_meta.json"

    if not bin_path.exists():
        build_memmap(input_dir, bin_path, meta_path, tokenizer)

    ds = GPTDatasetV2(bin_path, context_length, stride, split)

    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        drop_last=drop_last,
    )


def vocabulary(text: str) -> dict[str, int]:
    preprocessed: list[str] = sorted(list(set(tokenize(text))))
    preprocessed.extend(["<|EOF|>", "<|unk|>"])
    # print("[vocabulary()]preprocessed: ", preprocessed)

    vocab_size: int = len(preprocessed)
    vocab: dict[str, int] = {token: num for num, token in enumerate(preprocessed)}

    print("Taille du Vocabulaire: ", vocab_size)

    # for i, item in enumerate(vocab.items()):
    #     print(f"{i}-{item}")
    #     if i > 50:
    #         break

    return vocab


def tokenize(text: str) -> list[str]:
    input: list[str] = re.split(DELIMITERS, text)
    preprocessed: list[str] = [item.strip(" ") for item in input if item.strip(" ")]

    print("[TOKENS]: \n*", preprocessed[:50])
    return preprocessed


if __name__ == "__main__":
    from pathlib import Path

    text_dir = Path(__file__).resolve().parent.parent / "text"
    with open(text_dir / "the-verdict.txt", "r", encoding="utf-8") as fichier:
        text = fichier.read()
        vocab = vocabulary(text)
        tokenizer = tiktoken.get_encoding("gpt2")
        sample: str = text[:99]
        print("[APERU]:\n''", sample, "\n''")

        print("=" * 15, "[TESTING ENCODING]", "=" * 15)
        context_window: int = 40
        x = tokenizer.encode(text[:context_window])
        y = tokenizer.encode(text[1 : context_window + 1])
        print(f"Tokens:\n* x: {x}\n* y:\t{y}")
        # print("=" * 15, "[TESTING DECODING]", "=" * 15)
        # print(tokenizer.decode(encoded_sample))
        output_dim = 256
        context_length = 1024

        token_embedding_layer = torch.nn.Embedding(len(vocab), output_dim)
        pos_embedding_layer = torch.nn.Embedding(context_length, output_dim)

        batch_size = 8
        max_length = 4
        dataloader = create_dataloader_v1(
            text, batch_size=batch_size, max_length=max_length, stride=max_length
        )
