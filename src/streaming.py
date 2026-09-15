import os
import psutil
from datasets import load_dataset, interleave_datasets
from torch.utils.data import IterableDataset, DataLoader
from tiktoken import Encoding
from torch import Tensor
import tiktoken
import torch
import dotenv
import time
from requests.exceptions import ConnectionError, ChunkedEncodingError, HTTPError
import requests
import random
from memory_profiler import profile
from collections.abc import Iterator

MB = 1024**2

def _rss_mb() -> int:
    return psutil.Process(os.getpid()).memory_info().rss / MB


def _get_num_rows(src_path: str, name: str, split: str, retry: int = 3 ) -> int:
    """
    Utilise l'api hugging face pour approximer la taille
    du dataset
    """
    url = f"https://datasets-server.huggingface.co/size?dataset={src_path}&config={name}"

    for i in range(retry):
        try:
            response = requests.get(url).json()
        except (ConnectionError, HTTPError) as e:
            print("Error while trying to query dataset size :",repr(e))
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


def safe_next(it, max_retries=3, backoff=2.0) -> dict | None:
    """Retry on transient network errors, raise on permanent ones."""
    for attempt in range(max_retries):
        try:
            return next(it)
        except (ConnectionError, ChunkedEncodingError) as e:
            print(f"Failed to connect to hugging face. Retrying. [{attempt}/{max_retries}]")
            if attempt == max_retries - 1:
                raise e
            time.sleep(backoff * (attempt + 1))


def get_tokens(
    it,
    tokenizer,
    suffix: int,
) -> list[int] | None:
    """
    Retourne une liste des tokens renvoyés par next(it).
    """
    print(f"Before fetching: {_rss_mb():.2f} MB")
    try:
        example = safe_next(it)
    except StopIteration:
        return None
    print(f"After fetching: {_rss_mb():.2f} MB")

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
            sources: list de dict de format [{"path": "chemin/dataset", "name": "nom_dossier", "weight": 10}, ...]
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


def main() -> None:

    sources = [
        {
            "path": "HuggingFaceFW/fineweb-2",
            "name": "fra_Latn",
            "weight": 0.5,
        },
        {
            "path": "HuggingFaceFW/fineweb",
            "name": "CC-MAIN-2024-10",
            "weight": 0.3,
        },
        {
            "path": "dhlak/finewebedu-10b-gpt2-tokenized",
            "name": "default",
            "weight": 0.2,
        },
    ]
    tokenizer = tiktoken.get_encoding("gpt2")
    dataset = GPTDatasetV3(sources=sources, context_length=1024, stride=512, tokenizer=tokenizer, seed=15, split="train")
    train_loader = DataLoader(dataset, batch_size=8, num_workers=2, prefetch_factor=4)

    try:
        for tok in train_loader:
            print(tok)
    except KeyboardInterrupt:
        exit(0)


if __name__ == "__main__":
    dotenv.load_dotenv()
    main()
    # sources = [
    #     # {
    #     #     "path": "HuggingFaceFW/fineweb-2",
    #     #     "name": "fra_Latn",
    #     #     "weight": 2,
    #     # },

    #     # {
    #     #     "path": "HuggingFaceFW/fineweb",
    #     #     "name": "CC-MAIN-2024-10",
    #     #     "weight": 2,
    #     # },

    #     {
    #         "path": "dhlak/finewebedu-10b-gpt2-tokenized",
    #         "name": "default",
    #         "weight": 1,
    #     }
    # ]
    # tokenizer = tiktoken.get_encoding("gpt2")
    # dataset = GPTDatasetV3(sources=sources, context_length=1024, stride=512, tokenizer=tokenizer, seed=15, split="train")
    # loader = DataLoader(dataset, batch_size=8, num_workers=2, prefetch_factor=4)
