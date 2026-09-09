import os
import psutil
from datasets import load_dataset
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
        self.data = []

        # Estimation du nombre de tokens
        for src in self.sources:
            print("Querying", src["path"], src["name"])
            ds = load_dataset(
                src["path"],
                src.get("name"),
                split=self.split,
                streaming=True,
            )
            print(f"Received dataset {ds} from {src['name']}")
            self.data.append(ds)
            approx_length = _get_num_rows(src["path"], src["name"], split)
            self._length += approx_length

    def __iter__(self) -> tuple[Tensor]:
        buffer = []
        eof_id = self.tokenizer._special_tokens["<|endoftext|>"]
        ended_stream = set()
        stream_num = len(self.data)
        round_num = 0

        while len(ended_stream) < stream_num:
            for idx, ds in enumerate(self.data):

                # Pour éviter d'obtenir meme docs a chaque fois
                ds = ds.shuffle(buffer_size=1000, seed=self.seed + round_num)
                weight = self.sources[idx]["weight"]
                src = self.sources[idx]["path"]
                it = iter(ds)

                print(f"Start: {_rss_mb():.2f} MB")

                for _ in range(weight):


                    print("### PULLING FROM: ", src, "###")


                    print(f"Before tokenization : {_rss_mb():.2f}")
                    tokens = get_tokens(it, tokenizer=self.tokenizer, suffix=eof_id)
                    print(f"After tokenization: {_rss_mb():.2f}")
                    if tokens is None:
                        ended_stream.add(src)
                        break


                    buffer.extend(tokens)

                    print(f"RSS: {_rss_mb():.2f} MB  | Buffer: {len(buffer)}")

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

                del it
                round_num += 1
                print(f"End : {_rss_mb():.2f} MB")

    def __len__(self) -> int:
        return self._length


def main() -> None:

    sources = [
        {
            "path": "HuggingFaceFW/fineweb-2",
            "name": "fra_Latn",
            "weight": 1,
        },
        {
            "path": "HuggingFaceFW/fineweb",
            "name": "CC-MAIN-2024-10",
            "weight": 1,
        },
        # {
        #     "path": "dhlak/finewebedu-10b-gpt2-tokenized",
        #     "name": "default",
        #     "weight": 1,
        # },
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
