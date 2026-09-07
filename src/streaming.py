from datasets import load_dataset
from torch.utils.data import IterableDataset, DataLoader
from tiktoken import Encoding
from torch import Tensor
import tiktoken
import torch
import dotenv
import time
from requests.exceptions import ConnectionError, ChunkedEncodingError

def safe_next(it, max_retries=3, backoff=2.0):
    """Retry on transient network errors, raise on permanent ones."""
    for attempt in range(max_retries):
        try:
            return next(it)
        except (ConnectionError,
                ChunkedEncodingError):
            print(f"Failed to connect to hugging face. Retrying. [{attempt}/{max_retries}]")
            if attempt == max_retries - 1:
                raise
            time.sleep(backoff * (attempt + 1))

def get_tokens(
    it,
    tokenizer,
    suffix: int,
) -> list[int]:

    try:
        example = next(it)
        print(f"Example: {str(example):.99}")
    except StopIteration:
        return []

    if "input_ids" in example:
        # Pre-tokenized
        ids = example["input_ids"]

        return ids if isinstance(ids, list) else [ids]

    elif "text" in example:
        # Raw text
        return tokenizer.encode_ordinary(example["text"]) + [suffix]

    else:
        print("Unexpected response:", example)
        return []

class GPTDatasetV3(IterableDataset):
    """
    Version du dataset tirant ses sources de hugging face.
    Par souci d'optimisation du stockage, cette version
    stream les fichiers et les tokenize ad hoc si besoin
    """

    def __init__(self, sources: list[dict], tokenizer: Encoding, context_length: int, stride: int, seed: int, split: str) -> None:
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


    def __iter__(self) -> tuple[Tensor]:
        buffer = []
        eof_id = self.tokenizer._special_tokens["<|endoftext|>"]
        iters = []
        for src in self.sources:
            print("Querying", src["path"],src["name"])
            ds = load_dataset(
                src["path"],
                src.get("name"),
                split=self.split,
                streaming=True,
            )
            ds = ds.shuffle(buffer_size=10_000, seed=self.seed)

            iters.append((iter(ds), int(src.get("weight", 1))))

        while iters:
            next_iters = []

            for it, weight in iters:
                exhausted = False

                tokens = get_tokens(it, tokenizer=self.tokenizer, suffix=eof_id )
                print(f"From {it} got tokens: \n{str(tokens):.99}")
                buffer.extend(tokens)
                print(f"Updated buffer {buffer[:99]}")
                while len(buffer) >= self.ctx + 1:
                    x = torch.tensor(
                        buffer[:self.ctx],
                        dtype=torch.long,
                    )
                    y = torch.tensor(
                        buffer[1:self.ctx + 1],
                        dtype=torch.long,
                    )

                    yield x, y
                    buffer = buffer[self.stride:]

                if not exhausted:
                    next_iters.append((it, weight))

            iters = next_iters

def main()-> None:
    sources = [
        # {
        #     "path": "HuggingFaceFW/fineweb-2",
        #     "name": "fra_Latn",
        #     "weight": 2,
        # },

        # {
        #     "path": "HuggingFaceFW/fineweb",
        #     "name": "CC-MAIN-2024-10",
        #     "weight": 2,
        # },

        {
            "path": "dhlak/finewebedu-10b-gpt2-tokenized",
            "name": "default",
            "weight": 1,
        }
    ]
    tokenizer = tiktoken.get_encoding("gpt2")
    dataset = GPTDatasetV3(sources=sources, context_length=1024, stride=512, tokenizer=tokenizer, seed=15, split="train")
    train_loader = DataLoader(dataset, batch_size=8, num_workers=2, prefetch_factor=4)

    for tok in train_loader:
        print(tok)

if __name__ == '__main__':
    dotenv.load_dotenv()
    main()
