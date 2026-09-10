import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

OUT = os.path.join(ROOT, "train_notebook.ipynb")
SRC = os.path.join(ROOT, "train.py")

with open(SRC, "r", encoding="utf-8") as fh:
    source = fh.read()

if not source.endswith("\n"):
    source += "\n"

source_lines = source.splitlines(keepends=True)

# Strip the `if __name__ == "__main__": ...` guard so the definition cell
# only defines things. Inside a notebook `__name__ == "__main__"`, so the guard
# would auto-run training; instead we call resume() explicitly in its own cell.
guardless = source_lines[: source_lines.index('if __name__ == "__main__":\n')]
guardless.insert(0, "%%file train.py\n")
cells = []

cells.append(
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "# Train an LLM (Kaggle, P100)\n",
            "\nThe Kaggle container ships PyTorch **2.10+** by default, which is **incompatible "
            "with the P100 (sm_60)**. Install **PyTorch 2.4.0** from the CUDA 12.1 index so the GPU runs.",
        ],
    }
)

cells.append(
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "!pip install --quiet torch==2.4.0 torchvision==0.19.0 --index-url https://download.pytorch.org/whl/cu121\n",
            "!pip install --quiet datasets python-dotenv tiktoken memory_profiler\n",
            "!pip install memory_profiler -q\n",
            "%load_ext memory_profiler"
        ],
    }
)

cells.append(
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "import torch; assert torch.__version__.startswith('2.4.'), f'torch {torch.__version__} wrong'; "
            "print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
        ],
    }
)

cells.append(
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## Model code\n",
            "\nEverything from `train.py` (classes, data pipeline, training loop) lives in the cell below. "
            "It only defines things; the last cell runs training.",
        ],
    }
)

# Full train.py source, minus the trailing `__main__` guard.
cells.append(
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": guardless,

    }
)

cells.append(
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## Resume training\n",
        ],
    }
)

# Run training (resume from the mounted checkpoint).
cells.append(
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "from train import resume\n",
            "resume()"
        ],
    }
)

nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.11"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

with open(OUT, "w", encoding="utf-8") as fh:
    json.dump(nb, fh, indent=1)

print("cells:", len(cells))
print("embedded source lines:", len(guardless))
print("last definition cell tail:", "".join(guardless[-3:]))
print("final cell:", "".join(cells[-1]["source"]))
