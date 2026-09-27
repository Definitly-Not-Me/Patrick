# Patrick aka modded GPT2

This repository hosts the training scripts and architecture source code of
a modded version of the GPT2 including a modded version of Deepseek V2.

Both implementation are mostly me experimenting with how
Language Model actually works under the hood.

---

## Perfomance

GPT2 implementation uses standard MultiHead Attention
while the Deepseek one uses MultiHead Latent Attention
**without** Lightning Indexer (too much hassle).

Granted, I moddified The feed-forward Layer and
Activation function of both (SwiGLU instead of ReLU).

### GPT2 (Patrick)

* Parameters count: 191 M (< 800 MB)
* Total tokens seen = ~ 6M
* Best Loss: 2.96
* Total steps count: ~ 15.000 (~ 24 Hours on P100)

The model can generate somewhat coherent sentences in
English demonstrating multinlingual capabilities but
has limited knowlegde of common facts.

The GPT2 model achieves best loss of 2.96 on a mixed dateaset comprising
[fineweb-2](https://huggingface.co/datasets/HuggingFaceFW/fineweb-2)
and [fineweb-edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb).

### Deepseek

* Parameters count: 233 M (< 900 MB)
* Total tokens seen = ~ 2M
* Best Loss: 2.84
* Total steps count: ~ 11.000 (~ 18 hours on P100/T4)

More promising than the former. Can distinguish between two languages (French and English)
it was trained on better corpus. It possesses broader knowledge of common facts
thanks to [wikimedia](https://huggingface.co/datasets/OpenLLM-France/wikimedia/)
and [fineweb-edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb).
Alas his sentences aren't as coherent as the GPT2 model and it still produce garbage
characters. Possibily because it is severly undertrained.

---

### How to run locally

First of all, the requirements:

* torch
* sympy (optional dependency of torch)
* triton (optional dependency of torch)
* networkx (optional dependency of torch)
* mathplotlib (for the graphs)
* pandas
* pydantic-settings
* chainlit (optional if you needn't the web UI)
* aiofiles (chainlit dependency)
* requests
* datasets

#### Training

The cli interface provide a wide range of options
to test the training script locally.

```
python3 train.py --help
```

#### Web UI

First install chainlit

```bash
pip install chainlit
```

Then run the server

```bash
charmlit run app.py --checkpoints your/checkpoint/dir
```

The page should now open in your browser.

### How to deploy on Kaggle/ Collab

Modify the file `kernel-metadata.json` as needed.
Then modify the script `scripts/_gen_notebook.py` to
see the final notebook cells will be to your liking.
After that you just need to 'build' the notebook.

```bash
python3 scripts/gen_notebook.py  # assuming train.py in the root directory

kaggle kernels push  # Assuming you're using kaggle
```

> [!NOTE]
> Even though I did keep usability in mind, I'll have to
> admit that most of interesting part (like datasets streaming) are hardcoded
> for now. Feel free to open an issue to let me know if
> there's some inadequacy

A notebook to try the resulting models on Google Collab is coming soon (hopefully).

---

### Acknowledgements

Most of the what is in this repository wouldn't have been possible without:

* [Sebastian Raschka repo](https://github.com/rasbt/LLMs-from-scratch)

* [The modded gpt community](https://github.com/KellerJordan/modded-nanogpt/discussions)

* 3Blue1Brown [LLM series](https://youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi&si=kjEhoe_Yce0z7QgL)

*  [Dr. Jia-Bing Huang](https://youtube.com/@jbhuang0604?si=kVSOSrIFfv5nt_I7)
