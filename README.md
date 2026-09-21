# Patrick aka modded GPT2

This repository hosts the training scripts and architecture of
a modded version of the GPT2 including a modded version of Deepseek V2.

Both implementation are just me experimenting with how
Language Model actually works under the hood.

## Models cards and Performance

GPT2 implementation uses standard MultiHead Attention
while the Deepseek one uses MultiHead Latent Attention
**without** Lightning Indexer (too much headaches).

Granted, I moddified The feed-forward Layer and
Activation function of both (SwiGLU instead of ReLU).

### GPT2 (Patrick)

* Parameters count: 191 M (< 800 MB)
* Total tokens seen = ~ 6M
* Best Loss: 2.96
* Total steps count: ~ 15.000 (~ 24 Hours on P100)

The model can generate somewhat coherent sentences in French and in
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

More promising than the latter. Can distinguish the two languages it was
trained on better than the previous one. Broader knowlegde of common facts
since it was trained on the [wikimedia](https://huggingface.co/datasets/OpenLLM-France/wikimedia/)
and [fineweb-edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb).
Alas his sentences aren't as coherent as the gpt model and it still produce garbage
characters. Possibily because it is undertrained.

### How to run

The weight are freely avalaible on Kaggle:

* [Deepseek](https://www.kaggle.com/models/definitlynotme/patrick-lm/pyTorch/inter)
* [GPT2](https://www.kaggle.com/models/definitlynotme/patrick-gpt2)

A notebook to try them on Google Collab is coming soon.

### Acknowlegdements

* Most my infos come from [Sebastian Raschka repo](https://github.com/rasbt/LLMs-from-scratch)
and his numerous blogs.

* [The modded gpt community](https://github.com/KellerJordan/modded-nanogpt/discussions)

* 3Blue1Brown [LLM series](https://youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi&si=kjEhoe_Yce0z7QgL) which I needn't introduce

* The severly underated [Dr. Jia-Bing Huang](https://youtube.com/@jbhuang0604?si=kVSOSrIFfv5nt_I7)
