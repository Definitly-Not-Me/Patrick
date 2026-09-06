import math
import time
import torch
from torch import Tensor, device
from torch.optim.optimizer import Optimizer
from torch.utils.data.dataloader import DataLoader
from model import GPTModel
from monitor import Monitor
from tiktoken import Encoding


# -------------- Fonctions auxiliaire ----------------------------------------
def text_to_tokens(text: str, tokenizer: Encoding) -> Tensor:
    """
    Transforme du text en une représentation vectorielle i.e embedding
    """
    tokens_ids = tokenizer.encode(text, allowed_special={"<|EOF|>"})
    encoded_tensor = torch.tensor(tokens_ids).unsqueeze(0)

    return encoded_tensor


def tokensIds_to_text(tokens_ids: Tensor, tokenizer: Encoding) -> str:
    """
    Transform un représentation vectorielle (embedding) en text
    """
    flat = tokens_ids.squeeze(0)

    return tokenizer.decode(flat.tolist())


def calc_loss_batch(
    input_batch: Tensor, target_batch: Tensor, model: GPTModel, dev: device
) -> Tensor:
    """
    Calcule la fonction de perte pour une configuration  model
    """
    input_batch, target_batch = input_batch.to(dev), target_batch.to(dev)
    logits = model(input_batch)
    loss = torch.nn.functional.cross_entropy(
        logits.flatten(0, 1), target_batch.flatten()
    )
    return loss


def calc_loss_loader(
    data_loader: DataLoader, model: GPTModel, dev: device, num_batches: int = -1
) -> float:
    """
    Evalue le model sur un certain nombre d'examples du dataset
    Args:
         data_loader: Un objet de la class DataLoader contenant le dataser
         model: Le model a évalué
         dev : l'unité de calcul (cpu vs gpu) où éffectué le calcul
         num_batch  : le nombre d'examples à considére
    Returns:
        total_loss: la somme de toutes les pertes sur l'ensemble des examples
        pris en compte
    """
    total_loss: float = 0.0
    data_size = len(data_loader)

    if data_size == 0:
        raise ValueError(f"DataLoader {data_loader} refers to an empty Dataset")
    elif num_batches < 0:
        num_batches = data_size
    else:
        ## Plafonne le nombre d'examples
        num_batches = min(num_batches, data_size)

    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i < num_batches:
            loss = calc_loss_batch(input_batch, target_batch, model, dev)
            total_loss += loss.item()
        else:
            break

    return total_loss / num_batches


def evaluate_model(
    model: GPTModel,
    train_loader: DataLoader,
    val_loader: DataLoader,
    dev: device,
    num_batches: int,
) -> tuple[float, float]:
    """
    Compare la performance du model sur les données d'entraînement et ceux de
    validation
    Args:
        num_batches: la taille de l'échantillon de chaque dataset a considérer
        train_loader: le dataloader correspondant au datset d'entraînement
        val_loader: le dataloader correspondant au dataset de validation
    Returns:
        train_loss, val_loss: perf sur les données d'entraînement, perf sur ceux
        de validation
    """
    model.eval()
    with torch.no_grad():
        train_loss = calc_loss_loader(train_loader, model, dev, num_batches)
        val_loss = calc_loss_loader(val_loader, model, dev, num_batches)

    model.train()
    return train_loss, val_loss


def generate_and_print_sample(model, tokenizer, start_context, context_size, dev):
    model.eval()
    encoded = text_to_tokens(start_context, tokenizer).to(dev)
    out = model.generate(encoded, max_new_tokens=20, context_size=context_size)
    print(tokensIds_to_text(out, tokenizer))
    model.train()


# -----------------------------------------------------------------------------


def train_model(
    model: GPTModel,
    train_loader: DataLoader,
    val_loader: DataLoader,
    optimizer: Optimizer,
    num_epochs: int,
    eval_freq: float,
    num_batches: int,
    start_context,
    tokenizer: Encoding,
    warmup_steps: int,
    learn_rate_0: float = 3e-5,
    min_learn_rate: float = 1e-6,
    dev: device = torch.device("cpu"),
    save_path: str | None = None,
) -> tuple[list, list, list, list]:
    """
    Entraînement avec warmup, cosine decay, optimisation GD et gradient cliping :P
    Sauvegarde automatiquement le modèle en cas d'interruption (Ctrl+C).
    """
    train_losses, val_losses, track_tokens, track_lrs = [], [], [], []
    tokens_seen, global_step = 0, -1
    learn_rate = learn_rate_0

    # Maximum taux d'apprentissage
    max_learn_rate = optimizer.param_groups[0]["lr"]

    # Nombre total d'itérations
    total_training_steps = len(train_loader) * num_epochs

    # Taux d'Incrementation de l'apprentissage
    lr_increment = (max_learn_rate - learn_rate_0) / warmup_steps

    m = Monitor()

    def _save_checkpoint(tag: str):
        if save_path is None:
            return
        path = save_path.replace(".pt", f"_{tag}.pt")
        torch.save(
            {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "global_step": global_step,
                "epoch": epoch,
            },
            path,
        )
        print(f"\nCheckpoint sauvegardé: {path}")

    try:
        for epoch in range(num_epochs):
            model.train()
            epoch_start = time.time()

            for input_batch, target_batch in train_loader:
                optimizer.zero_grad()
                global_step += 1

                ## Cosine decay & lr warmup
                if global_step < warmup_steps:
                    learn_rate += global_step * lr_increment
                else:
                    progress = (global_step - warmup_steps) / (
                        total_training_steps - warmup_steps
                    )
                    learn_rate = min_learn_rate + (
                        max_learn_rate - min_learn_rate
                    ) * 0.5 * (1 + math.cos(math.pi * progress))

                for param_group in optimizer.param_groups:
                    param_group["lr"] = learn_rate
                track_lrs.append(learn_rate)

                # Backpropagation
                loss = calc_loss_batch(input_batch, target_batch, model, dev)
                loss.backward()

                # Gradient clipping + norm tracking
                grad_norm = (
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    if global_step >= warmup_steps
                    else 0.0
                )

                optimizer.step()
                tokens_seen += input_batch.numel()

                # Tokens per sec
                batch_elapsed = time.time() - epoch_start
                tps = tokens_seen / batch_elapsed if batch_elapsed > 0 else 0

                m.log(
                    step=global_step,
                    epoch=epoch + 1,
                    train_loss=loss.item(),
                    lr=learn_rate,
                    grad_norm=float(grad_norm),
                    tokens_per_sec=tps,
                    tokens_seen=tokens_seen,
                )

                ## Evaluation
                if global_step % eval_freq == 0:
                    train_loss, val_loss = evaluate_model(
                        model, train_loader, val_loader, dev, num_batches
                    )
                    train_losses.append(train_loss)
                    val_losses.append(val_loss)
                    track_tokens.append(tokens_seen)

                    m.log(
                        step=global_step,
                        epoch=epoch + 1,
                        val_loss=val_loss,
                        grad_norm=float(grad_norm),
                        tokens_per_sec=tps,
                        tokens_seen=tokens_seen,
                    )

                    print(
                        f"\n{'=' * 15} SAMPLE (step {global_step}, epoch {epoch + 1}) {'=' * 15}"
                    )
                    print(f"Input:  {start_context}")
                    generate_and_print_sample(
                        model, tokenizer, start_context, 1024, dev
                    )
                    print("=" * 60)

    except KeyboardInterrupt:
        _save_checkpoint("interrupted")
        m.close()
        raise

    m.close()
    return train_losses, val_losses, track_tokens, track_lrs


if __name__ == "__main__":
    from pathlib import Path
    from tiktoken import get_encoding
    from config import GPTConfig
    from data import create_dataloader_v1
    import os

    # ── Load all .txt and .md files ──
    project_root = Path(__file__).resolve().parent.parent
    text_dir = project_root / "text"
    files = sorted(
        p
        for p in text_dir.glob("*.*")
        if p.suffix in (".txt", ".md") and p.name not in ("README.md", "TODO.md")
    )
    print(f"Loading {len(files)} files:")
    corpus = ""
    for f in files:
        text = f.read_text(encoding="utf-8")
        print(f"  {f.name}: {len(text):,} chars")
        corpus += text + "<|EOF|>\n\n"
    print(f"Total: {len(corpus):,} chars\n")

    # ── Device ──
    if os.getenv("testing") == "1":
        dev = torch.device("cpu")
    elif torch.cuda.is_available():
        dev = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        dev = torch.device("mps")
    else:
        dev = torch.device("cpu")
    print(f"Device: {dev}\n")

    # ── Dataloaders (90/10 split) ──
    split = int(len(corpus) * 0.9)
    train_text = corpus[:split]
    val_text = corpus[split:]

    # ── Config ──
    config = GPTConfig(vocab_size=50257)

    batch_size = 4
    max_length = config.context_length
    stride = max_length

    train_loader = create_dataloader_v1(train_text, batch_size, max_length, stride)
    val_loader = create_dataloader_v1(
        val_text, batch_size, max_length, stride, shuffle=False
    )

    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}\n")

    # ── Model + Optimizer ──
    model = GPTModel(config).to(dev)
    model._size()

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.1)

    # ── Tokenizer + start context ──
    tokenizer = get_encoding("gpt2")
    start_context = "Bonjour, je suis"

    # ── Train ──
    save_path = str(project_root / "model_checkpoint.pt")
    train_losses, val_losses, track_tokens, track_lrs = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        num_epochs=10,
        eval_freq=10,
        num_batches=2,
        start_context=start_context,
        tokenizer=tokenizer,
        warmup_steps=10,
        learn_rate_0=1e-5,
        min_learn_rate=1e-6,
        dev=dev,
        save_path=save_path,
    )

    # ── Save ──
    final_path = project_root / "model_final.pt"
    torch.save(model.state_dict(), final_path)
    print(f"\nModel saved to {final_path}")
