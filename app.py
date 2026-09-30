"""
Patrick — Chainlit Chat Interface
Usage: chainlit run app.py -w
"""

from pathlib import Path
import torch
from torch import Tensor
import chainlit as cl
from chainlit.input_widget import Select, Slider
import tiktoken
from tiktoken import Encoding
import train
import asyncio
from train import GPTConfig, GPTModel, GPTModelV2

dev = torch.device("cpu")
MODELS = {
    "gpt2": dict(
        path=Path("./_checkpoints/model_checkpoint_best_model_v1.pt"),
        description="GPT-2 implementation, not fine-tuned. Perform better in english",
    ),
    "deepseek": dict(
        path=Path("./_checkpoints/deepseek_checkpoint_best-round1.pt"),
        description="DeepSeek implementation, not fine-tuned. Multilingual",
    ),
}
DEFAULT_MODEL = "gpt2"
tokenizer = tiktoken.get_encoding("gpt2")


# ── Model loading ──────────────────────────────────────────────────────────
def load_model(
    filepath: Path, config: GPTConfig, variant: str
) -> GPTModelV2 | GPTModel:

    model = GPTModel(config) if variant == "gpt2" else GPTModelV2(config)
    model.to(device=dev)
    checkpoint = torch.load(filepath, map_location=dev, weights_only=False)
    model.load_state_dict(checkpoint["model"], assign=True, strict=True)
    model.eval()

    return model


# -------------- Fonctions auxiliaires ----------------------------------------
def text_to_tokens(text: str, tokenizer: Encoding) -> Tensor:
    """
    Transforme du text en une représentation vectorielle i.e embedding
    """
    tokens_ids = tokenizer.encode_ordinary(text)
    encoded_tensor = torch.tensor(tokens_ids).unsqueeze(0)

    return encoded_tensor


def tokensIds_to_text(tokens_ids: Tensor, tokenizer: Encoding) -> str:
    """
    Transform un représentation vectorielle (embedding) en text
    """
    flat = tokens_ids.squeeze(0)

    return tokenizer.decode(flat.tolist(), errors="strict")

# ── Chat lifecycle ─────────────────────────────────────────────────────────
@cl.on_chat_start
async def start():
    # Load default model
    variant = DEFAULT_MODEL
    model = load_model(MODELS[variant]["path"], GPTConfig(top_k=10), variant)
    cl.user_session.set("model", model)
    cl.user_session.set("variant", variant)

    # Send settings panel
    await cl.ChatSettings(
        [
            Select(
                id="model",
                label="Model variant",
                values=list(MODELS.keys()),
                initial_index=0,
            ),
            Slider(
                id="max_new_tokens",
                label="Max new tokens",
                initial=64,
                min=10,
                max=500,
                step=1,
            ),
            Slider(
                id="temperature",
                label="Temperature",
                initial=0.8,
                min=0.1,
                max=2.0,
                step=0.1,
            ),
            Slider(
                id="top_k",
                label="Top-K",
                initial=10,
                min=1,
                max=100,
                step=1,
            ),
        ]
    ).send()

    await cl.Message(
        content=f"Model **{variant}** loaded — {MODELS[variant]['description']}"
    ).send()


# ── Settings update ────────────────────────────────────────────────────────
@cl.on_settings_update
async def on_settings(settings: dict):
    variant = settings["model"]

    # Reload model if variant changed
    if variant != cl.user_session.get("variant"):
        model = load_model(MODELS[variant]["path"], GPTConfig(top_k=10), variant)
        cl.user_session.set("model", model)
        cl.user_session.set("variant", variant)
        await cl.Message(content=f"Switched to **{variant}**").send()

    # Store all settings
    cl.user_session.set("settings", settings)


# ── Message handler ────────────────────────────────────────────────────────


async def send_animated_message(
    base_msg: str, frames: list[str], interval: float = 0.8
) -> None:
    """Display animated message with minimal resource usage"""
    msg = cl.Message(content=base_msg)
    await msg.send()

    progress = 0
    bar_length = 12  # Optimal length for progress bar

    try:
        while True:
            # Efficient progress calculation
            current_frame = frames[progress % len(frames)]
            progress_bar = ("▣" * (progress % bar_length)).ljust(bar_length, "▢")

            # Single update operation - overwrite entire content
            new_content = f"{current_frame} {base_msg}\n{progress_bar}"
            msg.content = new_content
            await msg.update()

            progress += 1
            await asyncio.sleep(interval)
    except asyncio.CancelledError:
        msg.content = base_msg
        await msg.update()


def _safe_next(gen):
    try:
        return next(gen)
    except StopIteration:
        return None


@cl.on_message
async def on_message(message: cl.Message):
    model: GPTModel | GPTModelV2 = cl.user_session.get("model")
    s = cl.user_session.get("settings") or {}
    max_new = int(s.get("max_new_tokens", 50))
    temp = float(s.get("temperature", 0.8))
    top_k = int(s.get("top_k", 10))

    animation_task = asyncio.create_task(
        send_animated_message(
            "Processing...", ["🌑", "🌒", "🌓", "🌔", "🌕", "🌖", "🌗", "🌘"], 0.8
        )
    )

    start_context = message.content
    eof = tokenizer._special_tokens.get("\n", None)

    msg = cl.Message(content="")

    gen = model.generate(
        text_to_tokens(start_context, tokenizer),
        context_size=1024,
        max_new_tokens=max_new,
        temp=temp,
        top_k=top_k,
        EOF_id=eof,
    )

    ## Garde l'entiéreté du message pour fallback
    # Fix temporaire en attendant que le modèle ne produise
    # plus de charactère '�'
    output_fallback = ""

    first = True

    while True:
        token = await asyncio.to_thread(_safe_next, gen)
        if token is None:
            break

        if first:
            animation_task.cancel()
            await animation_task
            msg.content = ""
            first = False


        try:

            token = tokensIds_to_text(token, tokenizer)
            await msg.stream_token(token)
            output_fallback += token
        except UnicodeDecodeError:
            max_new = abs(max_new - len(output_fallback))
            gen = model.generate(
                text_to_tokens(start_context + output_fallback, tokenizer),
                context_size=1024,
                max_new_tokens=max_new,
                temp=temp,
                top_k=top_k,
                EOF_id=eof,
            )


    await msg.update()
