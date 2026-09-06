# /usr/bin/env python3

import argparse
import sys
from pathlib import Path

import torch
from tiktoken import get_encoding

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import train


def load_model(ckpt_path: str, device: torch.device, temperature: float | None, top_k: int | None):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = train.GPTConfig(vocab_size=50257, drop_rate=0.1)
    if temperature is not None:
        cfg = cfg.model_copy(update={"temperature": temperature})
    if top_k is not None:
        cfg = cfg.model_copy(update={"top_k": top_k})
    model = train.GPTModel(cfg)
    model.load_state_dict(ckpt["model"])
    return model, cfg


def _autodetect_checkpoint() -> Path:
    candidates = sorted(Path.cwd().glob("*.pt"))
    if not candidates:
        raise SystemExit("[chat] no .pt file found in current directory; pass --checkpoint")
    if len(candidates) > 1:
        print(f"[chat] multiple .pt files: {[c.name for c in candidates]}; using {candidates[-1]}")
    return candidates[-1]


def _default_device() -> torch.device:
    if torch.cuda.is_available() and torch.version.hip is None:
        return torch.device("cuda")
    if torch.cuda.is_available():
        print("[chat] CUDA reports AMD (ROCm); falling back to cpu", flush=True)
    return torch.device("cpu")


def main():
    parser = argparse.ArgumentParser(description="Chat with the trained model")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Path to the saved checkpoint (default: newest .pt in current directory)",
    )
    parser.add_argument("--max-new-tokens", type=int, default=100)
    parser.add_argument("--temperature", type=float, default=None, help="Override the sampling temperature (default: config 0.8)")
    parser.add_argument("--top-k", type=int, default=None, help="Override top-k (default: config 40)")
    parser.add_argument("--device", type=str, default=None, help="cuda / cpu (default: cuda on NVIDIA, else cpu)")
    args = parser.parse_args()
    ckpt_path = args.checkpoint or _autodetect_checkpoint()

    device = torch.device(args.device or _default_device())
    print(f"[chat] loading {ckpt_path} on {device} ...", flush=True)
    model, cfg = load_model(str(ckpt_path), device, args.temperature, args.top_k)
    model.to(device).eval()

    tokenizer = get_encoding("gpt2")
    eof = tokenizer._special_tokens.get("<|endoftext|>", None)
    print("[chat] ready. Ctrl-C / Ctrl-D to exit.\n", flush=True)

    while True:
        try:
            prompt = input("prompt> ")
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not prompt.strip():
            continue

        input_t = train.text_to_tokens(prompt, tokenizer).to(device)
        with torch.no_grad(), torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
            out = model.generate(
                input_t,
                max_new_tokens=args.max_new_tokens,
                context_size=cfg.context_length,
                EOF_id=eof,
            )
        generated = train.tokensIds_to_text(out[0][input_t.shape[1] :], tokenizer)
        print(f"{prompt}{generated}\n", flush=True)


if __name__ == "__main__":
    main()

