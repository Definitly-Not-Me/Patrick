import time


# ── monitor ──────────────────────────────────────────────────────────────────


class Monitor:
    """Plain-text training monitor.

    Replaces the rich TUI: rich's Live/Console recursion under Kaggle's
    papermill execution, so we print one summary line per eval instead.
    """

    def __init__(self, max_history: int = 300, refresh_per_second: int = 4):
        # max_history / refresh_per_second kept for call-site compatibility.
        self.start_time = time.time()

    def log(
        self,
        *,
        step: int,
        epoch: int | None = None,
        train_loss: float | None = None,
        val_loss: float | None = None,
        lr: float | None = None,
        grad_norm: float | None = None,
        tokens_per_sec: float | None = None,
        tokens_seen: int | None = None,
    ):
        parts = [f"step {step}"]
        if epoch is not None:
            parts.append(f"epoch {epoch}")
        if train_loss is not None:
            parts.append(f"train {train_loss:.4f}")
        if val_loss is not None:
            parts.append(f"val {val_loss:.4f}")
        if lr is not None:
            parts.append(f"lr {lr:.3e}")
        if grad_norm is not None:
            parts.append(f"grad {grad_norm:.3f}")
        if tokens_per_sec is not None:
            parts.append(f"{tokens_per_sec:,.0f} tok/s")
        if tokens_seen is not None:
            parts.append(f"tokens {tokens_seen:,}")
        print("[monitor] " + " | ".join(parts), flush=True)

    def close(self):
        print(f"[monitor] done in {time.time() - self.start_time:.1f}s", flush=True)