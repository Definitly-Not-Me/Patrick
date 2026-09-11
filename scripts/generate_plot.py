import pandas as pand
import readline
import matplotlib.pyplot as plt
from pathlib import Path

CSV_HEADER = ["step", "train_loss", "val_loss", "grad_norm", "lr"]


def gen_plot(path: Path) -> None:
    df = pand.read_csv(path)
    figure, axes = plt.subplots(2, 1, sharex=True)
    filename = Path(f"assets/training_V{df.at[0, 'epoch']}.png")


    axes[0].plot(df["step"], df["train_loss"], label="train_loss", alpha=0.7)
    axes[0].plot(df["step"], df["val_loss"], label="val_loss")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Gradient norm
    axes[1].plot(df["step"], df["grad_norm"], color="orange", label="grad_norm")
    axes[1].plot(df["step"], df["lr"], color="green", label="lr")
    axes[1].set_ylabel("Grad Norm, Learning Rate")
    axes[1].set_xlabel("Step")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(filename)
    plt.show()


if __name__ == "__main__":
    try:
        path = Path(input(">> Path of the csv log file: "))
        gen_plot(path)
    except OSError as e:
        print(e)
    except KeyboardInterrupt:
        exit(0)

    if any(Path("assests/").glob("training_V*.png")):
        print("Successfully generated graph")
    else:
        print("Failed to save plot")
