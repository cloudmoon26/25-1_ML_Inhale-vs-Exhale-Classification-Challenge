import os
import random
import zipfile

import numpy as np
import torch


def set_seed(seed: int = 42) -> None:
    """재현성을 위한 random seed 고정."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def extract_zip_if_needed(zip_path: str, output_dir: str) -> None:
    """output_dir이 없을 때만 zip을 해제한다."""
    if os.path.isdir(output_dir):
        return

    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(output_dir)


def plot_history(history: dict, save_path: str = "") -> None:
    """Loss / Accuracy / AUC 학습 곡선 시각화."""
    import matplotlib.pyplot as plt

    train_losses = history["train_loss"]
    val_losses = history["val_loss"]
    train_accs = history["train_acc"]
    val_accs = history["val_acc"]
    val_aucs = history["val_auc"]

    plt.figure(figsize=(12, 4))

    plt.subplot(1, 3, 1)
    plt.plot(train_losses, label="Train")
    plt.plot(val_losses, label="Val")
    plt.title("Loss")
    plt.legend()

    plt.subplot(1, 3, 2)
    plt.plot(train_accs, label="Train")
    plt.plot(val_accs, label="Val")
    plt.title("Acc")
    plt.legend()

    plt.subplot(1, 3, 3)
    plt.plot(val_aucs, label="Val AUC")
    plt.title("AUC")
    plt.legend()

    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=150)

    plt.show()
