from dataclasses import dataclass, field
import torch


@dataclass
class Config:
    # -----------------------------
    # Paths
    # -----------------------------
    train_zip: str = "/content/train.zip"
    test_zip: str = "/content/test.zip"
    train_csv: str = "/content/train.csv"
    test_csv: str = "/content/test.csv"

    train_dir: str = "train"
    test_dir: str = "test"

    model_path: str = "best_model.pt"
    submission_path: str = "submission.csv"
    plot_path: str = ""  # 예: "learning_curves.png". 비워두면 화면 출력만 수행

    # -----------------------------
    # Audio / Mel-spectrogram
    # -----------------------------
    sr: int = 16000
    n_mels: int = 128
    hop_length: int = 256
    max_len: int = 64

    # -----------------------------
    # SpecAugment
    # -----------------------------
    time_mask: int = 4
    freq_mask: int = 5

    # -----------------------------
    # Training
    # -----------------------------
    batch_size: int = 64
    epochs: int = 50
    lr: float = 1e-3
    weight_decay: float = 1e-4
    patience: int = 5
    seed: int = 42
    val_size: float = 0.2
    num_workers: int = 2
    threshold: float = 0.5

    device: torch.device = field(
        default_factory=lambda: torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )
