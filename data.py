import os
import random

import numpy as np

# numpy <-> librosa 호환 패치
# 일부 환경에서 librosa가 np.complex / np.float을 참조할 수 있어 유지한다.
if not hasattr(np, "complex"):
    np.complex = complex
if not hasattr(np, "float"):
    np.float = float

import librosa
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset


def add_noise(y, factor=0.005):
    return y + factor * np.random.randn(len(y))


def time_stretch(y, rate):
    return librosa.effects.time_stretch(y=y, rate=rate)


def pitch_shift(y, sr, n):
    return librosa.effects.pitch_shift(y, sr=sr, n_steps=n)


def spec_augment(spec, time_mask, freq_mask):
    spec = spec.copy()

    # time mask
    t0 = random.randint(0, max(0, spec.shape[1] - time_mask))
    spec[:, t0:t0 + time_mask] = 0

    # freq mask
    f0 = random.randint(0, max(0, spec.shape[0] - freq_mask))
    spec[f0:f0 + freq_mask, :] = 0

    return spec


class BreathDataset(Dataset):
    def __init__(self, df, audio_dir, cfg, augment=False):
        self.df = df.reset_index(drop=True)
        self.audio_dir = audio_dir
        self.cfg = cfg
        self.augment = augment

    def __len__(self):
        return len(self.df)

    def _id_to_filename(self, sample_id):
        # 원본 코드의 파일명 처리 로직 유지
        fname = sample_id.replace("_I_", "_").replace("_E_", "_")
        if not fname.endswith(".wav"):
            fname += ".wav"
        return fname

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        sample_id = row["ID"]
        label = row["label"] if "label" in row.index else None

        fname = self._id_to_filename(sample_id)
        audio_path = os.path.join(self.audio_dir, fname)

        y, _ = librosa.load(audio_path, sr=self.cfg.sr)

        # -----------------------------
        # waveform-level augmentation
        # -----------------------------
        if self.augment:
            if random.random() < 0.3:
                y = add_noise(y, factor=random.uniform(0.002, 0.01))

            if random.random() < 0.3:
                y = time_stretch(y, random.uniform(0.9, 1.1))
                y = np.pad(y, (0, max(0, self.cfg.sr - len(y))))[:self.cfg.sr]

            if random.random() < 0.3:
                y = pitch_shift(y, self.cfg.sr, random.randint(-1, 1))

        # -----------------------------
        # Mel-spectrogram
        # -----------------------------
        spec = librosa.feature.melspectrogram(
            y=y,
            sr=self.cfg.sr,
            n_mels=self.cfg.n_mels,
            hop_length=self.cfg.hop_length,
        )
        spec = librosa.power_to_db(spec, ref=1.0)

        spec = np.pad(
            spec,
            ((0, 0), (0, max(0, self.cfg.max_len - spec.shape[1]))),
            mode="constant",
        )[:, :self.cfg.max_len]

        if self.augment:
            spec = spec_augment(spec, self.cfg.time_mask, self.cfg.freq_mask)

        tensor = torch.tensor(spec).unsqueeze(0).float()  # (1, n_mels, time)

        if label is not None:
            return tensor, torch.tensor(label, dtype=torch.long)

        return tensor, sample_id


def load_train_dataframe(cfg):
    train_df = pd.read_csv(cfg.train_csv)
    train_df["label"] = train_df["Target"].map({"I": 0, "E": 1})
    return train_df


def build_train_val_loaders(cfg):
    train_df = load_train_dataframe(cfg)

    train_idx, val_idx = train_test_split(
        np.arange(len(train_df)),
        test_size=cfg.val_size,
        stratify=train_df["label"].values,
        random_state=cfg.seed,
    )

    train_dataset = BreathDataset(
        train_df.iloc[train_idx],
        audio_dir=cfg.train_dir,
        cfg=cfg,
        augment=True,
    )
    val_dataset = BreathDataset(
        train_df.iloc[val_idx],
        audio_dir=cfg.train_dir,
        cfg=cfg,
        augment=False,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader


def build_test_loader(cfg):
    test_df = pd.read_csv(cfg.test_csv)

    test_dataset = BreathDataset(
        test_df,
        audio_dir=cfg.test_dir,
        cfg=cfg,
        augment=False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True,
    )

    return test_loader
