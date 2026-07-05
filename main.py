import argparse

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR

from config import Config
from data import build_test_loader, build_train_val_loaders
from engine import fit
from model import BreathNet
from predict import predict_test, save_submission
from utils import extract_zip_if_needed, plot_history, set_seed


def parse_args():
    parser = argparse.ArgumentParser()

    # path
    parser.add_argument("--train_zip", type=str, default=None)
    parser.add_argument("--test_zip", type=str, default=None)
    parser.add_argument("--train_csv", type=str, default=None)
    parser.add_argument("--test_csv", type=str, default=None)
    parser.add_argument("--train_dir", type=str, default=None)
    parser.add_argument("--test_dir", type=str, default=None)
    parser.add_argument("--model_path", type=str, default=None)
    parser.add_argument("--submission_path", type=str, default=None)
    parser.add_argument("--plot_path", type=str, default=None)

    # training
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--weight_decay", type=float, default=None)
    parser.add_argument("--patience", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--threshold", type=float, default=None)

    return parser.parse_args()


def update_config_from_args(cfg, args):
    for key, value in vars(args).items():
        if value is not None and hasattr(cfg, key):
            setattr(cfg, key, value)
    return cfg


def main():
    args = parse_args()
    cfg = update_config_from_args(Config(), args)

    print(f"Device: {cfg.device}")

    set_seed(cfg.seed)

    # 원본 코드처럼 train/test 폴더가 없을 때만 zip 해제
    extract_zip_if_needed(cfg.train_zip, cfg.train_dir)
    extract_zip_if_needed(cfg.test_zip, cfg.test_dir)

    train_loader, val_loader = build_train_val_loaders(cfg)

    model = BreathNet().to(cfg.device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(
        model.parameters(),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
    )
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=cfg.epochs,
        eta_min=1e-5,
    )

    history = fit(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        cfg=cfg,
    )

    plot_history(history, save_path=cfg.plot_path)

    test_loader = build_test_loader(cfg)

    model.load_state_dict(torch.load(cfg.model_path, map_location=cfg.device))
    ids, preds = predict_test(model, test_loader, cfg)
    save_submission(ids, preds, cfg.submission_path)


if __name__ == "__main__":
    main()
