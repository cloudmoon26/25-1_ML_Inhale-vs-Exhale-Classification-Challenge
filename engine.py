import torch
from sklearn.metrics import roc_auc_score
from torch.cuda.amp import GradScaler, autocast


def make_scaler(device):
    # cuda가 없으면 GradScaler는 자동 비활성화된다.
    return GradScaler(enabled=(device.type == "cuda"))


def train_one_epoch(model, loader, criterion, optimizer, scaler, cfg):
    model.train()

    total_loss = 0.0
    total_correct = 0
    total = 0

    for x, y in loader:
        x = x.to(cfg.device, non_blocking=True)
        y = y.to(cfg.device, non_blocking=True)

        optimizer.zero_grad()

        with autocast(enabled=(cfg.device.type == "cuda")):
            out = model(x)
            loss = criterion(out, y)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item() * x.size(0)
        total_correct += (out.argmax(1) == y).sum().item()
        total += y.size(0)

    return total_loss / total, total_correct / total


@torch.no_grad()
def validate_one_epoch(model, loader, criterion, cfg):
    model.eval()

    total_loss = 0.0
    total_correct = 0
    total = 0

    probs = []
    targets = []

    for x, y in loader:
        x = x.to(cfg.device, non_blocking=True)
        y = y.to(cfg.device, non_blocking=True)

        out = model(x)
        loss = criterion(out, y)

        total_loss += loss.item() * x.size(0)
        total_correct += (out.argmax(1) == y).sum().item()
        total += y.size(0)

        probs.extend(torch.softmax(out, dim=1)[:, 1].detach().cpu().numpy())
        targets.extend(y.detach().cpu().numpy())

    auc = roc_auc_score(targets, probs) if len(set(targets)) > 1 else 0.0

    return total_loss / total, total_correct / total, auc


def fit(model, train_loader, val_loader, criterion, optimizer, scheduler, cfg):
    scaler = make_scaler(cfg.device)

    best = float("inf")
    patience_count = 0

    history = {
        "train_loss": [],
        "val_loss": [],
        "train_acc": [],
        "val_acc": [],
        "val_auc": [],
    }

    for epoch in range(cfg.epochs):
        tr_loss, tr_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler, cfg
        )
        vl_loss, vl_acc, vl_auc = validate_one_epoch(
            model, val_loader, criterion, cfg
        )

        scheduler.step()

        history["train_loss"].append(tr_loss)
        history["val_loss"].append(vl_loss)
        history["train_acc"].append(tr_acc)
        history["val_acc"].append(vl_acc)
        history["val_auc"].append(vl_auc)

        print(
            f"Epoch {epoch + 1:02d} | "
            f"Train {tr_loss:.4f}/{tr_acc:.4f} | "
            f"Val {vl_loss:.4f}/{vl_acc:.4f} | AUC {vl_auc:.4f}"
        )

        if vl_loss < best:
            best = vl_loss
            patience_count = 0
            torch.save(model.state_dict(), cfg.model_path)
            print("  ↳ best model saved")
        else:
            patience_count += 1
            if patience_count >= cfg.patience:
                print("  ↳ Early stopping")
                break

    return history
