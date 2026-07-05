import pandas as pd
import torch


@torch.no_grad()
def predict_test(model, test_loader, cfg):
    model.eval()

    preds = []
    ids = []

    for x, sample_ids in test_loader:
        x = x.to(cfg.device, non_blocking=True)

        probs = torch.softmax(model(x), dim=1)[:, 1].detach().cpu()
        batch_preds = (probs > cfg.threshold).long().numpy()

        preds.extend(batch_preds)
        ids.extend(sample_ids)

    return ids, preds


def save_submission(ids, preds, output_path):
    submission = pd.DataFrame(
        {
            "ID": ids,
            "Target": ["E" if p == 1 else "I" for p in preds],
        }
    )
    submission.to_csv(output_path, index=False)
    print(f"✅ {output_path} 저장 완료")
