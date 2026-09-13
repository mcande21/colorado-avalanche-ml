"""Training loop, loss functions, and inference for hierarchical LSTM."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import torch
from torch import nn

from avalanche_ml.features.alignment import PROBLEM_TYPE_FLAGS
from avalanche_ml.models.lstm_model import HierarchicalLSTM


class FocalLoss(nn.Module):
    def __init__(self, gamma: float = 2.0, alpha: torch.Tensor | None = None):
        super().__init__()
        self.gamma = gamma
        self.register_buffer("alpha", alpha)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred = pred.clamp(1e-7, 1 - 1e-7)
        pt = target * pred + (1 - target) * (1 - pred)
        focal_weight = (1 - pt) ** self.gamma
        bce = -(target * pred.log() + (1 - target) * (1 - pred).log())
        loss = focal_weight * bce
        if self.alpha is not None:
            alpha_t = target * self.alpha + (1 - target) * (1 - self.alpha)
            loss = alpha_t * loss
        return loss.mean()


def combined_loss(
    outputs: dict[str, torch.Tensor],
    labels: dict[str, torch.Tensor],
    lambda_stage2: float = 1.0,
    lambda_multi: float = 0.5,
    focal_gamma: float = 2.0,
) -> torch.Tensor:
    focal = FocalLoss(gamma=focal_gamma)

    problem_target = torch.stack(
        [labels[k].float() for k in PROBLEM_TYPE_FLAGS], dim=1,
    )
    loss_stage1 = focal(outputs["problem_types"], problem_target)

    loss_stage2_binary = focal(
        outputs["danger_binary"].squeeze(-1), labels["danger_binary"].float(),
    )

    danger_target = (labels["danger_level"] - 1).clamp(0, 4)
    loss_stage2_multi = nn.functional.cross_entropy(
        outputs["danger_distribution"], danger_target,
    )

    return loss_stage1 + lambda_stage2 * loss_stage2_binary + lambda_multi * loss_stage2_multi


def train_lstm(
    model: HierarchicalLSTM,
    train_loader,
    val_loader,
    epochs: int = 100,
    lr: float = 1e-3,
    weight_decay: float = 1e-5,
    patience: int = 10,
    max_grad_norm: float = 1.0,
    device: str = "cpu",
) -> tuple[HierarchicalLSTM, dict]:
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=5, factor=0.5,
    )

    history: dict = {
        "train_loss": [],
        "val_loss": [],
        "lr": [],
        "max_grad_norm": max_grad_norm,
    }
    best_val_loss = float("inf")
    best_state = None
    wait = 0

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        n_batches = 0
        for batch in train_loader:
            inputs = {
                "branch1": batch["branch1"].to(device),
                "branch2": batch["branch2"].to(device),
                "branch3": batch["branch3"].to(device),
                "branch3_lengths": batch["branch3_lengths"],
            }
            labels = {k: v.to(device) for k, v in batch["labels"].items()}
            optimizer.zero_grad()
            outputs = model(**inputs)
            loss = combined_loss(outputs, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1

        avg_train = total_loss / max(n_batches, 1)
        history["train_loss"].append(avg_train)

        val_loss = _validate(model, val_loader, device)
        history["val_loss"].append(val_loss)
        history["lr"].append(optimizer.param_groups[0]["lr"])

        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, history


def _validate(model: HierarchicalLSTM, loader, device: str) -> float:
    model.eval()
    total = 0.0
    n = 0
    with torch.no_grad():
        for batch in loader:
            inputs = {
                "branch1": batch["branch1"].to(device),
                "branch2": batch["branch2"].to(device),
                "branch3": batch["branch3"].to(device),
                "branch3_lengths": batch["branch3_lengths"],
            }
            labels = {k: v.to(device) for k, v in batch["labels"].items()}
            outputs = model(**inputs)
            loss = combined_loss(outputs, labels)
            total += loss.item()
            n += 1
    return total / max(n, 1)


def predict(
    model: HierarchicalLSTM,
    dataloader,
    device: str = "cpu",
) -> pd.DataFrame:
    model.eval()
    rows = []
    with torch.no_grad():
        for batch in dataloader:
            inputs = {
                "branch1": batch["branch1"].to(device),
                "branch2": batch["branch2"].to(device),
                "branch3": batch["branch3"].to(device),
                "branch3_lengths": batch["branch3_lengths"],
            }
            outputs = model(**inputs)
            bs = outputs["problem_types"].shape[0]
            for i in range(bs):
                row = {
                    "station_id": batch["metadata"]["station_id"][i],
                    "date": batch["metadata"]["date"][i],
                    "elevation_band": batch["metadata"]["elevation_band"][i],
                }
                for j, name in enumerate(PROBLEM_TYPE_FLAGS):
                    row[f"prob_{name}"] = outputs["problem_types"][i, j].item()
                row["prob_danger_binary"] = outputs["danger_binary"][i, 0].item()
                for j in range(outputs["danger_distribution"].shape[1]):
                    row[f"prob_danger_{j + 1}"] = outputs["danger_distribution"][i, j].item()
                rows.append(row)
    return pd.DataFrame(rows)


def save_model(
    model: HierarchicalLSTM,
    path: str,
    metadata: dict | None = None,
) -> None:
    torch.save(model.state_dict(), path)
    if metadata is not None:
        meta_path = Path(path).with_name(
            Path(path).stem + "_metadata.json",
        )
        meta_path.write_text(json.dumps(metadata, indent=2))


def load_model(path: str) -> HierarchicalLSTM:
    model = HierarchicalLSTM()
    model.load_state_dict(torch.load(path, weights_only=True))
    return model
