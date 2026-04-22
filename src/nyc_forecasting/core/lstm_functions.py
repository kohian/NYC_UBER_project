import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import gcsfs
import copy


def train_one_epoch(model, loader: DataLoader, optimizer, criterion, device: str) -> float:
    model.train()
    total_loss = 0.0
    total_count = 0

    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device).squeeze(1)

        optimizer.zero_grad()
        pred = model(xb)
        loss = criterion(pred, yb)
        loss.backward()
        optimizer.step()

        batch_size = xb.size(0)
        total_loss += loss.item() * batch_size
        total_count += batch_size

    return total_loss / total_count


@torch.no_grad()
def evaluate(model, loader: DataLoader, criterion, device: str) -> float:
    model.eval()
    total_loss = 0.0
    total_count = 0

    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device).squeeze(1)
        pred = model(xb)
        loss = criterion(pred, yb)

        batch_size = xb.size(0)
        total_loss += loss.item() * batch_size
        total_count += batch_size

    return total_loss / total_count


@torch.no_grad()
def predict(model, loader: DataLoader, device: str) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    preds = []
    targets = []

    for xb, yb in loader:
        xb = xb.to(device)
        pred = model(xb).cpu().numpy()
        true = yb.squeeze(1).cpu().numpy()
        preds.append(pred)
        targets.append(true)

    return np.concatenate(preds, axis=0), np.concatenate(targets, axis=0)


# def fit_lstm(
#     model,
#     train_loader: DataLoader,
#     val_loader: DataLoader,
#     device: str,
#     learning_rate: float,
#     num_epochs: int,
#     best_model_path: str,  # now string (gs://...)
# ) -> tuple[object, dict]:

#     criterion = nn.MSELoss()
#     optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

#     best_val_loss = float("inf")
#     best_state_dict = None

#     history = {"train_loss": [], "val_loss": []}

#     for epoch in range(1, num_epochs + 1):
#         train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
#         val_loss = evaluate(model, val_loader, criterion, device)

#         history["train_loss"].append(train_loss)
#         history["val_loss"].append(val_loss)

#         print(f"Epoch {epoch:02d} | train_loss={train_loss:.6f} | val_loss={val_loss:.6f}")

#         if val_loss < best_val_loss:
#             best_val_loss = val_loss
#             # best_state_dict = model.state_dict()
#             best_state_dict = copy.deepcopy(model.state_dict())
#             print("  Found new best model")

#     if best_state_dict is None:
#         raise RuntimeError("No best model was captured during training.")
#     # Load best weights into model
#     model.load_state_dict(best_state_dict)

#     # Save to GCS
#     fs = gcsfs.GCSFileSystem()
#     with fs.open(best_model_path, "wb") as f:
#         torch.save(best_state_dict, f)

#     print(f"Saved best model to {best_model_path}")

#     return model, history



def fit_lstm(
    model,
    train_loader,
    val_loader,
    device: str,
    learning_rate: float,
    num_epochs: int,
):
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    best_val_loss = float("inf")
    best_state_dict = None

    history = {"train_loss": [], "val_loss": []}

    for epoch in range(1, num_epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss = evaluate(model, val_loader, criterion, device)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        print(f"Epoch {epoch:02d} | train_loss={train_loss:.6f} | val_loss={val_loss:.6f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state_dict = copy.deepcopy(model.state_dict())
            print("  Found new best model")

    if best_state_dict is None:
        raise RuntimeError("No best model found")

    model.load_state_dict(best_state_dict)

    return model, history, best_state_dict