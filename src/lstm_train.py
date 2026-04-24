



import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import classification_report, confusion_matrix

DATA_DIR = "data/processed"
BATCH_SIZE = 2048
EPOCHS = 5
LR = 1e-3
SEED = 42
NUM_WORKERS = 4


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class NpyDataset(Dataset):
    def __init__(self, x_path: str, y_path: str):
        self.X = np.load(x_path, mmap_mode="r")
        self.y = np.load(y_path, mmap_mode="r")

    def __len__(self) -> int:
        return self.X.shape[0]

    def __getitem__(self, idx: int):
        x = torch.tensor(self.X[idx], dtype=torch.float32)
        y = torch.tensor(int(self.y[idx]), dtype=torch.long)
        return x, y


class LSTMBaseline(nn.Module):
    def __init__(self, input_size: int = 46, hidden_size: int = 64, num_layers: int = 1):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(1)  # [batch, 46] -> [batch, 1, 46]
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        return self.fc(out)


def evaluate_lstm(model: nn.Module, loader: DataLoader, device: torch.device):
    model.eval()
    y_true_list = []
    y_pred_list = []

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            logits = model(x)
            preds = torch.argmax(logits, dim=1)

            y_true_list.append(y.cpu().numpy())
            y_pred_list.append(preds.cpu().numpy())

    y_true = np.concatenate(y_true_list)
    y_pred = np.concatenate(y_pred_list)
    return y_true, y_pred


def main() -> None:
    set_seed(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)
    if device.type == "cuda":
        print("gpu:", torch.cuda.get_device_name(0))

    train_ds = NpyDataset(f"{DATA_DIR}/X_train.npy", f"{DATA_DIR}/y_train.npy")
    test_ds = NpyDataset(f"{DATA_DIR}/X_test.npy", f"{DATA_DIR}/y_test.npy")

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    model = LSTMBaseline(input_size=46).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()

    start_time = time.time()

    for epoch in range(1, EPOCHS + 1):
        model.train()
        running_loss = 0.0

        for x, y in train_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_loss = running_loss / len(train_loader)
        print(f"epoch {epoch}/{EPOCHS} loss={avg_loss:.4f}")

    total_time = time.time() - start_time
    print("train time (s):", round(total_time, 2))

    y_true, y_pred = evaluate_lstm(model, test_loader, device)

    print("\nConfusion matrix:")
    print(confusion_matrix(y_true, y_pred))

    print("\nReport:")
    print(classification_report(y_true, y_pred, digits=4))


if __name__ == "__main__":
    main()
