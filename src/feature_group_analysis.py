






import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import classification_report, confusion_matrix, f1_score, recall_score

DATA_DIR = "data/processed"
BATCH_SIZE = 2048
EPOCHS = 5
LR = 1e-3
SEED = 42
EPSILONS = [0.01, 0.03, 0.05]
MODEL_TYPE = "lstm"   # "cnn" or "lstm"

# FEATURE_NAMES order must match the processed numpy feature order
FEATURE_NAMES = [
    "flow_duration", "Header_Length", "Protocol Type", "Duration", "Rate",
    "Srate", "Drate", "fin_flag_number", "syn_flag_number",
    "rst_flag_number", "psh_flag_number", "ack_flag_number",
    "ece_flag_number", "cwr_flag_number", "ack_count", "syn_count",
    "fin_count", "urg_count", "rst_count", "HTTP", "HTTPS", "DNS", "Telnet",
    "SMTP", "SSH", "IRC", "TCP", "UDP", "DHCP", "ARP", "ICMP", "IPv", "LLC",
    "Tot sum", "Min", "Max", "AVG", "Std", "Tot size", "IAT", "Number",
    "Magnitue", "Radius", "Covariance", "Variance", "Weight"
]

TIMING_FEATURES = {
    "flow_duration", "Duration", "Rate", "Srate", "Drate", "IAT"
}

STAT_FEATURES = {
    "ack_count", "syn_count", "fin_count", "urg_count", "rst_count",
    "Tot sum", "Min", "Max", "AVG", "Std", "Tot size", "Number",
    "Magnitue", "Radius", "Covariance", "Variance", "Weight"
}


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


class CNN1D(nn.Module):
    def __init__(self, n_features: int = 46):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveMaxPool1d(1),
        )
        self.fc = nn.Sequential(
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(1)
        x = self.conv(x).squeeze(-1)
        return self.fc(x)


class LSTMBaseline(nn.Module):
    def __init__(self, input_size: int = 46, hidden_size: int = 64, num_layers: int = 1):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 2)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(1)
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        return self.fc(out)


def get_model(model_type: str):
    if model_type == "cnn":
        return CNN1D(n_features=46)
    elif model_type == "lstm":
        return LSTMBaseline(input_size=46)
    else:
        raise ValueError("MODEL_TYPE must be 'cnn' or 'lstm'")


def build_mask(feature_set, device: torch.device) -> torch.Tensor:
    mask_list = [1.0 if f in feature_set else 0.0 for f in FEATURE_NAMES]
    return torch.tensor(mask_list, dtype=torch.float32, device=device).unsqueeze(0)


def train_model(model, train_loader, device):
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

    print("train time (s):", round(time.time() - start_time, 2))
    return criterion


def evaluate_clean(model, loader, device):
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


def fgsm_attack(model, criterion, x, y, epsilon, mask):
    was_training = model.training
    model.train()

    x_adv = x.clone().detach().requires_grad_(True)
    logits = model(x_adv)
    loss = criterion(logits, y)

    model.zero_grad()
    loss.backward()

    grad_sign = x_adv.grad.sign()
    grad_sign = grad_sign * mask

    x_adv = x_adv + epsilon * grad_sign
    x_adv = torch.clamp(x_adv, 0.0, 1.0)

    if not was_training:
        model.eval()

    return x_adv.detach()


def evaluate_group_attack(model, loader, device, criterion, epsilon, mask):
    model.eval()
    y_true_list = []
    y_pred_list = []

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        x_adv = fgsm_attack(model, criterion, x, y, epsilon, mask=mask)

        with torch.no_grad():
            logits = model(x_adv)
            preds = torch.argmax(logits, dim=1)

        y_true_list.append(y.cpu().numpy())
        y_pred_list.append(preds.cpu().numpy())

    y_true = np.concatenate(y_true_list)
    y_pred = np.concatenate(y_pred_list)
    return y_true, y_pred


def print_metrics(title, y_true, y_pred):
    print(f"\n{title}")
    print("Confusion matrix:")
    print(confusion_matrix(y_true, y_pred))
    print("\nReport:")
    print(classification_report(y_true, y_pred, digits=4))

    attack_recall = recall_score(y_true, y_pred, pos_label=1)
    macro_f1 = f1_score(y_true, y_pred, average="macro")
    weighted_f1 = f1_score(y_true, y_pred, average="weighted")

    print(f"Attack recall: {attack_recall:.4f}")
    print(f"Macro F1: {macro_f1:.4f}")
    print(f"Weighted F1: {weighted_f1:.4f}")


def main():
    set_seed(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)
    if device.type == "cuda":
        print("gpu:", torch.cuda.get_device_name(0))
    print("model type:", MODEL_TYPE)
    print("epsilons:", EPSILONS)

    timing_mask = build_mask(TIMING_FEATURES, device)
    stat_mask = build_mask(STAT_FEATURES, device)
    timing_stat_mask = build_mask(TIMING_FEATURES.union(STAT_FEATURES), device)

    print("Timing features:", sorted(list(TIMING_FEATURES)))
    print("Statistical features:", sorted(list(STAT_FEATURES)))
    print("Timing count:", int(timing_mask.sum().item()))
    print("Stat count:", int(stat_mask.sum().item()))
    print("Timing+Stat count:", int(timing_stat_mask.sum().item()))

    train_ds = NpyDataset(f"{DATA_DIR}/X_train.npy", f"{DATA_DIR}/y_train.npy")
    test_ds = NpyDataset(f"{DATA_DIR}/X_test.npy", f"{DATA_DIR}/y_test.npy")

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    model = get_model(MODEL_TYPE).to(device)
    criterion = train_model(model, train_loader, device)

    y_true_clean, y_pred_clean = evaluate_clean(model, test_loader, device)
    print_metrics("CLEAN RESULTS", y_true_clean, y_pred_clean)

    for eps in EPSILONS:
        print(f"\n===== EPSILON = {eps} =====")

        y_true_t, y_pred_t = evaluate_group_attack(
            model, test_loader, device, criterion, eps, timing_mask
        )
        print_metrics(f"TIMING-ONLY FGSM (epsilon={eps})", y_true_t, y_pred_t)

        y_true_s, y_pred_s = evaluate_group_attack(
            model, test_loader, device, criterion, eps, stat_mask
        )
        print_metrics(f"STATISTICAL-ONLY FGSM (epsilon={eps})", y_true_s, y_pred_s)

        y_true_ts, y_pred_ts = evaluate_group_attack(
            model, test_loader, device, criterion, eps, timing_stat_mask
        )
        print_metrics(f"TIMING+STATISTICAL FGSM (epsilon={eps})", y_true_ts, y_pred_ts)


if __name__ == "__main__":
    main()
