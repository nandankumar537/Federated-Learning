# import argparse
# import os
# import numpy as np
# import pandas as pd
# from typing import Tuple, Dict, List

# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torch.utils.data import TensorDataset, DataLoader

# from sklearn.model_selection import StratifiedKFold
# from sklearn.preprocessing import StandardScaler
# from sklearn.impute import SimpleImputer

# import flwr as fl

# # ----- Configuration -----
# FEATURE_COLUMNS = [
#     "age","sex","cp","trestbps","chol","fbs","restecg",
#     "thalach","exang","oldpeak","slope","ca","thal"
# ]
# TARGET_COLUMN = "target"

# DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# # ----- Model -----
# class MLP(nn.Module):
#     def __init__(self, in_dim: int = 13, hidden: int = 32, p: float = 0.2):
#         super().__init__()
#         self.net = nn.Sequential(
#             nn.Linear(in_dim, hidden),
#             nn.ReLU(),
#             nn.Dropout(p),
#             nn.Linear(hidden, hidden),
#             nn.ReLU(),
#             nn.Dropout(p),
#             nn.Linear(hidden, 1),
#             nn.Sigmoid(),
#         )

#     def forward(self, x):
#         return self.net(x)

# # ----- Data utilities -----
# def load_and_preprocess(csv_path: str) -> Tuple[np.ndarray, np.ndarray]:
#     df = pd.read_csv(csv_path)

#     # Keep only required columns if extra columns exist
#     keep_cols = FEATURE_COLUMNS + [TARGET_COLUMN]
#     df = df[keep_cols]

#     # Binarize target: 0 -> 0, {1,2,3,4} -> 1
#     df[TARGET_COLUMN] = (df[TARGET_COLUMN].astype(int) > 0).astype(int)

#     # Impute missing values (ca and thal commonly have '?', convert to NaN)
#     df = df.replace("?", np.nan)
#     for col in FEATURE_COLUMNS:
#         df[col] = pd.to_numeric(df[col], errors="coerce")

#     X = df[FEATURE_COLUMNS].values
#     y = df[TARGET_COLUMN].values

#     imputer = SimpleImputer(strategy="median")
#     X = imputer.fit_transform(X)

#     scaler = StandardScaler()
#     X = scaler.fit_transform(X)

#     return X, y

# def partition_data(
#     X: np.ndarray, y: np.ndarray, num_clients: int, client_id: int, seed: int = 42
# ) -> Tuple[np.ndarray, np.ndarray]:
#     """
#     Create a stratified partition: split indices into num_clients folds and
#     select the split for this client_id (0..num_clients-1).
#     """
#     skf = StratifiedKFold(n_splits=num_clients, shuffle=True, random_state=seed)
#     for i, (_, idx_client) in enumerate(skf.split(X, y)):
#         if i == client_id:
#             return X[idx_client], y[idx_client]
#     raise ValueError("Invalid client_id or partitioning failure")

# def train_val_split(
#     Xc: np.ndarray, yc: np.ndarray, val_ratio: float = 0.2, seed: int = 123
# ) -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
#     # Stratified shuffle for small client shard
#     rs = np.random.RandomState(seed)
#     idx = np.arange(len(yc))
#     # stratified split by simple per-class shuffle
#     cls0 = idx[yc == 0]
#     cls1 = idx[yc == 1]
#     rs.shuffle(cls0); rs.shuffle(cls1)
#     n0_val = max(1, int(len(cls0) * val_ratio))
#     n1_val = max(1, int(len(cls1) * val_ratio))
#     val_idx = np.concatenate([cls0[:n0_val], cls1[:n1_val]])
#     train_idx = np.concatenate([cls0[n0_val:], cls1[n1_val:]])
#     rs.shuffle(train_idx); rs.shuffle(val_idx)
#     return (Xc[train_idx], yc[train_idx]), (Xc[val_idx], yc[val_idx])

# def to_loader(
#     X: np.ndarray, y: np.ndarray, batch_size: int = 32, shuffle: bool = True
# ) -> DataLoader:
#     xt = torch.tensor(X, dtype=torch.float32)
#     yt = torch.tensor(y.reshape(-1, 1), dtype=torch.float32)
#     ds = TensorDataset(xt, yt)
#     return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)

# # ----- Training / Eval -----
# def train_one_epoch(
#     model: nn.Module, loader: DataLoader, optimizer: optim.Optimizer, criterion: nn.Module
# ) -> float:
#     model.train()
#     running_loss = 0.0
#     for xb, yb in loader:
#         xb, yb = xb.to(DEVICE), yb.to(DEVICE)
#         optimizer.zero_grad()
#         preds = model(xb)
#         loss = criterion(preds, yb)
#         loss.backward()
#         optimizer.step()
#         running_loss += loss.item() * xb.size(0)
#     return running_loss / len(loader.dataset)

# @torch.no_grad()
# def evaluate(model: nn.Module, loader: DataLoader, criterion: nn.Module) -> Tuple[float, float]:
#     model.eval()
#     total_loss, correct = 0.0, 0
#     total = 0
#     for xb, yb in loader:
#         xb, yb = xb.to(DEVICE), yb.to(DEVICE)
#         preds = model(xb)
#         loss = criterion(preds, yb)
#         total_loss += loss.item() * xb.size(0)
#         pred_cls = (preds >= 0.5).float()
#         correct += (pred_cls == yb).sum().item()
#         total += xb.size(0)
#     return total_loss / total, correct / total

# # ----- Flower client -----
# class HeartClient(fl.client.NumPyClient):
#     def __init__(
#         self,
#         train_loader: DataLoader,
#         val_loader: DataLoader,
#         model: nn.Module,
#         lr: float = 1e-3,
#         epochs: int = 3,
#     ):
#         self.train_loader = train_loader
#         self.val_loader = val_loader
#         self.model = model.to(DEVICE)
#         self.criterion = nn.BCELoss()
#         self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
#         self.epochs = epochs

#     def get_parameters(self, config):
#         return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

#     def set_parameters(self, parameters: List[np.ndarray]):
#         state_dict = self.model.state_dict()
#         for (k, _), v in zip(state_dict.items(), parameters):
#             state_dict[k] = torch.tensor(v, dtype=state_dict[k].dtype)
#         self.model.load_state_dict(state_dict, strict=True)

#     def fit(self, parameters, config):
#         self.set_parameters(parameters)

#         for _ in range(self.epochs):
#             train_one_epoch(self.model, self.train_loader, self.optimizer, self.criterion)

#         # Return updated params and number of examples
#         return self.get_parameters(config={}), len(self.train_loader.dataset), {}

#     def evaluate(self, parameters, config):
#         self.set_parameters(parameters)
#         loss, acc = evaluate(self.model, self.val_loader, self.criterion)
#         return float(loss), len(self.val_loader.dataset), {"accuracy": float(acc)}

# def main():
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--data", type=str, default="cleveland.csv", help="Path to cleveland dataset CSV")
#     parser.add_argument("--server", type=str, default="127.0.0.1:8080", help="Server address host:port")
#     parser.add_argument("--num_clients", type=int, default=2, help="Total number of clients in the federation")
#     parser.add_argument("--client_id", type=int, required=True, help="Client id in [0..num_clients-1]")
#     parser.add_argument("--batch_size", type=int, default=32)
#     parser.add_argument("--epochs", type=int, default=3)
#     parser.add_argument("--lr", type=float, default=1e-3)
#     args = parser.parse_args()

#     # Load and preprocess full dataset
#     X, y = load_and_preprocess(args.data)

#     # Partition this client's shard
#     Xc, yc = partition_data(X, y, num_clients=args.num_clients, client_id=args.client_id, seed=42)

#     # Local train/val split
#     (Xtr, ytr), (Xva, yva) = train_val_split(Xc, yc, val_ratio=0.2, seed=123)

#     train_loader = to_loader(Xtr, ytr, batch_size=args.batch_size, shuffle=True)
#     val_loader = to_loader(Xva, yva, batch_size=args.batch_size, shuffle=False)

#     # Model
#     model = MLP(in_dim=len(FEATURE_COLUMNS))
#     client = HeartClient(train_loader, val_loader, model, lr=args.lr, epochs=args.epochs)

#     # Start Flower client
#     # fl.client.start_numpy_client(server_address=args.server, client=client)
#     fl.client.start_client(server_address=args.server, client=client.to_client())

# if __name__ == "__main__":
#     main()


# client.py
import argparse
import os
import numpy as np
import pandas as pd
from typing import Tuple, Dict, List

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

import flwr as fl

# ----- Configuration -----
FEATURE_COLUMNS = [
    "age","sex","cp","trestbps","chol","fbs","restecg",
    "thalach","exang","oldpeak","slope","ca","thal"
]
TARGET_COLUMN = "target"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ----- Model -----
class MLP(nn.Module):
    def __init__(self, in_dim: int = 13, hidden: int = 32, p: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Dropout(p),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Dropout(p),
            nn.Linear(hidden, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.net(x)

# ----- Data utilities -----
def load_and_preprocess(csv_path: str) -> Tuple[np.ndarray, np.ndarray]:
    try:
        df = pd.read_csv(csv_path)
    except UnicodeDecodeError:
        try:
            df = pd.read_csv(csv_path, encoding="latin-1")
        except UnicodeDecodeError:
            df = pd.read_csv(csv_path, encoding="cp1252", on_bad_lines="skip", engine="python")

    if isinstance(df.columns, pd.RangeIndex) and df.shape[1] == 14:
        df.columns = FEATURE_COLUMNS + [TARGET_COLUMN]

    keep_cols = FEATURE_COLUMNS + [TARGET_COLUMN]
    missing = [c for c in keep_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing expected columns: {missing}. Got columns: {list(df.columns)}")

    df = df.replace("?", np.nan)
    for col in FEATURE_COLUMNS + [TARGET_COLUMN]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df[TARGET_COLUMN] = (df[TARGET_COLUMN].fillna(0).astype(int) > 0).astype(int)

    X = df[FEATURE_COLUMNS].values
    y = df[TARGET_COLUMN].values

    imputer = SimpleImputer(strategy="median")
    X = imputer.fit_transform(X)

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    return X, y

def partition_data(
    X: np.ndarray, y: np.ndarray, num_clients: int, client_id: int, seed: int = 42
) -> Tuple[np.ndarray, np.ndarray]:
    skf = StratifiedKFold(n_splits=num_clients, shuffle=True, random_state=seed)
    for i, (_, idx_client) in enumerate(skf.split(X, y)):
        if i == client_id:
            return X[idx_client], y[idx_client]
    raise ValueError("Invalid client_id or partitioning failure")

def train_val_split(
    Xc: np.ndarray, yc: np.ndarray, val_ratio: float = 0.2, seed: int = 123
) -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
    rs = np.random.RandomState(seed)
    idx = np.arange(len(yc))
    cls0 = idx[yc == 0]
    cls1 = idx[yc == 1]
    rs.shuffle(cls0); rs.shuffle(cls1)
    n0_val = max(1, int(len(cls0) * val_ratio))
    n1_val = max(1, int(len(cls1) * val_ratio))
    val_idx = np.concatenate([cls0[:n0_val], cls1[:n1_val]])
    train_idx = np.concatenate([cls0[n0_val:], cls1[n1_val:]])
    rs.shuffle(train_idx); rs.shuffle(val_idx)
    return (Xc[train_idx], yc[train_idx]), (Xc[val_idx], yc[val_idx])

def to_loader(
    X: np.ndarray, y: np.ndarray, batch_size: int = 32, shuffle: bool = True
) -> DataLoader:
    xt = torch.tensor(X, dtype=torch.float32)
    yt = torch.tensor(y.reshape(-1, 1), dtype=torch.float32)
    ds = TensorDataset(xt, yt)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)

# ----- Training / Eval -----
def train_one_epoch(
    model: nn.Module, loader: DataLoader, optimizer: optim.Optimizer, criterion: nn.Module, noise_std: float = 0.0
) -> float:
    model.train()
    running_loss = 0.0
    for xb, yb in loader:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        if noise_std > 0.0:
            xb = xb + torch.randn_like(xb) * noise_std
        optimizer.zero_grad()
        preds = model(xb)
        loss = criterion(preds, yb)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * xb.size(0)
    return running_loss / len(loader.dataset)

@torch.no_grad()
def evaluate_metrics(model: nn.Module, loader: DataLoader, criterion: nn.Module) -> Tuple[float, Dict[str, float]]:
    model.eval()
    total_loss, ys, ps = 0.0, [], []
    for xb, yb in loader:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        preds = model(xb)
        loss = criterion(preds, yb)
        total_loss += loss.item() * xb.size(0)
        ys.append(yb.cpu().numpy().ravel())
        ps.append(preds.cpu().numpy().ravel())
    total = sum(len(arr) for arr in ys)
    loss = total_loss / max(total, 1)
    y_true = np.concatenate(ys) if ys else np.array([])
    p = np.concatenate(ps) if ps else np.array([])
    y_pred = (p >= 0.5).astype(int) if p.size > 0 else np.array([])
    acc = float(accuracy_score(y_true, y_pred)) if y_true.size else 0.0
    prec = float(precision_score(y_true, y_pred, zero_division=0)) if y_true.size else 0.0
    rec = float(recall_score(y_true, y_pred, zero_division=0)) if y_true.size else 0.0
    f1 = float(f1_score(y_true, y_pred, zero_division=0)) if y_true.size else 0.0
    return loss, {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1}

def mean_cv_accuracy_local(X_local: np.ndarray, y_local: np.ndarray, epochs: int, batch_size: int, noise_std: float, folds: int = 5) -> float:
    # 5-fold on the client's shard
    skf = StratifiedKFold(n_splits=min(folds, len(np.unique(y_local)) if len(y_local) >= folds else 2), shuffle=True, random_state=42)
    accs = []
    for tr_idx, va_idx in skf.split(X_local, y_local):
        Xtr, Xva = X_local[tr_idx], X_local[va_idx]
        ytr, yva = y_local[tr_idx], y_local[va_idx]
        model = MLP(in_dim=X_local.shape[1]).to(DEVICE)
        opt = optim.Adam(model.parameters(), lr=1e-3)
        crit = nn.BCELoss()
        loader = DataLoader(
            TensorDataset(torch.tensor(Xtr, dtype=torch.float32),
                          torch.tensor(ytr.reshape(-1,1), dtype=torch.float32)),
            batch_size=batch_size, shuffle=True
        )
        # train
        for _ in range(epochs):
            model.train()
            for xb, yb in loader:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                if noise_std > 0.0:
                    xb = xb + torch.randn_like(xb) * noise_std
                opt.zero_grad()
                preds = model(xb)
                loss = crit(preds, yb)
                loss.backward()
                opt.step()
        # eval
        model.eval()
        with torch.no_grad():
            p = torch.sigmoid(model(torch.tensor(Xva, dtype=torch.float32).to(DEVICE))).cpu().numpy().ravel()
        yhat = (p >= 0.5).astype(int)
        accs.append(accuracy_score(yva, yhat))
    return float(np.mean(accs)) if accs else 0.0

# ----- Flower client -----
class HeartClient(fl.client.NumPyClient):
    def __init__(
        self,
        client_id: int,
        train_loader: DataLoader,
        val_loader: DataLoader,
        X_local: np.ndarray,
        y_local: np.ndarray,
        model: nn.Module,
        lr: float = 1e-3,
        epochs: int = 3,
        noise_std: float = 0.0,
        batch_size: int = 32,
        log_dir: str = ".",
    ):
        self.client_id = client_id
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.X_local = X_local
        self.y_local = y_local
        self.model = model.to(DEVICE)
        self.criterion = nn.BCELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.epochs = epochs
        self.noise_std = noise_std
        self.batch_size = batch_size
        os.makedirs(log_dir, exist_ok=True)
        self.log_path = os.path.join(log_dir, f"client{client_id}_log.txt")
        # write header
        if not os.path.exists(self.log_path):
            with open(self.log_path, "w", encoding="utf-8") as f:
                f.write("Round\tEpochs\tBatchSize\tNoise\tAccuracy\tPrecision\tRecall\tF1\tMeanCVAccuracy\n")

    def get_parameters(self, config):
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def set_parameters(self, parameters: List[np.ndarray]):
        state_dict = self.model.state_dict()
        for (k, _), v in zip(state_dict.items(), parameters):
            state_dict[k] = torch.tensor(v, dtype=state_dict[k].dtype)
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        server_round = int(config.get("server_round", -1))

        for _ in range(self.epochs):
            train_one_epoch(self.model, self.train_loader, self.optimizer, self.criterion, noise_std=self.noise_std)

        return self.get_parameters(config={}), len(self.train_loader.dataset), {"server_round": server_round}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        server_round = int(config.get("server_round", -1))

        loss, base_metrics = evaluate_metrics(self.model, self.val_loader, self.criterion)

        # Compute Mean CV Accuracy on local shard for this round
        mean_cv_acc = mean_cv_accuracy_local(
            self.X_local, self.y_local,
            epochs=self.epochs,
            batch_size=self.batch_size,
            noise_std=self.noise_std,
            folds=5
        )

        metrics = {
            "accuracy": float(base_metrics["accuracy"]),
            "precision": float(base_metrics["precision"]),
            "recall": float(base_metrics["recall"]),
            "f1": float(base_metrics["f1"]),
            "mean_cv_accuracy": float(mean_cv_acc),
            "server_round": server_round,
            "client_id": self.client_id,
        }

        # Append to client log
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(
                f"{server_round}\t{self.epochs}\t{self.batch_size}\t{self.noise_std:.3f}\t"
                f"{metrics['accuracy']:.4f}\t{metrics['precision']:.4f}\t{metrics['recall']:.4f}\t"
                f"{metrics['f1']:.4f}\t{metrics['mean_cv_accuracy']:.4f}\n"
            )

        return float(loss), len(self.val_loader.dataset), metrics

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="cleveland.csv", help="Path to cleveland dataset CSV")
    parser.add_argument("--server", type=str, default="127.0.0.1:8080", help="Server address host:port")
    parser.add_argument("--num_clients", type=int, default=2, help="Total number of clients in the federation")
    parser.add_argument("--client_id", type=int, required=True, help="Client id in [0..num_clients-1]")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--noise", type=float, default=0.0, help="Gaussian noise std added to inputs during training")
    parser.add_argument("--log_dir", type=str, default=".", help="Directory to write client logs")
    args = parser.parse_args()

    # Load and preprocess full dataset
    X, y = load_and_preprocess(args.data)

    # Partition this client's shard
    Xc, yc = partition_data(X, y, num_clients=args.num_clients, client_id=args.client_id, seed=42)

    # Local train/val split
    (Xtr, ytr), (Xva, yva) = train_val_split(Xc, yc, val_ratio=0.2, seed=123)

    train_loader = to_loader(Xtr, ytr, batch_size=args.batch_size, shuffle=True)
    val_loader = to_loader(Xva, yva, batch_size=256, shuffle=False)

    # Model
    model = MLP(in_dim=len(FEATURE_COLUMNS))
    client = HeartClient(
        client_id=args.client_id,
        train_loader=train_loader,
        val_loader=val_loader,
        X_local=Xc,
        y_local=yc,
        model=model,
        lr=args.lr,
        epochs=args.epochs,
        noise_std=args.noise,
        batch_size=args.batch_size,
        log_dir=args.log_dir,
    )

    # Start Flower client (new API)
    fl.client.start_client(server_address=args.server, client=client.to_client())

if __name__ == "__main__":
    main()
