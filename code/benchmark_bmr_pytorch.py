from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from torch.utils.data import DataLoader, TensorDataset

FEATURES = ["wet_Mass_kg", "temperature", "pc1", "pc2", "pc3", "pc4", "pc5"]
TARGET = "BMR"
MODEL_NAMES = ["power_law_3_4", "pytorch_nn_residual"]


def find_root(marker: str = ".gitignore") -> Path:
    for start in [Path.cwd(), Path(__file__).resolve().parent]:
        current = start.resolve()
        for candidate in [current, *current.parents]:
            if (candidate / marker).exists():
                return candidate
    raise FileNotFoundError(f"Cannot find project root by marker: {marker}")


def load_split_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = ["taxon_name", *FEATURES, TARGET]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"{path.name} missing required columns: {', '.join(missing)}")

    out = df[required].copy()
    out["taxon_name"] = out["taxon_name"].astype("string").str.strip()
    for col in FEATURES + [TARGET]:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    out = out.dropna(subset=required).copy()
    out = out[(out["wet_Mass_kg"] > 0) & (out[TARGET] > 0)].copy()
    out = out[out["taxon_name"] != ""].copy()
    return out.reset_index(drop=True)


def fit_alpha_three_quarter(mass_kg: np.ndarray, bmr_w: np.ndarray) -> float:
    log_m = np.log(mass_kg)
    log_y = np.log(bmr_w)
    return float(np.mean(log_y - 0.75 * log_m))


def residual_feature_frame(base_df: pd.DataFrame, alpha: float) -> tuple[pd.DataFrame, np.ndarray]:
    mass = base_df["wet_Mass_kg"].to_numpy()
    base_log_pred = alpha + 0.75 * np.log(mass)
    X_res = pd.DataFrame(
        {
            "log_mass": np.log(mass),
            "temperature": base_df["temperature"].to_numpy(),
            "pc1": base_df["pc1"].to_numpy(),
            "pc2": base_df["pc2"].to_numpy(),
            "pc3": base_df["pc3"].to_numpy(),
            "pc4": base_df["pc4"].to_numpy(),
            "pc5": base_df["pc5"].to_numpy(),
            "base_log_pred": base_log_pred,
        },
        index=base_df.index,
    )
    return X_res, base_log_pred


def standardize_train_test(
    X_train: pd.DataFrame, X_test: pd.DataFrame
) -> tuple[np.ndarray, np.ndarray]:
    mu = X_train.mean(axis=0).to_numpy(dtype=np.float32)
    sigma = X_train.std(axis=0, ddof=0).to_numpy(dtype=np.float32)
    sigma = np.where(sigma < 1e-8, 1.0, sigma)
    Xtr = (X_train.to_numpy(dtype=np.float32) - mu) / sigma
    Xte = (X_test.to_numpy(dtype=np.float32) - mu) / sigma
    return Xtr, Xte


class ResidualMLP(nn.Module):
    def __init__(self, in_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def train_nn_residual(
    X_train: np.ndarray,
    y_train: np.ndarray,
    random_state: int,
    epochs: int = 250,
    batch_size: int = 64,
    lr: float = 1e-3,
) -> tuple[ResidualMLP, list[float]]:
    torch.manual_seed(random_state)
    np.random.seed(random_state)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(random_state)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ds = TensorDataset(
        torch.from_numpy(X_train.astype(np.float32)),
        torch.from_numpy(y_train.astype(np.float32)),
    )
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=False)

    model = ResidualMLP(in_dim=X_train.shape[1]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    loss_fn = nn.MSELoss()

    history: list[float] = []
    model.train()
    for _ in range(epochs):
        total_loss = 0.0
        total_count = 0
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            optimizer.step()
            batch_n = int(xb.shape[0])
            total_loss += float(loss.item()) * batch_n
            total_count += batch_n
        history.append(total_loss / max(total_count, 1))
    return model, history


def predict_nn(model: ResidualMLP, X: np.ndarray) -> np.ndarray:
    device = next(model.parameters()).device
    model.eval()
    with torch.no_grad():
        pred = model(torch.from_numpy(X.astype(np.float32)).to(device)).cpu().numpy()
    return pred


def evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }


def save_pred_and_residual_plots(out_dir: Path, pred_df: pd.DataFrame) -> None:
    sns.set_theme(style="whitegrid")

    for model in MODEL_NAMES:
        plt.figure(figsize=(8, 7))
        plt.scatter(
            pred_df["y_true"],
            pred_df[model],
            s=14,
            alpha=0.55,
            color="#1f77b4",
            label=f"{model} prediction",
        )
        plt.scatter(
            pred_df["y_true"],
            pred_df["y_true"],
            s=12,
            alpha=0.35,
            color="#ff7f0e",
            label="observed (y=x)",
        )
        min_v = float(min(pred_df["y_true"].min(), pred_df[model].min()))
        max_v = float(max(pred_df["y_true"].max(), pred_df[model].max()))
        plt.plot([min_v, max_v], [min_v, max_v], "k--", linewidth=1)
        plt.xscale("log")
        plt.yscale("log")
        plt.xlabel("Observed BMR (W)")
        plt.ylabel("Predicted BMR (W)")
        plt.title(f"Observed vs Predicted ({model})")
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_dir / f"nn_observed_vs_predicted_scatter_{model}.png", dpi=160)
        plt.close()

    plt.figure(figsize=(8, 7))
    for model in MODEL_NAMES:
        residual = pred_df["y_true"] - pred_df[model]
        plt.scatter(pred_df[model], residual, s=14, alpha=0.45, label=model)
    plt.axhline(0.0, color="k", linestyle="--", linewidth=1)
    plt.xscale("log")
    plt.xlabel("Predicted BMR (W)")
    plt.ylabel("Residual (Observed - Predicted)")
    plt.title("Residual Plot (PyTorch)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "nn_residual_plot.png", dpi=160)
    plt.close()


def save_training_curve(out_dir: Path, loss_history: list[float]) -> None:
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(9, 5))
    plt.plot(np.arange(1, len(loss_history) + 1), loss_history, color="#2563eb")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.title("PyTorch Training Loss Curve")
    plt.tight_layout()
    plt.savefig(out_dir / "nn_training_loss_curve.png", dpi=160)
    plt.close()


def main() -> None:
    root = find_root()
    parser = argparse.ArgumentParser(
        description=(
            "Train a PyTorch residual model for BMR prediction, using 3/4 power-law "
            "as baseline and writing outputs with nn_* names to avoid overwrite."
        )
    )
    parser.add_argument(
        "--train",
        type=Path,
        default=Path("data/train/train.csv"),
        help="Train CSV path.",
    )
    parser.add_argument(
        "--test",
        type=Path,
        default=Path("data/test/test.csv"),
        help="Test CSV path.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/benchmark"),
        help="Output directory.",
    )
    parser.add_argument("--random-state", type=int, default=42, help="Random seed.")
    parser.add_argument("--epochs", type=int, default=250, help="Training epochs.")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate.")
    args = parser.parse_args()

    train_path = args.train if args.train.is_absolute() else root / args.train
    test_path = args.test if args.test.is_absolute() else root / args.test
    out_dir = args.output_dir if args.output_dir.is_absolute() else root / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    train_df = load_split_data(train_path)
    test_df = load_split_data(test_path)

    y_train = train_df[TARGET].to_numpy()
    y_test = test_df[TARGET].to_numpy()
    alpha = fit_alpha_three_quarter(train_df["wet_Mass_kg"].to_numpy(), y_train)

    X_train_res, train_log_base = residual_feature_frame(train_df, alpha)
    X_test_res, test_log_base = residual_feature_frame(test_df, alpha)
    residual_train = np.log(y_train) - train_log_base

    X_train_std, X_test_std = standardize_train_test(X_train_res, X_test_res)

    nn_model, loss_history = train_nn_residual(
        X_train=X_train_std,
        y_train=residual_train,
        random_state=args.random_state,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
    )
    residual_pred_test = predict_nn(nn_model, X_test_std)

    yhat_base = np.exp(test_log_base)
    yhat_nn = np.exp(test_log_base + residual_pred_test)

    preds = {
        "power_law_3_4": yhat_base,
        "pytorch_nn_residual": yhat_nn,
    }

    metrics_rows = []
    for model in MODEL_NAMES:
        metrics_rows.append({"model": model, **evaluate(y_test, preds[model])})
    metrics_df = pd.DataFrame(metrics_rows).sort_values("rmse")
    metrics_df.to_csv(out_dir / "nn_benchmark_metrics.csv", index=False, encoding="utf-8")

    pred_df = test_df[["taxon_name", *FEATURES]].copy()
    pred_df["y_true"] = y_test
    for model in MODEL_NAMES:
        pred_df[model] = preds[model]
    pred_df.to_csv(out_dir / "nn_benchmark_predictions_test.csv", index=False, encoding="utf-8")

    save_pred_and_residual_plots(out_dir=out_dir, pred_df=pred_df)
    save_training_curve(out_dir=out_dir, loss_history=loss_history)

    print(f"Train rows used: {len(train_df)}")
    print(f"Test rows used: {len(test_df)}")
    print(f"Saved outputs in: {out_dir}")
    print("\nPyTorch benchmark results:")
    print(metrics_df.to_string(index=False))


if __name__ == "__main__":
    main()
