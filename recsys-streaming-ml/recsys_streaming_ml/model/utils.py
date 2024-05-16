import pathlib
from datetime import datetime
import io

import pandas as pd
import torch
import matplotlib.pyplot as plt

from recsys_streaming_ml.config import RUNS_DIR
from recsys_streaming_ml.db import client


def collate_fn(batch):
    data, targets = zip(*batch)
    data = torch.stack(data)
    targets = torch.stack(targets)
    return data, targets


def neg_sampling(X, y, k=1, item_cols_idx=[1,2]):
    positive_idx = (y == 1).nonzero(as_tuple=True)[0]

    X_positive = torch.index_select(X, dim=0, index=positive_idx)
    permutation = torch.randperm(X_positive.shape[0] * k)
    X_items_permuted = X_positive[:, item_cols_idx][permutation]
    X_neg_sampl = torch.cat([X_positive[:, [0]], X_items_permuted], axis=1)

    return torch.cat([X, X_neg_sampl], axis=0), torch.cat([y, torch.zeros(X_neg_sampl.shape[0], 1)], axis=0)


def cast_df_to_tensor(*args):
    return (torch.Tensor(a.values) for a in args)


def binarize_target(target: torch.Tensor, threshold: float): 
    return torch.where(target >= threshold, 1, 0)


def build_log_dir_path(model_name: str):
     return RUNS_DIR / f"{model_name}/{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"


def plot_and_save(history: dict[str, list], save_path: pathlib.Path):
    save_path_file = save_path / "plots.png"

    fig, axs = plt.subplots(2, 2, figsize=(10, 8))

    axs[0, 0].plot(history["train_loss"])
    axs[0, 0].set_title('Train Loss')

    axs[0, 1].plot(history["train_roc_auc"])
    axs[0, 1].set_title('Train ROC AUC')

    axs[1, 0].plot(history["test_loss"])
    axs[1, 0].set_title('Test Loss')

    axs[1, 1].plot(history["test_roc_auc"])
    axs[1, 1].set_title('Test ROC AUC')

    plt.tight_layout()
    plt.savefig(save_path_file)


def dump_model(model, save_path: pathlib.Path):
    model_scripted = torch.jit.script(model.to('cpu')) # Export to TorchScript

    buffer = io.BytesIO()
    torch.jit.save(model_scripted, buffer)
    buffer.seek(0)

    client['model_versions'].insert_one({"model": model.__class__.__name__, "timestamp": datetime.now(), "binary": buffer.read()})


def save_history(history: dict[str, list], save_path: pathlib.Path):
    save_path_file = save_path / "history.xlsx"
    pd.DataFrame(history).to_excel(save_path_file)


