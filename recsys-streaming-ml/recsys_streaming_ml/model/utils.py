import pathlib
from datetime import datetime
import io
from pymongo.database import Database

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

from recsys_streaming_ml.db import read_latest_model_version_document
from recsys_streaming_ml.config import RUNS_DIR, DATASET_FILE


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


def build_input_tensor(inputs: np.ndarray) -> torch.Tensor:
    return torch.tensor(inputs, dtype=torch.long)


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


def _get_new_model_version(db: Database) -> int:
    latest_version = read_latest_model_version_document(db)

    if not latest_version:
        return 0
    
    return int(latest_version['version']) + 1


def _get_model_buffer(db: Database) -> io.BytesIO:
    latest_version = read_latest_model_version_document(db)

    if not latest_version:
        raise ValueError("Model not found in database!")

    return io.BytesIO(latest_version["binary"])


def dump_model(db: Database, model, save_path: pathlib.Path):
    model_scripted = torch.jit.script(model.to('cpu')) # Export to TorchScript

    buffer = io.BytesIO()
    torch.jit.save(model_scripted, buffer)
    buffer.seek(0)

    model_version = _get_new_model_version(db)
    db['model_versions'].insert_one({"model": model.__class__.__name__, "timestamp": datetime.now(), "binary": buffer.read(), "version": model_version})


def load_model_from_db(db: Database, device: str = 'cpu'):
    model_document = read_latest_model_version_document(db)
    model_version = _get_new_model_version(db)
    model_buffer = _get_model_buffer(db)

    print(f'Loading model: {model_document["model"]}:v{model_version}-{model_document["timestamp"]}')
    model = torch.jit.load(model_buffer, map_location=device)
    return model


def save_history(history: dict[str, list], save_path: pathlib.Path):
    save_path_file = save_path / "history.xlsx"
    pd.DataFrame(history).to_excel(save_path_file)


def load_dataset(dataset_path: pathlib.Path = DATASET_FILE):
    return pd.read_csv(DATASET_FILE / "dataset.csv", index_col=[0])
    # return {
    #     "train_data": pd.read_csv(DATASET_FILE / "train_data.csv"),
    #     "train_targets": pd.read_csv(DATASET_FILE / "train_targets.csv"),
    #     "valid_data": pd.read_csv(DATASET_FILE / "valid_data.csv"),
    #     "valid_targets": pd.read_csv(DATASET_FILE / "valid_targets.csv"),
    # }

