import os
from tqdm import tqdm
from datetime import datetime
import pickle
import argparse

import numpy as np
from sklearn.metrics import roc_auc_score
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader

from recsys_streaming_ml.model.model import DeepFM
from recsys_streaming_ml.config import DATASET_FILE
from recsys_streaming_ml.model.utils import (
    collate_fn, neg_sampling, cast_df_to_tensor, binarize_target, plot_and_save,
    build_log_dir_path, dump_model, save_history, load_dataset
)
from recsys_streaming_ml.data.utils import load_feature_maps
from recsys_streaming_ml.db import mongo_db


BINARY_MAP_THRESHOLD = 4.5


def _save_train_artifacts(model, history):
    run_id = build_log_dir_path(model_name=model.__class__.__name__)
    run_id.mkdir(parents=True)
    save_history(history, save_path=run_id)
    plot_and_save(history, save_path=run_id)
    dump_model(db=mongo_db, model=model, save_path=run_id)
    print(f"> Training Artifacts saved under {run_id}")
    

def train_epoch(model, criterion, optimizer, train_loader, device):
    model.train()

    running_loss = 0.
    preds, ground_truths = [], []
    for i_batch, (batch, y_true) in enumerate(tqdm(train_loader)):
        batch, y_true = batch.to(device), y_true.to(device)

        y_pred = model(batch)

        loss = criterion(y_pred, y_true)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        preds.append(y_pred)
        ground_truths.append(y_true)
        running_loss += loss.item()

    pred = torch.cat(preds, dim=0).detach().sigmoid().cpu().numpy()
    ground_truth = torch.cat(ground_truths, dim=0).detach().cpu().numpy()
    train_loss = running_loss / len(train_loader)
    train_roc_auc = roc_auc_score(ground_truth, pred)

    return train_loss, train_roc_auc


@torch.no_grad()
def test(model, criterion, val_loader, device):
    model.eval()

    running_loss = 0.
    preds, ground_truths = [], []

    for i_batch, (batch, y_true) in enumerate(val_loader):
        batch, y_true = batch.to(device), y_true.to(device)

        y_pred = model(batch)
        loss = criterion(y_pred, y_true)

        preds.append(y_pred)
        ground_truths.append(y_true)
        running_loss += loss.item()

    pred = torch.cat(preds, dim=0).sigmoid().cpu().numpy()
    ground_truth = torch.cat(ground_truths, dim=0).cpu().numpy()

    test_loss = running_loss / len(val_loader)
    test_roc_auc = roc_auc_score(ground_truth, pred)

    return test_loss, test_roc_auc


def train(args):
    device = 'cuda' if (args.cuda and torch.cuda.is_available()) else 'cpu'
    n_epochs = args.epochs
    seed = args.seed
    batch_size = args.batch_size

    torch.manual_seed(seed)

    data = load_dataset()

    (X_train, y_train, X_valid, y_valid) = \
        data['train_data'], data['train_targets'], data['train_data'], data['train_targets']
    
    X_train, y_train, X_valid, y_valid = cast_df_to_tensor(X_train, y_train, X_valid, y_valid)
    y_train = binarize_target(y_train, threshold=BINARY_MAP_THRESHOLD)
    y_valid = binarize_target(y_valid, threshold=BINARY_MAP_THRESHOLD)
    print(X_train.shape, y_train.shape, X_valid.shape, y_valid.shape)
    print("> Data loaded and casted to tensors")

    X_train, y_train = neg_sampling(X_train, y_train, k=1)
    X_valid, y_valid = neg_sampling(X_valid, y_valid, k=1) 
    print("> Negative sampling done")

    feature_maps = load_feature_maps()
    feature_sizes = (len(feature_maps['user_id_map']), len(feature_maps['parent_id_map']), len(feature_maps['store_id_map']))

    print("> Feature maps loaded")

    model = DeepFM(emb_dim=8, hidden_dim=[32, 24, 10], feature_sizes=feature_sizes).to(device)

    print("> Model initialized")


    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size, collate_fn=collate_fn, drop_last=False)
    val_dataset = TensorDataset(X_valid, y_valid)
    val_loader = DataLoader(val_dataset, shuffle=True, batch_size=batch_size, collate_fn=collate_fn, drop_last=False)

    print("> Datasets prepared")

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.RMSprop(params=model.parameters(), lr=1e-4, momentum=0.9)


    history = {"train_loss": [], "train_roc_auc": [], "test_loss": [], "test_roc_auc": []}
    print(f"> Training model[{model.__class__.__name__}] on device[{device}] begins...")
    for epoch in tqdm(range(n_epochs)):
        train_loss, train_roc_auc = train_epoch(
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            train_loader=train_loader,
            device=device
        )
        test_loss, test_roc_auc = test(
            model=model,
            criterion=criterion,
            val_loader=val_loader,
            device=device
        )

        history["train_loss"].append(train_loss)
        history["train_roc_auc"].append(train_roc_auc)
        history["test_loss"].append(test_loss)
        history["test_roc_auc"].append(test_roc_auc)

        print(f"""Epoch <{epoch}>\ntrain_loss: {train_loss} - train_roc_auc: {train_roc_auc}
test_loss: {test_loss} - test_roc_auc: {test_roc_auc}\n""")
        
    _save_train_artifacts(model, history)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    
    p.add_argument("-e", "--epochs", default=10, type=int)
    p.add_argument("-bs", "--batch_size", default=16, type=int)
    p.add_argument("-c", "--cuda", default=False, type=bool, help="use cuda to train")
    p.add_argument("-s", "--seed", default=42, type=int)

    return p.parse_known_args()[0]


def train_placeholer(users_actions_df, batch_id):
    if not users_actions_df.isEmpty():
        print(f"Showing data for batch: {batch_id}")
        users_actions_df.show()


def run():
    print("SCRIPT: Train model - START")

    args = parse_args()
    train(args)

    print(f'')
    print("SCRIPT: Train model - END")