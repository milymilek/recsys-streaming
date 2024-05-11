import pathlib

import torch


def load_model(path: pathlib.Path):
    model = torch.jit.load(path, map_location='cpu')
    return model


def build_input_tensor(user_id: int, item_id: int) -> torch.Tensor:
    return torch.tensor([user_id, item_id, 0], dtype=torch.long).view(1, -1)