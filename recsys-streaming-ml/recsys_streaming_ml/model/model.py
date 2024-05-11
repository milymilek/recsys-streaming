import torch
from torch import nn


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, dropout=0.1, batch_norm=True):
        super(MLP, self).__init__()

        _layers = []
        dims = [input_dim] + hidden_dim
        for in_dim, out_dim in zip(dims, dims[1:]):
            _layers.append(nn.Linear(in_dim, out_dim))
            if batch_norm:
                _layers.append(nn.BatchNorm1d(out_dim))
            _layers.append(nn.ReLU())
            _layers.append(nn.Dropout(p=dropout))
        _layers.append(nn.Linear(dims[-1], 1))

        self.layers = nn.Sequential(*_layers)

    def forward(self, emb_x):
        return self.layers(emb_x)


class EmbeddingNet(nn.Module):
    def __init__(self, emb_dim, feature_sizes):
        super(EmbeddingNet, self).__init__()

        self.feature_sizes = feature_sizes

        _embeddings = {
            str(i): nn.Embedding(
                size,
                emb_dim,
            ) for i, size in enumerate(feature_sizes)
        }
        self.embeddings = nn.ModuleDict(_embeddings)

    def forward(self, x):
        x_emb = []
        for i, embed_matrix in self.embeddings.items():
            x_feat = x[:, int(i)].to(torch.long)
            emb = embed_matrix(x_feat)
            x_emb.append(emb)
        x_emb = torch.cat(x_emb, axis=1)
        return x_emb
    

class FM(nn.Module):
    def forward(self, inputs):
        fm_input = inputs

        square_of_sum = torch.pow(torch.sum(fm_input, dim=1, keepdim=True), 2)
        sum_of_square = torch.sum(fm_input * fm_input, dim=1, keepdim=True)
        cross_term = 0.5 * (square_of_sum - sum_of_square)

        return cross_term
    

class DeepFM(nn.Module):
    def __init__(self, emb_dim, hidden_dim, feature_sizes):
        super(DeepFM, self).__init__()

        self.V = EmbeddingNet(emb_dim=emb_dim, feature_sizes=feature_sizes)
        self.fm = FM()
        self.dnn = MLP(len(feature_sizes) * emb_dim, hidden_dim)

    def forward(self, x):
        x = self.V(x)
        x = self.fm(x) + self.dnn(x)
        return x