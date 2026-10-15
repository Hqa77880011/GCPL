import torch
from torch import nn
from torch.nn import functional as F


class Encoder(nn.Module):
    def __init__(self, dimension=64):
        super().__init__()
        self.net = nn.Sequential(nn.Conv2d(1, 24, 3, padding=1), nn.GroupNorm(4, 24), nn.ReLU(), nn.MaxPool2d(2),
                                 nn.Conv2d(24, 48, 3, padding=1), nn.GroupNorm(8, 48), nn.ReLU(), nn.MaxPool2d(2),
                                 nn.Conv2d(48, 64, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(64, dimension))

    def forward(self, x):
        return F.normalize(self.net(x), dim=-1)


def neighbors(prototypes, k):
    similarity = F.normalize(prototypes, dim=-1) @ F.normalize(prototypes, dim=-1).T
    similarity = similarity - torch.eye(len(prototypes), device=prototypes.device) * 1e6
    support = torch.zeros_like(similarity, dtype=torch.bool)
    if len(prototypes) > 1:
        support.scatter_(1, similarity.topk(min(k, len(prototypes) - 1), dim=1).indices, True)
    return support


def relation_graph(prototypes, dispersion, support, temperature, eta, reliability=True):
    normalized = F.normalize(prototypes, dim=-1)
    rho = torch.exp(-dispersion.mean(1) / eta) if reliability else torch.ones(len(prototypes), device=prototypes.device)
    similarity = (normalized @ normalized.T) * rho[:, None] * rho[None] / temperature
    scores = similarity.masked_fill(~support, -1e9).softmax(1) * support
    return scores / scores.sum(1, keepdim=True).clamp_min(1e-8)


def rank_loss(prototypes, previous_adjacency, margin):
    old = len(previous_adjacency)
    z = F.normalize(prototypes[:old], dim=-1)
    similarity = z @ z.T
    total = prototypes.sum() * 0
    edges = 0
    for i in range(old):
        positive = torch.nonzero(previous_adjacency[i] > 0, as_tuple=True)[0]
        negative = torch.nonzero((previous_adjacency[i] == 0) & (torch.arange(old, device=z.device) != i), as_tuple=True)[0]
        if len(positive) and len(negative):
            total = total + F.relu(margin - similarity[i, positive, None] + similarity[i, negative][None]).sum()
            edges += len(positive)
    return total / max(1, edges)
