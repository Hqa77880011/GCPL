import argparse
import json
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.signal import stft
from torch.nn import functional as F

from .method import Encoder, neighbors, rank_loss, relation_graph


def prepare(args):
    manifest = Path(args.manifest)
    rows = json.loads(manifest.read_text(encoding="utf-8"))
    groups, packed = {}, {}
    sessions = json.loads(Path(args.sessions).read_text())
    flat = [c for session in sessions for c in session]
    if len(flat) != len(set(flat)) or len(sessions[0]) < 2:
        raise ValueError("Sessions must introduce unique classes with at least two base classes")
    for row in rows:
        group, split = str(row["group"]), row["split"]
        if group in groups and groups[group] != split:
            raise ValueError("Acquisition or bearing identity crosses partitions")
        groups[group] = split
        signal = np.load(manifest.resolve().parent / row["file"], allow_pickle=False).reshape(-1)
        session = next(i for i, classes in enumerate(sessions) if row["label"] in classes)
        for start in range(0, len(signal) - args.window + 1, args.stride):
            x = signal[start:start + args.window]
            x = (x - x.mean()) / max(x.std(), 1e-6)
            _, _, spectrum = stft(x, nperseg=256, noverlap=128, boundary=None)
            feature = np.log1p(np.abs(spectrum)).astype(np.float32)
            packed.setdefault((split, session), []).append((feature, flat.index(row["label"]), group))
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    for (split, session), rows in packed.items():
        x, y, group = zip(*rows)
        np.savez_compressed(out / f"{split}_{session}.npz", x=np.array(x)[:, None], y=y, group=group)
    (out / "sessions.json").write_text(json.dumps(sessions, indent=2))


def data(root, split, session):
    return np.load(Path(root) / f"{split}_{session}.npz", allow_pickle=False)


def embedded(net, x, args):
    with torch.no_grad():
        return torch.cat([net(torch.as_tensor(x[start:start + args.batch_size], device=args.device)) for start in range(0, len(x), args.batch_size)])


def sparse_graph(adjacency):
    edges = torch.nonzero(adjacency > 0).T
    return edges.cpu(), adjacency[edges[0], edges[1]].cpu()


def restore_graph(saved, device):
    count = len(saved["prototypes"])
    a = torch.zeros(count, count, device=device)
    edge = saved["edge_index"].to(device)
    a[edge[0], edge[1]] = saved["edge_weight"].to(device)
    return a


def train(args):
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    sessions = json.loads((Path(args.data) / "sessions.json").read_text())
    net = Encoder(args.dimension).to(args.device)
    previous_p, previous_s, previous_a, prior = None, None, None, None
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    for session, classes in enumerate(sessions):
        session_start = time.perf_counter()
        current = data(args.data, "train", session)
        selected = np.arange(len(current["y"]))
        if session and args.shots:
            selected = np.concatenate([rng.permutation(np.flatnonzero(current["y"] == c))[:args.shots] for c in np.unique(current["y"])])
        x, y = current["x"][selected], current["y"][selected]
        net.eval()
        z = embedded(net, x, args)
        labels = torch.as_tensor(y, device=args.device)
        new_classes = sorted(np.unique(y))
        if len(new_classes) != len(classes):
            raise ValueError("Every new class needs training examples")
        means = torch.stack([z[labels == c].mean(0) for c in new_classes])
        if session == 0:
            base_dev = data(args.data, "dev", 0)
            vz = embedded(net, base_dev["x"], args)
            variances = [vz[base_dev["y"] == c].var(0, unbiased=True) for c in new_classes if np.sum(base_dev["y"] == c) >= 2]
            if not variances:
                raise ValueError("Base validation requires at least two samples in a class for the dispersion prior")
            prior = torch.stack(variances).mean(0).detach()
        dispersion = torch.stack([z[labels == c].var(0, unbiased=True) if int((labels == c).sum()) > 1 else prior for c in new_classes])
        old = 0 if previous_p is None else len(previous_p)
        prototypes = torch.nn.Parameter(means if not old else torch.cat([previous_p, means]))
        all_s = dispersion if not old else torch.cat([previous_s, dispersion])
        support = neighbors(prototypes.detach(), args.neighbors)
        if old and not args.global_rebuild:
            fresh = support.clone()
            support[:old, :old] = previous_a > 0
            for i in range(old):
                additions = torch.nonzero(fresh[i, old:], as_tuple=True)[0] + old
                for j in additions:
                    support[i, j] = True
        optimizer = torch.optim.AdamW(list(net.parameters()) + [prototypes], lr=args.lr)
        for _ in range(args.epochs):
            net.train()
            order = rng.permutation(len(x))
            for start in range(0, len(order), args.batch_size):
                index = order[start:start + args.batch_size]
                feature = net(torch.as_tensor(x[index], device=args.device))
                target = labels[index]
                active_support = neighbors(prototypes.detach(), args.neighbors) if args.global_rebuild else support
                graph = relation_graph(prototypes, all_s, active_support, args.temperature, args.eta, not args.no_reliability)
                smooth = (1 - args.alpha) * prototypes + args.alpha * graph @ prototypes
                distance = torch.cdist(feature, smooth).square()
                loss = F.cross_entropy(-distance / args.logit_temperature, target)
                if old:
                    loss = loss + args.anchor_weight * (prototypes[:old] - previous_p).square().sum()
                    loss = loss + args.edge_weight * (graph[:old, :old] - previous_a).square().sum()
                    loss = loss + args.rank_weight * rank_loss(prototypes, previous_a, args.rank_margin)
                    boundary_terms = []
                    for i, c in enumerate(target):
                        hard = torch.nonzero(active_support[c, :old], as_tuple=True)[0]
                        if len(hard):
                            boundary_terms.append(F.relu(args.boundary_margin + distance[i, c] - distance[i, hard].min()))
                    if boundary_terms:
                        loss = loss + args.boundary_weight * torch.stack(boundary_terms).sum() / len(target)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        net.eval()
        final_z = embedded(net, x, args)
        if session == 0:
            base_dev = data(args.data, "dev", 0)
            validation_features = embedded(net, base_dev["x"], args)
            variances = [validation_features[base_dev["y"] == c].var(0, unbiased=True)
                         for c in new_classes if np.sum(base_dev["y"] == c) >= 2]
            prior = torch.stack(variances).mean(0).detach()
        final_s = torch.stack([final_z[labels == c].var(0, unbiased=True) if int((labels == c).sum()) > 1 else prior for c in new_classes])
        previous_p = prototypes.detach()
        previous_s = final_s if not old else torch.cat([previous_s, final_s])
        if args.global_rebuild:
            support = neighbors(previous_p, args.neighbors)
        previous_a = relation_graph(previous_p, previous_s, support, args.temperature, args.eta, not args.no_reliability).detach()
        edge_index, edge_weight = sparse_graph(previous_a)
        torch.save({"encoder": net.state_dict(), "prototypes": previous_p.cpu(), "dispersion": previous_s.cpu(),
                    "edge_index": edge_index, "edge_weight": edge_weight, "prior": prior.cpu(), "dimension": args.dimension,
                    "alpha": args.alpha, "logit_temperature": args.logit_temperature,
                    "training_seconds": time.perf_counter() - session_start}, out / f"session_{session}.pt")


def evaluate(args):
    sessions = json.loads((Path(args.data) / "sessions.json").read_text())
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    history, accuracies = [], {}
    base_graph = base_p = None
    for session in range(len(sessions)):
        session_start = time.perf_counter()
        saved = torch.load(Path(args.checkpoints) / f"session_{session}.pt", map_location=args.device, weights_only=True)
        net = Encoder(saved["dimension"]).to(args.device)
        net.load_state_dict(saved["encoder"])
        net.eval()
        p, adjacency = saved["prototypes"], restore_graph(saved, args.device)
        smooth = (1 - saved["alpha"]) * p + saved["alpha"] * adjacency @ p
        all_probability, all_labels = [], []
        for group in range(session + 1):
            test = data(args.data, "test", group)
            feature = embedded(net, test["x"], args)
            probability = (-torch.cdist(feature, smooth).square() / saved["logit_temperature"]).softmax(1).cpu().numpy()
            all_probability.append(probability)
            all_labels.append(test["y"])
        probability, labels = np.concatenate(all_probability), np.concatenate(all_labels)
        predicted, confidence = probability.argmax(1), probability.max(1)
        class_accuracy = {int(c): float(np.mean(predicted[labels == c] == c)) for c in np.unique(labels)}
        if session:
            old = list(accuracies)
            forgetting = float(np.mean([max(accuracies[c]) - class_accuracy[c] for c in old]))
            old_accuracy = float(np.mean(predicted[labels < sum(len(s) for s in sessions[:session])] == labels[labels < sum(len(s) for s in sessions[:session])]))
        else:
            forgetting, old_accuracy = None, None
        for c, value in class_accuracy.items():
            accuracies.setdefault(c, []).append(value)
        ece = 0
        for lower in np.arange(15) / 15:
            mask = (confidence > lower) & (confidence <= lower + 1 / 15)
            if mask.any():
                ece += mask.mean() * abs((predicted[mask] == labels[mask]).mean() - confidence[mask].mean())
        count_new = sum(len(s) for s in sessions[:session])
        new = labels >= count_new
        if base_graph is None:
            base_graph, base_p = adjacency.cpu().numpy(), p.cpu().numpy()
        base_count = len(base_graph)
        current_base = adjacency[:base_count, :base_count].cpu().numpy()
        base_cos = base_p @ base_p.T / np.maximum(np.linalg.norm(base_p, axis=1)[:, None] * np.linalg.norm(base_p, axis=1)[None], 1e-8)
        current = F.normalize(p[:base_count], dim=-1).cpu().numpy()
        current_cos = current @ current.T
        np.fill_diagonal(base_cos, -np.inf)
        np.fill_diagonal(current_cos, -np.inf)
        before_rank, after_rank = np.argsort(np.argsort(-base_cos, axis=1), axis=1), np.argsort(np.argsort(-current_cos, axis=1), axis=1)
        neighbor_mask = base_graph > 0
        history.append({"session": session, "accuracy": float(np.mean(predicted == labels)),
                        "balanced_accuracy": float(np.mean(list(class_accuracy.values()))), "old_accuracy": old_accuracy,
                        "new_accuracy": float(np.mean(predicted[new] == labels[new])), "forgetting": forgetting, "ece": float(ece),
                        "edge_drift": float(np.linalg.norm(current_base - base_graph)),
                        "rank_drift": float(np.mean(np.abs(before_rank[neighbor_mask] - after_rank[neighbor_mask]) > 1)) if neighbor_mask.any() else 0,
                        "training_seconds": saved["training_seconds"],
                        "evaluation_ms_per_window": 1000 * (time.perf_counter() - session_start) / len(labels),
                        "retained_statistics_bytes": int(p.numel() * p.element_size() + saved["dispersion"].numel() * saved["dispersion"].element_size() + saved["edge_index"].numel() * 8 + saved["edge_weight"].numel() * 4),
                        "class_accuracy": class_accuracy})
    (out / "metrics.json").write_text(json.dumps({"aia": float(np.mean([r["accuracy"] for r in history])), "sessions": history}, indent=2))
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot([r["session"] for r in history], [r["accuracy"] for r in history], marker="o")
    ax.set(xlabel="Session", ylabel="Cumulative accuracy")
    fig.tight_layout()
    fig.savefig(out / "incremental.png", dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--manifest", required=True)
    p.add_argument("--sessions", default="configs/sessions.json")
    p.add_argument("--output", default="data/prepared")
    p.add_argument("--window", type=int, default=2048)
    p.add_argument("--stride", type=int, default=1024)
    p.set_defaults(function=prepare)
    for command in ["train", "evaluate"]:
        p = sub.add_parser(command)
        p.add_argument("--data", default="data/prepared")
        p.add_argument("--output", default=f"runs/{command}")
        p.add_argument("--device", default="cpu")
        p.add_argument("--batch-size", type=int, default=32)
        if command == "train":
            for key, value in [("epochs", 40), ("dimension", 64), ("neighbors", 5), ("shots", 0), ("seed", 42)]:
                p.add_argument("--" + key, type=int, default=value)
            for key, value in [("lr", 1e-3), ("alpha", 0.3), ("temperature", 0.2), ("logit-temperature", 0.1), ("eta", 0.1), ("anchor-weight", 1), ("edge-weight", 1), ("rank-weight", 0.1), ("boundary-weight", 0.1), ("rank-margin", 0.1), ("boundary-margin", 0.2)]:
                p.add_argument("--" + key, type=float, default=value)
            p.add_argument("--global-rebuild", action="store_true")
            p.add_argument("--no-reliability", action="store_true")
            p.set_defaults(function=train)
        else:
            p.add_argument("--checkpoints", default="runs/train")
            p.set_defaults(function=evaluate)
    args = parser.parse_args()
    args.function(args)
