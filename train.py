from typing import Tuple, Dict

import random
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from torch_geometric.data import Data

from .config import ExperimentConfig
from .data import MultiDomainBreastUSDataset, DomainHarmonizer
from .models import USFMAEBackbone, GraphClassifier
from .graph import build_knn_graph
from .acquisition import mc_dropout_predict, compute_acquisition_scores


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def compute_embeddings(
    dataset: MultiDomainBreastUSDataset,
    backbone: USFMAEBackbone,
    device: torch.device,
    batch_size: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute embeddings for all dataset samples using the frozen backbone.
    Returns:
        embeddings: [N, d]
        labels: [N]
        domains: [N]
    """
    backbone.eval()
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )

    all_emb = []
    all_labels = []
    all_domains = []

    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)
            labels = batch["label"].to("cpu")
            domains = batch["domain"].to("cpu")

            feats = backbone(images).detach().cpu()
            all_emb.append(feats)
            all_labels.append(labels)
            all_domains.append(domains)

    embeddings = torch.cat(all_emb, dim=0)
    labels = torch.cat(all_labels, dim=0)
    domains = torch.cat(all_domains, dim=0)
    return embeddings, labels, domains


def create_split_masks(
    dataset: MultiDomainBreastUSDataset,
    train_split: str,
    val_split: str,
    test_split: str,
) -> Dict[str, torch.Tensor]:
    N = len(dataset)
    train_idx = dataset.get_split_indices(train_split)
    val_idx = dataset.get_split_indices(val_split)
    test_idx = dataset.get_split_indices(test_split)

    train_mask = torch.zeros(N, dtype=torch.bool)
    val_mask = torch.zeros(N, dtype=torch.bool)
    test_mask = torch.zeros(N, dtype=torch.bool)

    train_mask[train_idx] = True
    val_mask[val_idx] = True
    test_mask[test_idx] = True

    return {
        "train_mask": train_mask,
        "val_mask": val_mask,
        "test_mask": test_mask,
    }


def create_initial_labeled_mask(
    labels: torch.Tensor,
    train_mask: torch.Tensor,
    initial_fraction: float,
) -> torch.Tensor:
    """
    Sample an initial labeled subset from the training pool.
    Ensures at least one example per class if possible.
    """
    N = labels.shape[0]
    labeled_mask = torch.zeros(N, dtype=torch.bool)

    train_indices = torch.nonzero(train_mask, as_tuple=False).view(-1)
    num_train = train_indices.numel()
    target_num = max(1, int(initial_fraction * num_train))

    # Ensure at least one example per class if possible
    train_labels = labels[train_mask]
    unique_classes = torch.unique(train_labels)

    selected_indices = []
    for c in unique_classes.tolist():
        class_indices = train_indices[labels[train_indices] == c]
        if class_indices.numel() > 0:
            rand_idx = class_indices[torch.randint(0, class_indices.numel(), (1,))].item()
            selected_indices.append(rand_idx)

    remaining = max(0, target_num - len(selected_indices))
    if remaining > 0:
        remaining_pool = train_indices[
            ~torch.isin(train_indices, torch.tensor(selected_indices))
        ]
        if remaining_pool.numel() > 0:
            perm = torch.randperm(remaining_pool.numel())
            sampled = remaining_pool[perm[:remaining]].tolist()
            selected_indices.extend(sampled)

    labeled_mask[torch.tensor(selected_indices, dtype=torch.long)] = True
    return labeled_mask


def train_one_round(
    model: GraphClassifier,
    data: Data,
    train_mask: torch.Tensor,
    labeled_mask: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    num_epochs: int,
    device: torch.device,
) -> None:
    model.to(device)
    data = data.to(device)
    criterion = nn.CrossEntropyLoss()

    train_labeled_mask = (train_mask & labeled_mask).to(device)

    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()

        logits = model(data)
        loss = criterion(logits[train_labeled_mask], data.y[train_labeled_mask])

        loss.backward()
        optimizer.step()


def evaluate(
    model: GraphClassifier,
    data: Data,
    mask: torch.Tensor,
    device: torch.device,
) -> Dict[str, float]:
    model.eval()
    data = data.to(device)
    eval_mask = mask.to(device)

    with torch.no_grad():
        logits = model(data)
        probs = F.softmax(logits, dim=1)
        preds = probs.argmax(dim=1)

    y_true = data.y[eval_mask]
    y_pred = preds[eval_mask]

    correct = (y_true == y_pred).sum().item()
    total = y_true.numel()
    acc = correct / total if total > 0 else 0.0

    return {"accuracy": acc}


def run_active_learning(exp_cfg: ExperimentConfig) -> None:
    """
    Full USF-GAL active learning loop:
      1. Load dataset
      2. Fit domain harmonizer
      3. Compute USF-MAE embeddings
      4. Build k-NN graph
      5. Active learning rounds: train GNN, acquire new labels, evaluate
    """
    set_seed(exp_cfg.training.random_seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Load dataset
    dataset = MultiDomainBreastUSDataset(
        csv_path=exp_cfg.data.csv_path,
        root_dir=exp_cfg.data.root_dir,
        transform=None,
    )

    num_classes = len(dataset.label_names)
    num_domains = len(dataset.domain_names)

    # 2. Domain harmonization fit (on training split)
    harmonizer = DomainHarmonizer(
        num_domains=num_domains,
        idx2domain=dataset.idx2domain,
        img_size=exp_cfg.model.img_size,
        reference_domain_idx=0,  # can be set to the index of a specific source domain
    )
    harmonizer.fit(dataset, split_name=exp_cfg.data.train_split_name)
    dataset.transform = harmonizer

    # 3. Backbone + embeddings
    backbone = USFMAEBackbone(
        model_name=exp_cfg.model.backbone_name,
        pretrained=exp_cfg.model.backbone_pretrained,
    ).to(device)
    embeddings, labels, domains = compute_embeddings(
        dataset, backbone, device, exp_cfg.training.batch_size_embeddings
    )
    # Overwrite embedding_dim from actual backbone
    exp_cfg.model.embedding_dim = embeddings.shape[1]

    # 4. Build graph and split masks
    masks = create_split_masks(
        dataset,
        train_split=exp_cfg.data.train_split_name,
        val_split=exp_cfg.data.val_split_name,
        test_split=exp_cfg.data.test_split_name,
    )
    graph_data = build_knn_graph(
        embeddings=embeddings,
        labels=labels,
        domains=domains,
        k=exp_cfg.model.knn_k,
    )

    train_mask = masks["train_mask"]
    val_mask = masks["val_mask"]
    test_mask = masks["test_mask"]

    # 5. Active learning loop (on training pool)
    labeled_mask = create_initial_labeled_mask(
        labels=labels,
        train_mask=train_mask,
        initial_fraction=exp_cfg.al.initial_labeled_fraction,
    )

    for round_idx in range(exp_cfg.al.num_rounds):
        print(f"\n=== Active Learning Round {round_idx + 1}/{exp_cfg.al.num_rounds} ===")
        print(f"Labeled samples in train pool: {int((train_mask & labeled_mask).sum().item())}")

        # Initialize / warm-start GNN
        model = GraphClassifier(
            embedding_dim=exp_cfg.model.embedding_dim,
            num_classes=num_classes,
            num_domains=num_domains,
            hidden_dim=exp_cfg.model.gnn_hidden_dim,
            num_layers=exp_cfg.model.gnn_num_layers,
            domain_emb_dim=exp_cfg.model.domain_emb_dim,
            dropout=exp_cfg.model.dropout,
        )

        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=exp_cfg.training.learning_rate,
            weight_decay=exp_cfg.training.weight_decay,
        )

        # Train on current labeled subset
        train_one_round(
            model=model,
            data=graph_data,
            train_mask=train_mask,
            labeled_mask=labeled_mask,
            optimizer=optimizer,
            num_epochs=exp_cfg.training.num_epochs_per_round,
            device=device,
        )

        # Evaluate on validation and test sets
        val_metrics = evaluate(model, graph_data, val_mask, device)
        test_metrics = evaluate(model, graph_data, test_mask, device)
        print(f"Validation accuracy: {val_metrics['accuracy']:.4f}")
        print(f"Test accuracy:       {test_metrics['accuracy']:.4f}")

        # Last round: do not acquire more labels
        if round_idx == exp_cfg.al.num_rounds - 1:
            break

        # MC-dropout predictions for acquisition
        graph_data_device = graph_data.to(device)
        mean_probs = mc_dropout_predict(
            model=model,
            data=graph_data_device,
            mc_passes=exp_cfg.al.mc_dropout_passes,
            device=device,
        ).cpu()
        graph_data = graph_data_device.to("cpu")

        # Compute acquisition scores and select new samples from unlabeled training pool
        scores = compute_acquisition_scores(
            mean_probs=mean_probs,
            embeddings=embeddings,
            labels=labels,
            domains=domains,
            labeled_mask=labeled_mask,
            train_mask=train_mask,
            al_cfg=exp_cfg.al,
        )

        unlabeled_train_mask = train_mask & (~labeled_mask)
        candidate_indices = torch.nonzero(unlabeled_train_mask, as_tuple=False).view(-1)

        if candidate_indices.numel() == 0:
            print("No more unlabeled samples in training pool.")
            break

        # Top-B acquisition
        B = min(exp_cfg.al.acquisition_batch_size, candidate_indices.numel())
        topk_scores, topk_indices = torch.topk(
            scores[candidate_indices],
            k=B,
            largest=True,
        )
        newly_selected = candidate_indices[topk_indices]

        labeled_mask[newly_selected] = True
        print(f"Acquired {B} new labeled samples.")
