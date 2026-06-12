from dataclasses import dataclass


@dataclass
class ModelConfig:
    # Image / backbone
    img_size: int = 256
    backbone_name: str = "vit_base_patch16_224"  # USF-MAE-compatible ViT
    backbone_pretrained: bool = False            # set True if you load public weights
    embedding_dim: int = 768                     # overwritten at runtime from backbone.num_features

    # GNN
    gnn_hidden_dim: int = 256
    gnn_num_layers: int = 2
    domain_emb_dim: int = 16
    dropout: float = 0.2

    # Graph construction
    knn_k: int = 10


@dataclass
class ActiveLearningConfig:
    initial_labeled_fraction: float = 0.05
    acquisition_batch_size: int = 32
    num_rounds: int = 10
    mc_dropout_passes: int = 10

    lambda_uncertainty: float = 1.0
    lambda_coverage: float = 1.0
    lambda_domain: float = 1.0


@dataclass
class TrainingConfig:
    num_epochs_per_round: int = 30
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    batch_size_embeddings: int = 32
    random_seed: int = 42


@dataclass
class DataConfig:
    csv_path: str = "data/metadata.csv"   # CSV with columns: image_path,label,domain,split
    root_dir: str = "data/images"         # root directory for image files
    train_split_name: str = "train"
    val_split_name: str = "val"
    test_split_name: str = "test"


@dataclass
class ExperimentConfig:
    data: DataConfig
    model: ModelConfig
    al: ActiveLearningConfig
    training: TrainingConfig
