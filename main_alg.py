import argparse

from .config import (
    ExperimentConfig,
    DataConfig,
    ModelConfig,
    ActiveLearningConfig,
    TrainingConfig,
)
from .trainer import run_active_learning


def build_default_experiment(
    csv_path: str,
    root_dir: str,
) -> ExperimentConfig:
    data_cfg = DataConfig(
        csv_path=csv_path,
        root_dir=root_dir,
        train_split_name="train",
        val_split_name="val",
        test_split_name="test",
    )

    model_cfg = ModelConfig(
        img_size=256,
        backbone_name="vit_base_patch16_224",  # replace with USF-MAE encoder name if available
        backbone_pretrained=False,             # set True if loading pretrained weights
        embedding_dim=768,                     # overwritten at runtime
        gnn_hidden_dim=256,
        gnn_num_layers=2,
        domain_emb_dim=16,
        dropout=0.2,
        knn_k=10,
    )

    al_cfg = ActiveLearningConfig(
        initial_labeled_fraction=0.05,
        acquisition_batch_size=32,
        num_rounds=10,
        mc_dropout_passes=10,
        lambda_uncertainty=1.0,
        lambda_coverage=1.0,
        lambda_domain=1.0,
    )

    training_cfg = TrainingConfig(
        num_epochs_per_round=30,
        learning_rate=1e-3,
        weight_decay=1e-4,
        batch_size_embeddings=32,
        random_seed=42,
    )

    exp_cfg = ExperimentConfig(
        data=data_cfg,
        model=model_cfg,
        al=al_cfg,
        training=training_cfg,
    )
    return exp_cfg


def main():
    parser = argparse.ArgumentParser(description="USF-GAL Active Learning Experiment")
    parser.add_argument(
        "--csv",
        type=str,
        required=True,
        help="Path to metadata CSV (with columns: image_path,label,domain,split).",
    )
    parser.add_argument(
        "--root",
        type=str,
        required=True,
        help="Root directory containing images.",
    )
    args = parser.parse_args()

    exp_cfg = build_default_experiment(csv_path=args.csv, root_dir=args.root)
    run_active_learning(exp_cfg)


if __name__ == "__main__":
    main()
