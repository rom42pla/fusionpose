"""Trains a configuration like `train.py`, but only on `limit_subjects` subjects,
and only produces a confusion matrix (no wandb logging, no checkpointing).
"""
import argparse
import os
from os.path import join
from pprint import pprint

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import torch
import torchmetrics
import yaml
from lightning.pytorch import Trainer
from torch.utils.data import DataLoader, Subset

from datasets.ml2hp import MotionLeap2Dataset
from datasets.mmhgdhgr import MultiModalHandGestureDatasetForHandGestureRecognition
from datasets.tiny_hgr import TinyHandGestureRecognitionDataset
from model import HandGestureRecognitionModel
from utils import (
    get_device_from_string,
    get_loso_runs,
    get_optimistic_splits,
    get_train_test_splits,
    set_global_seed,
)


def main(
    cfg: str,
    dataset_path: str = None,
    limit_subjects: int = None,
    max_epochs: int = 3,
    seed: int = 42,
):
    num_workers = os.cpu_count()
    torch.set_float32_matmul_precision("medium")
    set_global_seed(seed=seed)

    with open(cfg, "r") as fp:
        cfg_dict = yaml.safe_load(fp)
    if dataset_path is not None:
        cfg_dict["dataset_path"] = dataset_path
    pprint(cfg_dict)

    # sets up the dataset
    if cfg_dict["dataset"] == "ml2hp":
        dataset = MotionLeap2Dataset(
            dataset_path=cfg_dict["dataset_path"],
            normalize_landmarks=cfg_dict["normalize_landmarks"],
        )
        dataset.set_mode(
            return_horizontal_images=cfg_dict["use_horizontal_image"],
            return_vertical_images=cfg_dict["use_vertical_image"],
            return_horizontal_landmarks=cfg_dict["use_horizontal_landmarks"],
            return_vertical_landmarks=cfg_dict["use_vertical_landmarks"],
        )
    elif cfg_dict["dataset"] == "mmhgdhgr":
        dataset = MultiModalHandGestureDatasetForHandGestureRecognition(
            dataset_path=cfg_dict["dataset_path"],
            normalize_landmarks=cfg_dict["normalize_landmarks"],
            img_size=224,
        )
        dataset.set_mode(
            return_images=any(
                [cfg_dict["use_horizontal_image"], cfg_dict["use_vertical_image"]]
            ),
            return_landmarks=any(
                [
                    cfg_dict["use_horizontal_landmarks"],
                    cfg_dict["use_vertical_landmarks"],
                ]
            ),
        )
    elif cfg_dict["dataset"] == "tiny_hgr":
        dataset = TinyHandGestureRecognitionDataset(
            dataset_path=cfg_dict["dataset_path"],
            normalize_landmarks=cfg_dict["normalize_landmarks"],
            img_size=224,
        )
        dataset.set_mode(
            return_images=any(
                [cfg_dict["use_horizontal_image"], cfg_dict["use_vertical_image"]]
            ),
            return_landmarks=any(
                [
                    cfg_dict["use_horizontal_landmarks"],
                    cfg_dict["use_vertical_landmarks"],
                ]
            ),
        )
    else:
        raise NotImplementedError()

    # sets up the validation scheme, limited to `limit_subjects` subjects/runs
    if cfg_dict["dataset"] == "ml2hp" or cfg_dict["validation"] == "loso":
        runs = get_loso_runs(dataset=dataset, limit_subjects=limit_subjects)
    elif cfg_dict["dataset"] in {"mmhgdhgr"}:
        runs = get_train_test_splits(dataset=dataset, limit_subjects=limit_subjects)
    elif cfg_dict["dataset"] in {"tiny_hgr"}:
        runs = get_optimistic_splits(dataset=dataset)
    else:
        raise NotImplementedError()

    print(f"{len(runs)} run(s)")

    # sets up the model
    device = get_device_from_string(cfg_dict["device"])
    model = HandGestureRecognitionModel(
        num_labels=dataset.num_labels,
        num_landmarks=dataset.num_landmarks,
        img_channels=dataset.img_channels,
        img_size=dataset.img_size,
        image_backbone_name=cfg_dict["image_backbone_name"],
        landmarks_backbone_name=cfg_dict["landmarks_backbone_name"],
        use_horizontal_images=cfg_dict["use_horizontal_image"],
        use_vertical_images=cfg_dict["use_vertical_image"],
        use_horizontal_landmarks=cfg_dict["use_horizontal_landmarks"],
        use_vertical_landmarks=cfg_dict["use_vertical_landmarks"],
        num_epochs=max_epochs,
        lr=cfg_dict["lr"],
    )

    initial_state_dict_path = "./initial_weights.pth"
    torch.save(model.state_dict(), initial_state_dict_path)

    # trains and predicts on every run, accumulating preds/labels across all of them
    all_preds, all_labels = [], []
    for i_run, run in enumerate(runs):
        run_name = (
            run["subject_id"] if cfg_dict["validation"] == "loso" else f"run_{i_run}"
        )
        print(f"doing run {run_name} ({(i_run + 1) / len(runs) * 100:.1f}%)")

        dataloader_train = DataLoader(
            dataset=Subset(dataset, indices=run["train_idx"]),
            batch_size=cfg_dict["batch_size"],
            shuffle=True,
            pin_memory=False,
            num_workers=num_workers // 2,
            persistent_workers=True,
        )
        dataloader_val = DataLoader(
            dataset=Subset(dataset, indices=run["val_idx"]),
            batch_size=cfg_dict["batch_size"],
            shuffle=False,
            pin_memory=False,
            num_workers=num_workers,
            persistent_workers=True,
        )
        if "test_idx" in run.keys():
            dataloader_test = DataLoader(
                dataset=Subset(dataset, indices=run["test_idx"]),
                batch_size=cfg_dict["batch_size"],
                shuffle=False,
                pin_memory=False,
                num_workers=num_workers,
                persistent_workers=True,
            )
        else:
            dataloader_test = dataloader_val

        # re-initializes the model for this run
        model.load_state_dict(
            torch.load(initial_state_dict_path, map_location=device), strict=False
        )
        model.to(device)

        trainer = Trainer(
            logger=False,
            accelerator=device,
            log_every_n_steps=10,
            precision="16-mixed",
            gradient_clip_val=1.0,
            max_epochs=max_epochs,
            accumulate_grad_batches=cfg_dict["accumulate_grad_batches"],
            enable_model_summary=False,
            enable_checkpointing=False,
        )
        trainer.fit(model, dataloader_train, dataloader_val)

        # collects predictions and labels on the test set
        model.eval()
        model.to(device)
        with torch.no_grad():
            for batch in dataloader_test:
                if "image" in batch:
                    if model.use_horizontal_images:
                        batch["image_horizontal"] = batch["image"]
                    elif model.use_vertical_images:
                        batch["image_vertical"] = batch["image"]
                if "landmarks" in batch:
                    if model.use_horizontal_landmarks:
                        batch["landmarks_horizontal"] = batch["landmarks"]
                    elif model.use_vertical_landmarks:
                        batch["landmarks_vertical"] = batch["landmarks"]
                batch.pop("image", None)
                batch.pop("landmarks", None)
                label = batch.pop("label")
                outs = model(**batch)
                all_preds.append(outs["cls_logits"].argmax(dim=1).cpu())
                all_labels.append(label.cpu())

    # computes the (aggregated) confusion matrix
    confmat = torchmetrics.functional.confusion_matrix(
        preds=torch.cat(all_preds, dim=0),
        target=torch.cat(all_labels, dim=0),
        task="multiclass",
        num_classes=dataset.num_labels,
    )
    np.save(f"{cfg_dict['dataset']}_cm.npy", confmat.numpy())
    print(confmat)

    # plots the confusion matrix
    class_names = [
        name for name, _ in sorted(dataset.poses_dict.items(), key=lambda item: item[1])
    ]

    fig, ax = plt.subplots(figsize=(8, 8), tight_layout=True)
    confmat_norm = confmat.float() / confmat.sum(dim=1, keepdim=True).clamp(min=1)
    im = ax.imshow(confmat_norm * 100, cmap="Blues", vmin=0, vmax=100)

    t = 0  # don't show values under this threshold (%)

    for i in range(confmat.shape[0]):
        for j in range(confmat.shape[1]):
            pct = confmat_norm[i, j].item() * 100
            if pct < t:
                continue
            label = "0" if pct == 0 else f"{pct:.1f}"
            ax.text(
                j, i, label,
                ha="center", va="center",
                color="white" if pct > 50 else "black",
                fontsize=8,
            )

    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45)
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("Ground truth")
    cbar = fig.colorbar(im, ax=ax, label="Actual class (%)", shrink=0.8, pad=0.02)
    cbar.set_ticks([0, 20, 40, 60, 80, 100])
    cbar.ax.yaxis.set_major_formatter(mtick.FuncFormatter(lambda x, _: f"{x:.1f}"))
    plt.savefig(f"{cfg_dict['dataset']}_confusion_matrix.png", dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train a configuration and produce a confusion matrix"
    )
    parser.add_argument(
        "--cfg", type=str, help="Path to the configuration", required=True
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        default=None,
        help="Overrides the dataset_path in the configuration file",
        required=False,
    )
    parser.add_argument(
        "--limit_subjects",
        type=int,
        default=None,
        help="The number of subjects to consider for training",
        required=False,
    )
    parser.add_argument(
        "--max_epochs",
        type=int,
        default=3,
        help="The number of epochs to train for",
        required=False,
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="The random seed to use", required=False
    )
    line_args = vars(parser.parse_args())

    main(**line_args)
