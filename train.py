from copy import deepcopy
import os
from os import makedirs
from os.path import join, splitext, basename
import gc
from pprint import pprint
import argparse
from datetime import datetime
from typing import List
import torch
import torchmetrics
import matplotlib.pyplot as plt
from torch.utils.data import Subset, DataLoader
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint, StochasticWeightAveraging
from lightning.pytorch import Trainer
import wandb
import yaml
from tqdm import tqdm

from datasets.mmhgdhgr import MultiModalHandGestureDatasetForHandGestureRecognition
from datasets.ml2hp import MotionLeap2Dataset

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
    run_name: str = None,
    disable_checkpointing: bool = True,
    limit_subjects: int = None,
    seed: int = 42,
):
    # setup
    torch.set_float32_matmul_precision("medium")

    # loads the configuration file
    num_workers = os.cpu_count()  # type: ignore
    with open(cfg, "r") as fp:
        cfg_dict = yaml.safe_load(fp)
    pprint(cfg_dict)

    # sets the random seed
    set_global_seed(seed=seed)

    # sets the logging folder
    datetime_str: str = datetime.now().strftime("%Y%m%d-%H%M")
    experiment_name: str = f"{cfg_dict['name']}-{datetime_str}"
    experiment_path: str = join(cfg_dict["checkpoints_path"], experiment_name)
    makedirs(experiment_path, exist_ok=True)

    # sets up the dataset(s)
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

    # sets up the validation scheme
    if cfg_dict["dataset"] == "ml2hp" or cfg_dict["validation"] == "loso":
        runs = get_loso_runs(dataset=dataset, limit_subjects=limit_subjects)
    elif cfg_dict["dataset"] in {"mmhgdhgr"}:
        runs = get_train_test_splits(dataset=dataset, limit_subjects=limit_subjects)
    elif cfg_dict["dataset"] in {"tiny_hgr"}:
        runs = get_optimistic_splits(dataset=dataset)
    else:
        raise NotImplementedError()

    # setup the model
    device = get_device_from_string(cfg_dict["device"])  # "cuda" or "cpu"
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
        # train_image_backbone=cfg_dict["train_image_backbone"],
        num_epochs=cfg_dict["max_epochs"],
        lr=cfg_dict["lr"],
    )

    # saves the initial weights of the model
    initial_state_dict_path = join(".", "initial_weights.pth")
    torch.save(model.state_dict(), initial_state_dict_path)

    # metas
    date = datetime.now().strftime("%Y%m%d_%H%M")
    cfg_name = splitext(basename(cfg))[0]
    experiment_name = f"{date}_{cfg_name}"
    if run_name is not None:
        experiment_name += f"_{run_name}"
    # saves the parameters used in the config
    with open(join(experiment_path, "cfg.yaml"), "w") as fp:
        yaml.dump(cfg_dict, fp, default_flow_style=False)

    # loops over runs
    for i_run, run in enumerate(runs):
        if cfg_dict["validation"] == "loso":
            print(
                f"doing run for subject {run['subject_id']} ({((i_run+1)/len(runs)) * 100:.1f}%)"  # type: ignore
            )
            run_name = run["subject_id"]  # type: ignore

        else:
            print(
                f"doing run {i_run+1} of {len(runs)} ({((i_run+1)/len(runs)) * 100:.1f}%)"
            )
            run_name = f"run_{i_run}"
        experiment_run_path = join(experiment_path, run_name)
        makedirs(experiment_run_path, exist_ok=True)

        # splits the dataset
        dataloader_train = DataLoader(
            dataset=Subset(dataset, indices=run["train_idx"]),  # type: ignore
            batch_size=cfg_dict["batch_size"],
            shuffle=True,
            pin_memory=False,
            num_workers=num_workers // 2,
            persistent_workers=True,
        )
        dataloader_val = DataLoader(
            dataset=Subset(dataset, indices=run["val_idx"]),  # type: ignore
            batch_size=cfg_dict["batch_size"],
            shuffle=False,
            pin_memory=False,
            num_workers=num_workers,
            persistent_workers=True,
        )
        if "test_idx" in run.keys():
            dataloader_test = DataLoader(
                dataset=Subset(dataset, indices=run["test_idx"]),  # type: ignore
                batch_size=cfg_dict["batch_size"],
                shuffle=False,
                pin_memory=False,
                num_workers=num_workers,
                persistent_workers=True,
            )
        else:
            dataloader_test = dataloader_val

        # initialize the model
        model.load_state_dict(
            torch.load(initial_state_dict_path, map_location=device), strict=False
        )
        model.to(device)

        wandb_logger = WandbLogger(
            project=cfg_dict["dataset"],
            name=experiment_name,
            log_model=False,
            prefix=run_name,
        )

        # do the training
        checkpoint_callback = ModelCheckpoint(
            dirpath=experiment_run_path,
            filename="{epoch:01d}-{cls_f1_loss:.4f}-{cls_f1_val:.4f}",
            monitor=f"cls_f1_val",
            mode="max",
            every_n_epochs=1,
        )
        trainer = Trainer(
            logger=wandb_logger,
            accelerator=device,
            log_every_n_steps=10,
            precision="16-mixed",
            gradient_clip_val=1.0,
            max_epochs=cfg_dict["max_epochs"],
            accumulate_grad_batches=cfg_dict["accumulate_grad_batches"],
            enable_model_summary=True,
            enable_checkpointing=True,
            default_root_dir=experiment_run_path,
            callbacks=[checkpoint_callback],
        )
        trainer.fit(model, dataloader_train, dataloader_val)

        # compute metrics using the final model
        model = HandGestureRecognitionModel.load_from_checkpoint(
            checkpoint_callback.best_model_path,
            map_location=device,
            strict=False,
        )
        metrics = trainer.test(model, dataloader_test)
        with open(join(experiment_run_path, "metrics.yaml"), "w") as fp:
            yaml.dump(metrics, fp, default_flow_style=False)

        # computes and saves the confusion matrix on the test set
        model.eval()
        model.to(device)
        all_preds, all_labels = [], []
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
        confmat = torchmetrics.functional.confusion_matrix(
            preds=torch.cat(all_preds, dim=0),
            target=torch.cat(all_labels, dim=0),
            task="multiclass",
            num_classes=dataset.num_labels,
        )
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.imshow(confmat, cmap="Blues")
        ax.set_xlabel("predicted")
        ax.set_ylabel("true")
        ax.set_title(f"confusion matrix - {run_name}")
        fig.savefig(join(experiment_run_path, "confusion_matrix.png"))
        plt.close(fig)
        with open(join(experiment_run_path, "confusion_matrix.yaml"), "w") as fp:
            yaml.dump(confmat.tolist(), fp, default_flow_style=False)

        # eventually removes the checkpoint
        if disable_checkpointing:
            os.remove(checkpoint_callback.best_model_path)

    wandb.finish()

    # frees some memory
    del dataset
    gc.collect()


if __name__ == "__main__":
    # arguments parsing
    parser = argparse.ArgumentParser(description="Train a model")
    parser.add_argument(
        "--cfg", type=str, help="Path to the configuration", required=True
    )
    parser.add_argument(
        "--disable_checkpointing",
        default=False,
        action="store_true",
        help="Whether not to save model's weights",
    )
    parser.add_argument(
        "--run_name",
        type=str,
        help="The name to prepend to the run when logging",
        required=False,
    )
    parser.add_argument(
        "--limit_subjects",
        type=int,
        default=None,
        help="The number of subjects to consider for training",
        required=False,
    )
    line_args = vars(parser.parse_args())

    main(**line_args)
