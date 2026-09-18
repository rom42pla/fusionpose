from abc import abstractmethod
import itertools
import math
from pprint import pprint
import time
from types import FunctionType
import torch
import torch.nn.functional as F
import lightning as pl
import torchmetrics
from typing import Any, Callable, List, Tuple
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import timm
from transformers import AutoModel, CLIPVisionModel, ConvNextV2ForImageClassification
import gc
import torchvision.transforms as T

from datasets.ml2hp import MotionLeap2Dataset
from utils import set_global_seed


class HandGestureRecognitionModel(pl.LightningModule):
    _possible_landmarks_backbones = {None, "linear", "mlp"}
    _possible_image_backbones = {
        None,
        "resnet18",
        "convnextv2-t",
        "convnextv2-b",
        "clip-b",
        "dinov2-s",
        "dinov2-b",
    }
    _dont_save_image_backbone = True

    def __init__(
        self,
        num_labels: int,
        num_landmarks: int,
        use_horizontal_images: bool,
        use_vertical_images: bool,
        use_horizontal_landmarks: bool,
        use_vertical_landmarks: bool,
        img_channels: int,
        image_backbone_name: str = "resnet18",
        landmarks_backbone_name: str = "mlp",
        use_data_augmentation: bool = True,
        lr: float = 1e-3,
        num_epochs: int = 1,
        linear_dropout_p: float = 0.2,
        train_image_backbone: bool = False,
        **kwargs,
    ):
        super(HandGestureRecognitionModel, self).__init__()
        assert isinstance(use_horizontal_images, bool)
        assert isinstance(use_vertical_images, bool)
        assert isinstance(use_horizontal_landmarks, bool)
        assert isinstance(use_vertical_landmarks, bool)
        self.use_horizontal_images = use_horizontal_images
        self.use_vertical_images = use_vertical_images
        self.use_horizontal_landmarks = use_horizontal_landmarks
        self.use_vertical_landmarks = use_vertical_landmarks

        self.linear_dropout_p = linear_dropout_p

        # parses channels
        if (
            self.use_horizontal_images or self.use_vertical_images
        ) and image_backbone_name is None:
            image_backbone_name = "resnet18"
        self.image_backbone_name = image_backbone_name
        if not any([self.use_horizontal_images, self.use_vertical_images]):
            self.img_channels = 0
            image_backbone_name = None
        else:
            assert img_channels > 0, f"got {img_channels=}, expected > 0"
            self.img_channels = img_channels

        # parses number of landmarks
        if (
            self.use_horizontal_landmarks or self.use_vertical_landmarks
        ) and landmarks_backbone_name is None:
            landmarks_backbone_name = "linear"
        self.landmarks_backbone_name = landmarks_backbone_name
        if not any([self.use_horizontal_landmarks, self.use_vertical_landmarks]):
            self.num_landmarks = 0
            landmarks_backbone_name = None
        else:
            assert num_landmarks > 0, f"got {self.num_landmarks=}, expected > 0"
            self.num_landmarks = num_landmarks

        # parses number of labels
        assert isinstance(num_labels, int) and num_labels > 0, num_labels
        self.num_classes = num_labels

        # image backbone
        assert isinstance(
            train_image_backbone, bool
        ), f"got {train_image_backbone} ({type(train_image_backbone)})"
        self.train_image_backbone = train_image_backbone
        self.adapter = nn.Sequential(
            nn.Conv2d(
                self.img_channels, 32, kernel_size=1, stride=1, padding=0, bias=False
            ),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(32, 3, kernel_size=3, stride=1, padding=1, bias=True),
        )
        (
            self.image_features_size,
            self.img_size,
            self.img_mean,
            self.img_std,
            self._image_backbone,
            self.image_embedder,
        ) = self._parse_image_backbone(name=self.image_backbone_name)

        # landmarks backbone
        assert (
            landmarks_backbone_name in self._possible_landmarks_backbones
        ), f"got {landmarks_backbone_name}, expected one of {self._possible_landmarks_backbones}"
        self.landmarks_backbone_name = landmarks_backbone_name
        (
            self.landmarks_features_size,
            self.landmarks_backbone,
            self.landmarks_embedder,
        ) = self._parse_landmarks_backbone(name=self.landmarks_backbone_name)

        # classifier
        (
            self.image_features_embedder,
            self.landmarks_features_embedder,
            self.cls_head,
            self.classify,
        ) = self._parse_merging_method(name="concatenate")

        # image transforms
        self.image_transforms_train = [
            T.Resize(self.img_size),
            T.CenterCrop(self.img_size),
        ]
        assert isinstance(
            use_data_augmentation, bool
        ), f"got {use_data_augmentation}, expected bool"
        self.use_data_augmentation = use_data_augmentation
        if self.use_data_augmentation:
            self.image_transforms_train.extend(
                [
                    T.RandomHorizontalFlip(p=0.5),
                    T.RandomAffine(
                        degrees=45, translate=(0.02, 0.02), scale=(0.8, 1.2)
                    ),
                    # T.RandomErasing(p=0.5, scale=(0.02, 0.1), ratio=(0.3, 3.3)),
                ]
            )  # type: ignore
        self.image_transforms_train = T.Compose(self.image_transforms_train)
        self.image_transforms_val_test = T.Compose(
            [
                T.Resize(self.img_size),
            ]
        )
        # optimizer params
        assert lr > 0
        self.lr = lr

        assert num_epochs > 0 and isinstance(num_epochs, int), num_epochs
        self.num_epochs = num_epochs

        self.epoch_metrics = {}

        self.save_hyperparameters(ignore=["epoch_metrics"])

    # returns image features size, the image backbone and the function to call the backbone
    def _parse_image_backbone(
        self, name
    ) -> Tuple[int, int, List[float], List[float], nn.Module, Callable]:
        assert (
            name in self._possible_image_backbones
        ), f"got {name}, expected one of {self._possible_image_backbones}"
        mean, std = [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]
        if name is None:
            return 0, 0, mean, std, nn.Identity(), nn.Identity()
        elif name == "resnet18":
            image_features_size = 512
            image_backbone = timm.create_model("resnet18.a1_in1k", pretrained=True)
            # image_backbone.conv1 = self.adapter
            image_backbone.fc = nn.Identity()
            image_size = 288
            image_embedder = lambda imgs: image_backbone(imgs)
        elif name == "convnextv2-t":
            image_features_size = 768
            image_backbone = ConvNextV2ForImageClassification.from_pretrained(
                "facebook/convnextv2-tiny-22k-224"
            )
            # image_backbone.convnextv2.embeddings.patch_embeddings = self.adapter
            # image_backbone.config.num_channels = self.img_channels
            # image_backbone.convnextv2.embeddings.num_channels = self.img_channels
            image_backbone.classifier = nn.Identity()
            image_size = 224
            image_embedder = lambda imgs: image_backbone(imgs).logits
            mean, std = (
                [0.485, 0.456, 0.406],
                [0.229, 0.224, 0.225],
            )
        elif name == "convnextv2-b":
            image_features_size = 1024
            image_backbone = ConvNextV2ForImageClassification.from_pretrained(
                "facebook/convnextv2-base-22k-224"
            )
            # image_backbone.convnextv2.embeddings.patch_embeddings = self.adapter
            # image_backbone.config.num_channels = self.img_channels
            # image_backbone.convnextv2.embeddings.num_channels = self.img_channels
            image_backbone.classifier = nn.Identity()
            image_size = 224
            image_embedder = lambda imgs: image_backbone(imgs).logits
            mean, std = (
                [0.485, 0.456, 0.406],
                [0.229, 0.224, 0.225],
            )
        elif name == "clip-b":
            image_features_size = 768
            image_backbone = CLIPVisionModel.from_pretrained(
                "openai/clip-vit-base-patch32"
            )
            # image_backbone.config.num_channels = self.img_channels
            # image_backbone.vision_model.embeddings.patch_embedding = self.adapter
            image_size = 224
            image_embedder = lambda imgs: image_backbone(
                pixel_values=imgs
            ).pooler_output
            mean, std = (
                [0.48145466, 0.4578275, 0.40821073],
                [0.26862954, 0.26130258, 0.27577711],
            )
        elif name == "dinov2-s":
            image_features_size = 384
            image_backbone = AutoModel.from_pretrained("facebook/dinov2-small")
            # image_backbone.config.num_channels = self.img_channels
            # image_backbone.embeddings.patch_embeddings.num_channels = self.img_channels
            # image_backbone.embeddings.patch_embeddings.projection = self.adapter
            # image_size = 518
            image_size = 224
            image_embedder = lambda imgs: image_backbone(
                pixel_values=imgs
            ).pooler_output
            mean, std = (
                [0.485, 0.456, 0.406],
                [0.229, 0.224, 0.225],
            )
        elif name == "dinov2-b":
            image_features_size = 768
            image_backbone = AutoModel.from_pretrained("facebook/dinov2-base")
            # image_backbone.config.num_channels = self.img_channels
            # image_backbone.embeddings.patch_embeddings.num_channels = self.img_channels
            # image_backbone.embeddings.patch_embeddings.projection = self.adapter
            # image_size = 518
            image_size = 224
            image_embedder = lambda imgs: image_backbone(
                pixel_values=imgs
            ).pooler_output
            mean, std = (
                [0.485, 0.456, 0.406],
                [0.229, 0.224, 0.225],
            )
        else:
            raise NotImplementedError(
                f"got image backbone '{name}', expected one of {self._possible_image_backbones}"
            )
        # eventually freezes image backbone's parameters
        for param in image_backbone.parameters():
            param.requires_grad = self.train_image_backbone
        return (
            image_features_size,
            image_size,
            mean,
            std,
            image_backbone,
            image_embedder,
        )

    def _parse_landmarks_backbone(self, name) -> Tuple[int, nn.Module, Callable]:
        landmarks_features_size = 768
        if name is None:
            return 0, nn.Identity(), nn.Identity()
        if name == "unprocessed":
            landmarks_features_size = self.num_landmarks
            landmarks_backbone = nn.Identity()
        elif name == "linear":
            landmarks_backbone = nn.Linear(self.num_landmarks, landmarks_features_size)
        elif name == "mlp":
            landmarks_backbone = nn.Sequential(
                # nn.Dropout(self.linear_dropout_p),
                nn.Linear(self.num_landmarks, 512 * 4),
                nn.LayerNorm(512 * 4),
                nn.LeakyReLU(inplace=True),
                nn.Dropout(self.linear_dropout_p),
                nn.Linear(512 * 4, landmarks_features_size),
                nn.LayerNorm(landmarks_features_size),
                (
                    nn.LeakyReLU(inplace=True)
                    if not (self.use_horizontal_images or self.use_vertical_images)
                    else nn.Identity()
                ),
                (
                    nn.Dropout(self.linear_dropout_p)
                    if not (self.use_horizontal_images or self.use_vertical_images)
                    else nn.Identity()
                ),
            )
            # landmarks_backbone = nn.Sequential(
            #     nn.Linear(self.num_landmarks, landmarks_features_size),
            #     nn.LeakyReLU(inplace=True),
            # )
        else:
            raise NotImplementedError(
                f"got landmarks backbone '{name}', expected one of {self._possible_landmarks_backbones}"
            )
        landmarks_backbone.apply(self.initialize_weights_xavier)
        landmarks_embedder = lambda landmarks: landmarks_backbone(landmarks)
        return landmarks_features_size, landmarks_backbone, landmarks_embedder

    def _parse_merging_method(
        self, name
    ) -> Tuple[nn.Module, nn.Module, nn.Module, Callable]:
        image_features_embedder, landmarks_features_embedder = (
            nn.Identity(),
            nn.Identity(),
        )
        if name == "concatenate":
            cls_head = nn.Sequential(
                nn.Linear(
                    self.image_features_size + self.landmarks_features_size,
                    512 * 4,
                ),
                nn.LayerNorm(512 * 4),
                nn.LeakyReLU(inplace=True),
                nn.Dropout(self.linear_dropout_p),
                nn.Linear(
                    512 * 4,
                    self.num_classes,
                ),
            )
            cls_head.apply(self.initialize_weights_xavier)
            # cls_head = nn.Sequential(
            #     nn.Linear(
            #         self.image_features_size + self.landmarks_features_size,
            #         self.num_classes,
            #     ),
            # )

            def classify(image_features, landmarks_features):
                assert any([image_features is not None, landmarks_features is not None])
                tensors = []
                if self.use_horizontal_images or self.use_vertical_images:
                    assert image_features is not None, "there are no image features"
                    assert (
                        image_features.shape[1] == self.image_features_size
                    ), f"got {image_features.shape[1]=} with {self.image_backbone_name=}, expected {self.image_features_size=}"
                    tensors.append(image_features)
                if self.use_horizontal_landmarks or self.use_vertical_landmarks:
                    assert (
                        landmarks_features is not None
                    ), "there are no landmarks features"
                    assert (
                        landmarks_features.shape[1] == self.landmarks_features_size
                    ), f"got {landmarks_features.shape[1]=}, expected {self.landmarks_features_size=}"
                    tensors.append(landmarks_features)
                tensors = torch.concatenate(tensors, dim=1)
                assert tensors.shape[1] == (
                    self.image_features_size + self.landmarks_features_size
                ), f"got {tensors.shape[1]=}, expected {self.image_features_size + self.landmarks_features_size=}. Combination is {self.image_backbone_name=} and {self.landmarks_backbone_name=}. Params are {self.use_horizontal_images=}, {self.use_vertical_images=}, {self.use_horizontal_landmarks=}, {self.use_vertical_landmarks=}"
                assert tensors.shape[0] > 0, f"got {tensors.shape=}"
                return cls_head(tensors)

        else:
            raise NotImplementedError(
                f"merging method {self._merging_method} not implemented"
            )
        
        return image_features_embedder, landmarks_features_embedder, cls_head, classify

    @staticmethod
    def _plot_bw_image(img):
        import matplotlib.pyplot as plt

        plt.imshow(img, cmap="gray")
        plt.axis("off")
        plt.show()

    @staticmethod
    def initialize_weights_xavier(module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def on_save_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        if self.train_image_backbone:
            return
        keys_to_remove = [
            k
            for k in checkpoint["state_dict"]
            if k.startswith("_image_backbone.")  # only this module
        ]
        for k in keys_to_remove:
            del checkpoint["state_dict"][k]

    def on_load_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        (
            self.image_features_size,
            self.img_size,
            self.img_mean,
            self.img_std,
            self._image_backbone,
            self.image_embedder,
        ) = self._parse_image_backbone(name=self.image_backbone_name)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            # [
            #     # {"params": self.adapter.parameters(), "lr": 5e-4}, # self.adapter is included here
            #     {
            #         "params": [
            #             p
            #             for module in [
            #                 self.adapter,
            #                 self.landmarks_backbone,
            #                 # self.landmarks_embedder,
            #                 # self.image_features_embedder,
            #                 # self.landmarks_features_embedder,
            #                 self.cls_head,
            #             ]
            #             for p in module.parameters()
            #         ],
            #         "lr": self.lr,
            #     },
            # ],
            self.parameters(),
            lr=self.lr,
        )
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.num_epochs * 2, eta_min=self.lr * 0.01
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": lr_scheduler,
        }

    def optimizer_zero_grad(self, epoch, batch_idx, optimizer):
        optimizer.zero_grad(set_to_none=True)

    @abstractmethod
    def forward(
        self,
        landmarks_horizontal: torch.Tensor = None,
        landmarks_vertical: torch.Tensor = None,
        image_horizontal: torch.Tensor = None,
        image_vertical: torch.Tensor = None,
        **kwargs,
    ):
        outs = {
            "imgs_embs": None,
            "landmarks_embs": None,
        }
        # images branch
        if self.use_horizontal_images:
            assert image_horizontal is not None, "there are no horizontal images"
        if self.use_vertical_images:
            assert image_vertical is not None, "there are no vertical images"
        if self.use_horizontal_images or self.use_vertical_images:
            if self.training:
                image_transforms = self.image_transforms_train
            else:
                image_transforms = self.image_transforms_val_test
            imgs = []
            if self.use_horizontal_images and image_horizontal is not None:
                imgs.append(image_horizontal.to(self.device))
            if self.use_vertical_images and image_vertical is not None:
                imgs.append(image_vertical.to(self.device))
            imgs = torch.cat(imgs, dim=1).float()
            outs["imgs_embs"] = self.image_embedder(
                T.Normalize(
                    mean=self.img_mean,
                    std=self.img_std,
                )(self.adapter(image_transforms(imgs)))
            )  # type: ignore

        # landmarks branch
        if self.use_horizontal_landmarks:
            assert landmarks_horizontal is not None, "there are no horizontal landmarks"
        if self.use_vertical_landmarks:
            assert landmarks_vertical is not None, "there are no vertical landmarks"
        if self.use_horizontal_landmarks or self.use_vertical_landmarks:
            landmarks = []
            if self.use_horizontal_landmarks and landmarks_horizontal is not None:
                landmarks.append(landmarks_horizontal.to(self.device))
            if self.use_vertical_landmarks and landmarks_vertical is not None:
                landmarks.append(landmarks_vertical.to(self.device))
            landmarks = torch.cat(landmarks, dim=1).float()
            outs["landmarks_embs"] = self.landmarks_embedder(landmarks)

        # merging branch
        outs["cls_logits"] = self.classify(
            image_features=outs["imgs_embs"], landmarks_features=outs["landmarks_embs"]
        )
        return outs

    def step(self, batch, phase, batch_idx=None):
        # initializes self.epoch_metrics
        if phase not in self.epoch_metrics:
            self.epoch_metrics[phase] = {}
        for key in ["cls_labels", "cls_logits", "loss", "time", "num_params"]:
            if not key in self.epoch_metrics[phase]:
                self.epoch_metrics[phase][key] = []

        # Track time, params, MACs, and FLOPs only for the first batch of the first epoch
        # if batch_idx == 0 and self.current_epoch == 0 and phase in {"val", "test"}:
        start_time = time.time()
        # adapts to single image and landmarks input
        if "image" in batch:
            if self.use_horizontal_images:
                batch["image_horizontal"] = batch["image"]
            elif self.use_vertical_images:
                batch["image_vertical"] = batch["image"]
        if "landmarks" in batch:
            if self.use_horizontal_landmarks:
                batch["landmarks_horizontal"] = batch["landmarks"]
            elif self.use_vertical_landmarks:
                batch["landmarks_vertical"] = batch["landmarks"]
        outs = self(**batch)
        elapsed = time.time() - start_time
        self.epoch_metrics[phase]["time"].append(
            torch.as_tensor([elapsed], dtype=torch.float32)
        )

        # number of parameters
        num_params = sum(p.numel() for p in self.parameters())
        self.epoch_metrics[phase]["num_params"] = torch.as_tensor(
            [num_params], dtype=torch.float32
        )

        # MACs and FLOPs
        # TODO
        # else:
        #     outs = self(**batch)

        batch["label"] = batch["label"].to(self.device)
        outs["loss"] = F.cross_entropy(
            input=outs["cls_logits"], target=batch["label"], label_smoothing=0.1
        )
        # for key in outs:
        #     if key in ["loss"]:
        #         continue
        #     if isinstance(outs[key], torch.Tensor) and "cpu" not in str(outs[key].device):
        #         outs[key] = outs[key].detach().cpu()
        self.epoch_metrics[phase]["cls_labels"].append(batch["label"].cpu())
        self.epoch_metrics[phase]["cls_logits"].append(
            outs["cls_logits"].detach().cpu()
        )
        self.epoch_metrics[phase]["loss"].append(outs["loss"].detach().cpu())

        return outs

    def on_epoch_end(self, phase):
        logits_stacked = torch.cat(self.epoch_metrics[phase]["cls_logits"], dim=0)
        labels_stacked = torch.cat(self.epoch_metrics[phase]["cls_labels"], dim=0)
        metrics = {
            "cls_loss": torch.stack(self.epoch_metrics[phase]["loss"], dim=0).mean(),
            "cls_prec": torchmetrics.functional.precision(
                preds=logits_stacked,
                target=labels_stacked,
                task="multiclass",
                num_classes=self.num_classes,
                average="macro",
            ),
            "cls_rec": torchmetrics.functional.recall(
                preds=logits_stacked,
                target=labels_stacked,
                task="multiclass",
                num_classes=self.num_classes,
                average="macro",
            ),
            "cls_acc": torchmetrics.functional.accuracy(
                preds=logits_stacked,
                target=labels_stacked,
                task="multiclass",
                num_classes=self.num_classes,
                average="macro",
            ),
            "cls_f1": torchmetrics.functional.f1_score(
                preds=logits_stacked,
                target=labels_stacked,
                task="multiclass",
                num_classes=self.num_classes,
                average="macro",
            ),
            "time": sum(self.epoch_metrics[phase]["time"])
            / len(self.epoch_metrics[phase]["time"]),
            "num_params": self.epoch_metrics[phase]["num_params"],
        }
        for metric_name, metric_value in metrics.items():
            self.log(
                name=f"{metric_name}_{phase}",
                value=metric_value,
                prog_bar=True if "f1" in metric_name else False,
            )
        del self.epoch_metrics[phase]

    def training_step(self, batch, batch_idx):
        outs = self.step(batch, batch_idx=batch_idx, phase="train")
        return outs

    def validation_step(self, batch, batch_idx):
        with torch.no_grad():
            outs = self.step(batch, batch_idx=batch_idx, phase="val")
        return outs

    def test_step(self, batch, batch_idx):
        with torch.no_grad():
            outs = self.step(batch, batch_idx=batch_idx, phase="test")
        return outs

    def on_train_epoch_end(self):
        self.on_epoch_end(phase="train")

    def on_validation_epoch_end(self):
        self.on_epoch_end(phase="val")

    def on_test_epoch_end(self):
        self.on_epoch_end(phase="test")


if __name__ == "__main__":
    # sets the seed
    set_global_seed(42)
    # define the dataset
    dataset = MotionLeap2Dataset(dataset_path="../../datasets/ml2hp")

    # define params for the tests
    params = {
        "batch_size": 16,
        # "device": "cuda" if torch.cuda.is_available() else "cpu",
        "device": "cpu",
    }

    for (
        landmarks_backbone_name,
        image_backbone_name,
    ) in tqdm(
        list(
            itertools.product(
                HandGestureRecognitionModel._possible_landmarks_backbones,
                HandGestureRecognitionModel._possible_image_backbones,
            )
        ),
        desc="trying all backbones combinations",
    ):
        # skips incompatible combinations
        if not (landmarks_backbone_name or image_backbone_name):
            continue
        dataloader = DataLoader(
            dataset=dataset,
            batch_size=params["batch_size"],
            shuffle=False,
            num_workers=1,
        )
        batch = dataloader.__iter__().__next__()
        model = HandGestureRecognitionModel(
            use_horizontal_images=True,
            use_vertical_images=True,
            use_horizontal_landmarks=True,
            use_vertical_landmarks=True,
            num_landmarks=dataset.num_landmarks,
            img_size=dataset.img_size,
            num_labels=dataset.num_labels,
            image_backbone_name=image_backbone_name,
            landmarks_backbone_name=landmarks_backbone_name,
            **params,
        ).to(params["device"])
        model.step(batch=batch, phase="train")
        del model
        gc.collect()
        if "cuda" in params["device"]:
            torch.cuda.empty_cache()
        time.sleep(1)

    for (
        use_horizontal_images,
        use_vertical_images,
        use_horizontal_landmarks,
        use_vertical_landmarks,
    ) in tqdm(
        list(
            itertools.product(
                [True, False], [True, False], [True, False], [True, False]
            )
        ),
        desc="trying all modes combinations",
    ):
        # skips incompatible combinations
        if (
            (
                not any(
                    [
                        use_horizontal_images,
                        use_vertical_images,
                        use_horizontal_landmarks,
                        use_vertical_landmarks,
                    ]
                )
            )
            or (
                (use_horizontal_images or use_vertical_images)
                and (image_backbone_name is None)
            )
            or (
                (use_horizontal_landmarks or use_vertical_landmarks)
                and (landmarks_backbone_name is None)
            )
        ):
            continue
        dataset.set_mode(
            return_horizontal_images=use_horizontal_images,
            return_vertical_images=use_vertical_images,
            return_horizontal_landmarks=use_horizontal_landmarks,
            return_vertical_landmarks=use_vertical_landmarks,
        )
        dataloader = DataLoader(
            dataset=dataset,
            batch_size=params["batch_size"],
            shuffle=False,
            num_workers=1,
        )
        batch = dataloader.__iter__().__next__()
        model = HandGestureRecognitionModel(
            use_horizontal_images=dataset.return_horizontal_images,
            use_vertical_images=dataset.return_vertical_images,
            use_horizontal_landmarks=dataset.return_horizontal_landmarks,
            use_vertical_landmarks=dataset.return_vertical_landmarks,
            num_landmarks=dataset.num_landmarks,
            img_size=dataset.img_size,
            num_labels=dataset.num_labels,
            image_backbone_name=(
                "clip-b" if use_horizontal_images or use_vertical_images else None
            ),
            landmarks_backbone_name=(
                "mlp" if use_horizontal_landmarks or use_vertical_landmarks else None
            ),
            **params,
        ).to(params["device"])
        model.step(batch=batch, phase="train")
        del model
        gc.collect()
        if "cuda" in params["device"]:
            torch.cuda.empty_cache()
        time.sleep(1)
    print("All tests passed successfully!")
