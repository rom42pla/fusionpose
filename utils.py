import random
from typing import Any, Dict, List

import numpy as np
import pandas as pd

import torch


def find_samples_of_subject(dataset, subject_id):
    indices = []
    for i_sample, sample in enumerate(dataset.samples):
        if sample["subject_id"] == subject_id:
            indices.append(i_sample)
    assert all([dataset.samples[i]["subject_id"] == subject_id for i in indices])
    return indices


def set_global_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def get_device_from_string(device: str):
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    elif device in {"cuda", "gpu"}:
        return "cuda"
    elif device == "cpu":
        return "cpu"
    raise NotImplementedError(f"unrecognized device {device}")


def get_simple_runs(dataset, train_perc=0.8) -> List[Dict[str, Any]]:
    assert 0 < train_perc < 1
    indices_shuffled = np.arange(len(dataset))
    np.random.shuffle(indices_shuffled)
    pivot = int(len(dataset) * train_perc)
    runs = [
        {"train_idx": indices_shuffled[:pivot], "val_idx": indices_shuffled[pivot:]}
    ]
    assert set(runs[-1]["train_idx"]) & set(runs[-1]["val_idx"]) == set()
    return runs


def get_k_fold_runs(k: int, dataset) -> List[List[int]]:
    assert isinstance(k, int) and k > 1
    indices_shuffled = np.arange(len(dataset))
    np.random.shuffle(indices_shuffled)
    folds = np.array_split(indices_shuffled, indices_or_sections=k)
    folds = [fold.tolist() for fold in folds]
    all_indices = np.concatenate(folds)
    assert len(all_indices) == len(np.unique(all_indices))
    assert np.array_equal(np.sort(all_indices), np.arange(len(dataset)))

    # builds the runs dict
    runs = []
    for i_fold in range(len(folds)):
        runs.append(
            {
                "train_idx": [
                    i
                    for i_fold_inner, fold in enumerate(folds)
                    for i in fold
                    if i_fold_inner != i_fold
                ],
                "val_idx": [
                    i
                    for i_fold_inner, fold in enumerate(folds)
                    for i in fold
                    if i_fold_inner == i_fold
                ],
            }
        )
        assert set(runs[-1]["train_idx"]) & set(runs[-1]["val_idx"]) == set()
    assert len(runs) == k
    return runs

def get_train_test_splits(dataset, limit_subjects: int = None) -> List[List[int]]:
    subject_ids_to_use = dataset._get_subject_ids()
    indices_per_subject = (
        dataset.get_indices_per_subject()
    )  # {1: [1, 2, 3], 2: [4, 5, 6]}
    if limit_subjects is not None:
        assert isinstance(
            limit_subjects, int
        ), f"got {limit_subjects} ({type(limit_subjects)})"
        assert 2 <= limit_subjects <= len(subject_ids_to_use), f"got {limit_subjects}"
        subject_ids_to_use = subject_ids_to_use[:limit_subjects]

    # builds the runs dict
    run = {
        "train_idx": [],
        "val_idx": [],
    }
    for i_sample, sample in enumerate(dataset.samples):
        if sample["subject_id"] not in subject_ids_to_use:
            continue
        if sample["split"] == "train":
            key = "train_idx"
        elif sample["split"] == "test":
            key = "val_idx"
        else:
            raise ValueError(f"unrecognized split {sample['split']} in sample {sample}")
        run[key].append(i_sample)
    runs = [run]
    return runs

def get_optimistic_splits(dataset, limit_subjects: int = None) -> List[List[int]]:
    subject_ids_to_use = sorted(dataset._get_subject_ids())
    assert len(subject_ids_to_use) == 40, f"got {len(subject_ids_to_use)}: {subject_ids_to_use}"

    # selects train, val and test subjects
    subject_ids_train = set(subject_ids_to_use[:25])
    subject_ids_val = set(subject_ids_to_use[25:30])

    # builds the runs dict
    run = {
        "train_idx": [],
        "val_idx": [],
        "test_idx": [],
    }
    for i_sample, sample in enumerate(dataset.samples):
        if sample["subject_id"] in subject_ids_train:
            run["train_idx"].append(i_sample)
        elif sample["subject_id"] in subject_ids_val:
            run["val_idx"].append(i_sample)
        else:
            run["test_idx"].append(i_sample)
    runs = [run]
    return runs



    # for subject in indices_per_subject:
    #     runs.append(
    #         {
    #             "subject_id": str(subject).zfill(3),
    #             "train_idx": [
    #                 i
    #                 for subject_id_inner, indices in indices_per_subject.items()
    #                 for i in indices
    #                 if subject_id_inner != subject
    #             ],
    #             "val_idx": [
    #                 i
    #                 for subject_id_inner, indices in indices_per_subject.items()
    #                 for i in indices
    #                 if subject_id_inner == subject
    #             ],
    #         }
    #     )
    #     assert set(runs[-1]["train_idx"]) & set(runs[-1]["val_idx"]) == set()
    # assert len(runs) == (limit_subjects if limit_subjects is not None else len(dataset.subject_ids))
    return runs

def get_loso_runs(dataset, limit_subjects: int) -> List[List[int]]:
    subject_ids_to_use = dataset._get_subject_ids()
    indices_per_subject = (
        dataset.get_indices_per_subject()
    )  # {1: [1, 2, 3], 2: [4, 5, 6]}
    if limit_subjects is not None:
        assert isinstance(
            limit_subjects, int
        ), f"got {limit_subjects} ({type(limit_subjects)})"
        assert 2 <= limit_subjects <= len(subject_ids_to_use), f"got {limit_subjects}"
        subject_ids_to_use = subject_ids_to_use[:limit_subjects]
        indices_per_subject = {
            subject_id: i_samples
            for subject_id, i_samples in indices_per_subject.items()
            if subject_id in subject_ids_to_use
        }
    # all_indices = [
    #     i for subject in indices_per_subject for i in indices_per_subject[subject]
    # ]
    # assert len(all_indices) == len(dataset), f"{len(all_indices)} != {len(dataset)}"
    # assert sorted(all_indices) == sorted(list(range(len(dataset))))

    # builds the runs dict
    runs = []
    for subject in indices_per_subject:
        runs.append(
            {
                "subject_id": str(subject).zfill(3),
                "train_idx": [
                    i
                    for subject_id_inner, indices in indices_per_subject.items()
                    for i in indices
                    if subject_id_inner != subject
                ],
                "val_idx": [
                    i
                    for subject_id_inner, indices in indices_per_subject.items()
                    for i in indices
                    if subject_id_inner == subject
                ],
            }
        )
        assert set(runs[-1]["train_idx"]) & set(runs[-1]["val_idx"]) == set()
    assert len(runs) == (limit_subjects if limit_subjects is not None else len(dataset.subject_ids))
    return runs
