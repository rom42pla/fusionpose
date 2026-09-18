# Hand Gesture Recognition
Repository for the paper [xyz](), published at [xyz]().

## Pre-processing

If it does not already exist, a folder `_preprocessed_dataset` inside the current directory is created.
This folder will contain the preprocessed landmarks, for faster use during training/inference.

## Pre-requisites

### Dependencies
The required dependendencies to run the code are defined in the `requirements.txt` file. 
For simplicity, use `conda` to create a virtual environment:

```bash
conda create --name fusionpose python=3.9
conda activate fusionpose
```

Once in the `fusionpose` environment, install the dependencies:

```bash
conda install pip
pip install -r requirements.txt
```

### Datasets

Make sure to put the `_mmhgdhgr_preprocessed_landmarks`,  `_tiny_hgr_preprocessed_landmarks`, and  `_ml2hp_preprocessed_landmarks` into this folder, if you have it.
If you need to generate those preprocessed landmarks, you have to put the `ml2hp` and `tiny_hgr` folders into `../../datasets`.

Make also sure to put the `mmhgdhgr` dataset into `../../datasets`.

You can [download the preprocessed datasets from here](https://drive.google.com/drive/folders/1FuDFZ6jN_PLjUluN3ogTgVV_AunZ7vgF?usp=drive_link), and [the datasets from here](https://drive.google.com/drive/folders/1rUuR0Dhluwjc3jyPczjkVFUSK0lwKwAq?usp=drive_link).

### Checkpoints

Make sure to create a directory like `./checkpoints` and put the `_mmhgdhgr_results`,  `_tiny_hgr_results` there.

You can [download the checkpoints from here](https://drive.google.com/drive/folders/1JfO-QBPXbrRFR1PmQMi8XbOWKkjswDCi?usp=drive_link).

## Run a test

You can use the `do_validation.py` script, passing the path of the checkpoint folder that you want to test. For example:

```bash
python do_validation.py checkpoints/tiny_hgr_results/tiny_hgr_simple_convnextv2-t_mlp_h-images_h-landmarks-20250724-1426/
```