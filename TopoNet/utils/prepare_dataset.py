import os
from pathlib import Path


def get_split(path):
    dataset_path = Path(path)

    train_file_names = []
    val_file_names = []
    test_file_names = []

    if dataset_path.exists():
        sets = os.listdir(dataset_path)
        for set_dir in sets:
            set_lower = set_dir.lower()
            if set_lower == "train":
                train_file_names = sorted(list((dataset_path / set_dir / 'images').glob('*')))
            elif set_lower == "test":
                test_file_names = sorted(list((dataset_path / set_dir / 'images').glob('*')))
            elif set_lower in ["val", "validation"]:
                val_file_names = sorted(list((dataset_path / set_dir / 'images').glob('*')))

    return train_file_names, test_file_names, val_file_names


