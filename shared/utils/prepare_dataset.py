import os
from pathlib import Path


def get_split(path):
    dataset_path = Path(path)

    train_file_names = []
    val_file_names = []
    test_file_names = []

    valid_exts = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'}

    def _collect_files(dir_path):
        if not os.path.exists(dir_path):
            return []
        files = []
        for root, _, filenames in os.walk(dir_path, followlinks=True):
            for f in filenames:
                if Path(f).suffix.lower() in valid_exts:
                    files.append(Path(root) / f)
        return sorted(files)

    if dataset_path.exists():
        sets = os.listdir(dataset_path)
        for set_dir in sets:
            set_lower = set_dir.lower()
            img_dir = dataset_path / set_dir / 'images'
            if set_lower == "train":
                train_file_names = _collect_files(img_dir)
            elif set_lower == "test":
                test_file_names = _collect_files(img_dir)
            elif set_lower in ["val", "validation"]:
                val_file_names = _collect_files(img_dir)

    # Fallback for Kaggle input structures if dataset_path symlinks failed
    if len(train_file_names) == 0 and os.path.exists('/kaggle/input'):
        for root, _, filenames in os.walk('/kaggle/input', followlinks=True):
            if os.path.basename(root).lower() == 'images':
                parent_dir = os.path.basename(os.path.dirname(root)).lower()
                collected = sorted([Path(root) / f for f in filenames if Path(f).suffix.lower() in valid_exts])
                if 'train' in parent_dir and len(train_file_names) == 0:
                    train_file_names = collected
                elif ('val' in parent_dir or 'validation' in parent_dir) and len(val_file_names) == 0:
                    val_file_names = collected
                elif 'test' in parent_dir and len(test_file_names) == 0:
                    test_file_names = collected

    return train_file_names, test_file_names, val_file_names


