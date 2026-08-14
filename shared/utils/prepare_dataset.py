import os
import sys
import argparse
from pathlib import Path

def setup_dataset(target_dir: str = "/kaggle/working/L3D") -> str:
    """
    Sets up symlinks for train, val, and test splits into target_dir (/kaggle/working/L3D):
      /kaggle/working/L3D/train/images -> .../Train/images
      /kaggle/working/L3D/train/labels -> .../Train/labels
      /kaggle/working/L3D/val/images   -> .../Val/images
      /kaggle/working/L3D/val/labels   -> .../Val/labels
      /kaggle/working/L3D/test/images  -> .../Test/images
      /kaggle/working/L3D/test/labels  -> .../Test/labels
    """
    target_path = Path(target_dir).resolve()
    target_path.mkdir(parents=True, exist_ok=True)
    print(f"📦 Setting up dataset symlinks in '{target_path}'...")

    splits = ["train", "val", "test"]
    subdirs = ["images", "labels"]

    # Candidate source map for Kaggle inputs
    known_mappings = {
        ("train", "images"): [
            "/kaggle/input/datasets/khoatrytopublish/l3d-train/Train/images",
            "/kaggle/input/l3d-train/Train/images",
            "/kaggle/input/laparoscopic-liver-landmarks/train/images",
            "data/laparoscopic_liver/train/images",
        ],
        ("train", "labels"): [
            "/kaggle/input/datasets/khoatrytopublish/l3d-train/Train/labels",
            "/kaggle/input/l3d-train/Train/labels",
            "/kaggle/input/laparoscopic-liver-landmarks/train/labels",
            "data/laparoscopic_liver/train/labels",
        ],
        ("val", "images"): [
            "/kaggle/input/datasets/khoatrytopublish/l3d-val/Val/images",
            "/kaggle/input/l3d-val/Val/images",
            "/kaggle/input/laparoscopic-liver-landmarks/val/images",
            "data/laparoscopic_liver/val/images",
        ],
        ("val", "labels"): [
            "/kaggle/input/datasets/khoatrytopublish/l3d-val/Val/labels",
            "/kaggle/input/l3d-val/Val/labels",
            "/kaggle/input/laparoscopic-liver-landmarks/val/labels",
            "data/laparoscopic_liver/val/labels",
        ],
        ("test", "images"): [
            "/kaggle/input/datasets/khoatrytopublish/l3d-test/Test/images",
            "/kaggle/input/l3d-test/Test/images",
            "/kaggle/input/laparoscopic-liver-landmarks/test/images",
            "data/laparoscopic_liver/test/images",
        ],
        ("test", "labels"): [
            "/kaggle/input/datasets/khoatrytopublish/l3d-test/Test/labels",
            "/kaggle/input/l3d-test/Test/labels",
            "/kaggle/input/laparoscopic-liver-landmarks/test/labels",
            "data/laparoscopic_liver/test/labels",
        ]
    }

    for split in splits:
        for sub in subdirs:
            target_sub = target_path / split / sub
            target_sub.parent.mkdir(parents=True, exist_ok=True)
            
            if target_sub.exists() or target_sub.is_symlink():
                continue

            src_matched = None
            for candidate in known_mappings.get((split, sub), []):
                if os.path.exists(candidate):
                    src_matched = candidate
                    break

            # Fallback dynamic search if specific path is not found directly
            if not src_matched and os.path.exists("/kaggle/input"):
                for root, dirs, _ in os.walk("/kaggle/input", followlinks=True):
                    root_lower = root.lower()
                    if sub in root_lower and split in root_lower:
                        src_matched = root
                        break

            if src_matched:
                try:
                    os.symlink(src_matched, target_sub)
                    print(f"🔗 Created symlink: '{target_sub}' ──► '{src_matched}'")
                except Exception as e:
                    print(f"⚠️ Failed to create symlink '{target_sub}': {e}")
            else:
                print(f"⚠️ Source directory for {split}/{sub} not found under /kaggle/input.")

    print(f"✅ Symlink setup completed in '{target_path}'.")
    return str(target_path)

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

    return train_file_names, test_file_names, val_file_names

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare Surgical Dataset symlinks")
    parser.add_argument("--target_dir", type=str, default="/kaggle/working/L3D", help="Target path to create symlinks for")
    args = parser.parse_args()

    setup_dataset(target_dir=args.target_dir)
