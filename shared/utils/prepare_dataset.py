import os
import sys
import argparse
import shutil
from pathlib import Path

def download_via_kagglehub(split_name: str) -> str:
    """
    Downloads Kaggle dataset split via official kagglehub Python library.
    split_name: 'train', 'val', or 'test'
    """
    try:
        import kagglehub
        handle = f"khoatrytopublish/l3d-{split_name}"
        print(f"📥 Downloading Kaggle dataset '{handle}' via kagglehub...")
        download_path = kagglehub.dataset_download(handle)
        print(f"✅ Successfully downloaded '{handle}' to: '{download_path}'")
        return download_path
    except Exception as e:
        print(f"⚠️ Could not download dataset via kagglehub ({e}).")
        return None

def setup_dataset(target_dir: str = "/content/L3D") -> str:
    """
    Sets up dataset symlinks for train, val, and test splits into target_dir:
      target_dir/train/images -> .../Train/images
      target_dir/train/labels -> .../Train/labels
      target_dir/val/images   -> .../Val/images
      target_dir/val/labels   -> .../Val/labels
      target_dir/test/images  -> .../Test/images
      target_dir/test/labels  -> .../Test/labels
    If dataset directories are not present locally or in /kaggle/input, it downloads them from Kaggle.
    """
    target_path = Path(target_dir).resolve()
    target_path.mkdir(parents=True, exist_ok=True)
    print(f"📦 Setting up dataset directory structure in '{target_path}'...")

    splits = ["train", "val", "test"]
    subdirs = ["images", "labels", "depth_anything_v2"]

    # Known candidate paths across Kaggle notebook, local, and custom environments
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
        ("train", "depth_anything_v2"): [
            "/kaggle/input/datasets/khoale05/l3d-depth/L3D/train/depth_anything_v2",
            "/kaggle/input/datasets/khoale05/l3d-depth/L3D/Train/depth_anything_v2",
            "/kaggle/input/l3d-depth/L3D/train/depth_anything_v2",
            "/kaggle/input/l3d-depth/L3D/Train/depth_anything_v2",
            "/kaggle/input/l3d-depth/train/depth_anything_v2",
            "/kaggle/input/l3d-depth/Train/depth_anything_v2",
            "data/laparoscopic_liver/train/depth_anything_v2",
            "data/depth_maps/train/depth_anything_v2",
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
        ("val", "depth_anything_v2"): [
            "/kaggle/input/datasets/khoale05/l3d-depth/L3D/val/depth_anything_v2",
            "/kaggle/input/datasets/khoale05/l3d-depth/L3D/Val/depth_anything_v2",
            "/kaggle/input/l3d-depth/L3D/val/depth_anything_v2",
            "/kaggle/input/l3d-depth/L3D/Val/depth_anything_v2",
            "/kaggle/input/l3d-depth/val/depth_anything_v2",
            "/kaggle/input/l3d-depth/Val/depth_anything_v2",
            "data/laparoscopic_liver/val/depth_anything_v2",
            "data/depth_maps/val/depth_anything_v2",
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
        ],
        ("test", "depth_anything_v2"): [
            "/kaggle/input/datasets/khoale05/l3d-depth/L3D/test/depth_anything_v2",
            "/kaggle/input/datasets/khoale05/l3d-depth/L3D/Test/depth_anything_v2",
            "/kaggle/input/l3d-depth/L3D/test/depth_anything_v2",
            "/kaggle/input/l3d-depth/L3D/Test/depth_anything_v2",
            "/kaggle/input/l3d-depth/test/depth_anything_v2",
            "/kaggle/input/l3d-depth/Test/depth_anything_v2",
            "data/laparoscopic_liver/test/depth_anything_v2",
            "data/depth_maps/test/depth_anything_v2",
        ]
    }

    # First pass: try existing local paths
    downloaded_cache = {}
    for split in splits:
        for sub in subdirs:
            target_sub = target_path / split / sub
            target_sub.parent.mkdir(parents=True, exist_ok=True)

            # Handle dangling/broken symlink
            if target_sub.is_symlink() and not target_sub.exists():
                print(f"🧹 Removing broken symlink: '{target_sub}'")
                target_sub.unlink()

            if target_sub.exists():
                continue

            src_matched = None
            for candidate in known_mappings.get((split, sub), []):
                if os.path.exists(candidate):
                    src_matched = candidate
                    break

            # Dynamic search in /kaggle/input if not in known_mappings
            if not src_matched and os.path.exists("/kaggle/input"):
                for root, dirs, _ in os.walk("/kaggle/input", followlinks=True):
                    parts_lower = [p.lower() for p in Path(root).parts]
                    if split in parts_lower and os.path.basename(root).lower() == sub.lower():
                        src_matched = root
                        break

            # If not found locally and it's an original split (images/labels), download split via kagglehub
            if not src_matched and sub in ["images", "labels"]:
                if split not in downloaded_cache:
                    downloaded_cache[split] = download_via_kagglehub(split)
                
                dl_base = downloaded_cache[split]
                if dl_base and os.path.exists(dl_base):
                    # Search inside downloaded kagglehub folder for matching images/labels
                    for root, dirs, _ in os.walk(dl_base, followlinks=True):
                        if os.path.basename(root).lower() == sub:
                            src_matched = root
                            break

            if src_matched:
                try:
                    os.symlink(src_matched, target_sub)
                    print(f"🔗 Created symlink: '{target_sub}' ──► '{src_matched}'")
                except Exception as e:
                    # Fallback to copytree if symlink is not permitted
                    try:
                        shutil.copytree(src_matched, target_sub)
                        print(f"📁 Copied folder: '{target_sub}'")
                    except Exception as copy_err:
                        print(f"⚠️ Could not link/copy '{target_sub}': {copy_err}")
            else:
                print(f"⚠️ Source directory for {split}/{sub} not found.")

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
    parser = argparse.ArgumentParser(description="Prepare Surgical Dataset symlinks & download from Kaggle")
    parser.add_argument("--target_dir", type=str, default="/content/L3D", help="Target path to create symlinks for")
    args = parser.parse_args()

    setup_dataset(target_dir=args.target_dir)
