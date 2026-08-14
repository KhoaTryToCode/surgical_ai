import os
import sys
import argparse
import zipfile
import shutil
from pathlib import Path

def setup_dataset(target_dir: str, source_dir: str = None) -> str:
    """
    Sets up dataset symlinks or extracts zip files into target_dir (e.g., /kaggle/working/L3D).
    Searches /kaggle/input and local data directories automatically if source_dir is not provided.
    """
    target_path = Path(target_dir).resolve()
    target_path.mkdir(parents=True, exist_ok=True)

    print(f"📦 Setting up dataset target directory: '{target_path}'")

    # Locate source directory
    if source_dir and os.path.exists(source_dir):
        src_path = Path(source_dir).resolve()
    else:
        candidates = [
            Path("/kaggle/input/laparoscopic-liver-landmarks"),
            Path("/kaggle/input/laparoscopic-liver"),
            Path("/kaggle/input/l3d"),
            Path("./data/laparoscopic_liver"),
            Path("./data"),
        ]
        src_path = None
        for c in candidates:
            if c.exists():
                src_path = c
                break

        if src_path is None and os.path.exists("/kaggle/input"):
            # Search /kaggle/input for any dataset folder containing images or labels or zips
            for root, dirs, files in os.walk("/kaggle/input"):
                if "images" in dirs or "labels" in dirs or "annotations" in dirs:
                    src_path = Path(root)
                    break
                for f in files:
                    if f.endswith(".zip"):
                        src_path = Path(root)
                        break
                if src_path:
                    break

    if src_path is None:
        print(f"⚠️ Source dataset directory not found in /kaggle/input or local data/. Created empty '{target_path}'.")
        return str(target_path)

    print(f"🔍 Found dataset source at: '{src_path}'")

    # Handle ZIP archives if present
    zip_files = list(src_path.glob("*.zip"))
    if zip_files:
        for zf in zip_files:
            print(f"⚡ Extracting '{zf.name}' into '{target_path}'...")
            with zipfile.ZipFile(zf, 'r') as zip_ref:
                zip_ref.extractall(target_path)

    # Symlink subdirectories (images, labels, annotations, depth, train, val, test)
    subdirs_to_link = ["images", "labels", "annotations", "depth", "train", "val", "test"]
    
    # Check top-level subdirectories in src_path
    for item in src_path.iterdir():
        if item.is_dir():
            target_sub = target_path / item.name
            if not target_sub.exists():
                try:
                    os.symlink(item, target_sub)
                    print(f"🔗 Created symlink: '{target_sub}' ──► '{item}'")
                except Exception as e:
                    # Fallback to copytree if symlink fails
                    try:
                        shutil.copytree(item, target_sub)
                        print(f"📁 Copied directory: '{target_sub}'")
                    except Exception as copy_err:
                        print(f"⚠️ Could not link/copy '{item.name}': {copy_err}")
        elif item.is_file() and not item.name.endswith(".zip"):
            target_file = target_path / item.name
            if not target_file.exists():
                try:
                    os.symlink(item, target_file)
                except Exception:
                    shutil.copy2(item, target_file)

    print(f"✅ Dataset preparation completed successfully for '{target_path}'.")
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

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare Surgical Dataset symlinks / target directory")
    parser.add_argument("--target_dir", type=str, default="/kaggle/working/L3D", help="Target dataset path to set up")
    parser.add_argument("--source_dir", type=str, default=None, help="Source dataset path (optional)")
    args = parser.parse_args()

    setup_dataset(target_dir=args.target_dir, source_dir=args.source_dir)
