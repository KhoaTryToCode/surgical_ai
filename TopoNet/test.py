import argparse
import os
import cv2
import numpy as np
np.Inf = np.inf
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from utils.metrics import evaluation
from utils import prepare_dataset
from utils.dataset import LandmarkDataset, save_img
from models.TopoNet import TopoNet
from medpy.metric import assd



def main(args):
    train_file, test_file, val_file = prepare_dataset.get_split(args.data_path)

    if args.split.lower() == 'val':
        eval_files = val_file
    elif args.split.lower() == 'train':
        eval_files = train_file
    else:
        eval_files = test_file

    test_dataset = LandmarkDataset(eval_files, transform=None, mode='test')
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = TopoNet(1024, 1024, depth_path=args.depth_path).to(device)
    model_checkpoint = torch.load(args.model_path, map_location=device)
    model.load_state_dict(model_checkpoint)
    model.eval()

    validation_IOU = []
    mDice = []
    mAssd = []

    with torch.no_grad():
        for index, (X_batch, depth, y_batch, name) in tqdm(enumerate(test_loader)):

            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            depth = depth.to(device)

            output, _ = model(X_batch)

            output = torch.argmax(torch.softmax(output, dim=1), dim=1).detach().cpu().numpy()

            y_batch = torch.argmax(y_batch, dim=1)

            tmp2 = y_batch.detach().cpu().numpy()
            tmp = output
            tmp = tmp[0]
            tmp2 = tmp2[0]

            pred = np.array([tmp == i for i in range(4)]).astype(np.uint8)
            gt = np.array([tmp2 == i for i in range(4)]).astype(np.uint8)

            iou, dice = evaluation(pred[1:].flatten(), gt[1:].flatten())

            validation_IOU.append(iou)
            mDice.append(dice)

            if np.count_nonzero(pred[1:]) == 0 or np.count_nonzero(gt[1:]) == 0:
                mAssd.append(80)          # 1. Store penalty score (80) for this empty image
            else:
                assd_value = assd(pred[1:], gt[1:])
                mAssd.append(assd_value)  # 2. Store real calculated score (assd_value) for this image

            toprint = save_img(tmp)
            img_stem = Path(name[0] if isinstance(name, (list, tuple)) else name).stem
            save_file_name = f"{img_stem}.png"
            cv2.imwrite(os.path.join(args.save_path, save_file_name), toprint)


    print(f"[{args.split.upper()} SPLIT] Mean IoU:", np.mean(validation_IOU))
    print(f"[{args.split.upper()} SPLIT] Mean Dice:", np.mean(mDice))
    print(f"[{args.split.upper()} SPLIT] Mean ASSD:", np.mean(mAssd))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', default="best_model_path.pth")
    parser.add_argument('--data_path', default='data/')
    parser.add_argument('--depth_path', default='depth_anything_v2_vitb.pth')
    parser.add_argument('--save_path', default='test_results/')
    parser.add_argument('--split', default='test', choices=['test', 'val', 'train'], help='Which dataset split to evaluate (test, val, or train)')
    args = parser.parse_args()

    os.makedirs(args.save_path, exist_ok=True)

    main(args=args)


