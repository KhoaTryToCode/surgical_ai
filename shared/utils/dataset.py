import os
import json
import xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np
import cv2
from torch.utils.data import Dataset
from torchvision import transforms as T
from skimage.metrics import structural_similarity as SSIM



class LandmarkDataset(Dataset):
    def __init__(self, file_names, transform=None, mode='train'):
        self.file_names = file_names

        if transform:
            self.transform = transform
        else:
            to_tensor = T.ToTensor()
            self.transform = lambda x, y, z: (to_tensor(x), to_tensor(y), to_tensor(z))

        self.mode = mode

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, idx):
        img_file_name = self.file_names[idx]
        image = load_image(img_file_name)
        mask = load_mask(img_file_name)
        depth = np.zeros((1024, 1024)).astype(np.uint8)
        image, mask, depth = self.transform(image, mask.transpose(1, 2, 0), depth)

        if self.mode == 'train':
            return image, depth, mask, str(img_file_name)
        else:
            return image, depth, mask, str(img_file_name)


def load_image(path):
    img = cv2.imread(str(path))
    img = cv2.resize(img, (1024, 1024))

    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def load_depth(path):
    img = cv2.imread(str(path).replace('images', 'depth').replace('jpg', 'png'), 0)
    img = cv2.resize(img, (1024, 1024))

    return img.astype(np.uint8)


def load_mask(path):
    # Load ground truth from JSON labels
    base_path = str(path).replace('images', 'labels')
    json_path = os.path.splitext(base_path)[0] + '.json'

    if os.path.exists(json_path):
        mask = load_json(json_path)
    else:
        mask = np.zeros((1024, 1024), dtype=np.uint8)

    mask = cv2.resize(mask, (1024, 1024), interpolation=cv2.INTER_NEAREST)
    masks = np.zeros(shape=(4, 1024, 1024), dtype=np.uint8)
    masks[0][mask == 0] = 255
    masks[1][mask == 1] = 255
    masks[2][mask == 2] = 255
    masks[3][mask == 3] = 255

    return masks


def load_xml(path):
    img = np.zeros((1080, 1920), dtype=np.uint8)
    tree = ET.parse(path)
    root = tree.getroot()

    for contour in root.findall('contour'):
        ctype = contour.find('contourType').text

        if ctype == 'Ridge':
            color = 1
        elif ctype == 'Silhouette':
            color = 2
        else:
            color = 3

        x_coords = [float(x) for x in contour.find('imagePoints/x').text.split(',')]
        y_coords = [float(y) for y in contour.find('imagePoints/y').text.split(',')]

        for i in range(1, len(x_coords)):
            pt1 = tuple(map(int, [x_coords[i-1], y_coords[i-1]]))
            pt2 = tuple(map(int, [x_coords[i], y_coords[i]]))
            cv2.line(img, pt1, pt2, color, 35)

    return img


def load_json(path):
    with open(path, 'r') as f:
        data = json.load(f)

    # Use dimensions from JSON metadata (confirmed always present and correct)
    img_h = data.get('imageHeight', 1080)
    img_w = data.get('imageWidth', 1920)

    image = np.zeros((img_h, img_w), dtype=np.uint8)

    for shape in data['shapes']:
        points = shape['points']
        label = str(shape.get('label', '')).lower()

        if label.startswith('r') or 'ridge' in label or 'rigde' in label:
            color = 1
        elif label.startswith('s') or 'sil' in label:
            color = 2
        elif label.startswith('l') or 'lig' in label:
            color = 3
        else:
            color = 0

        for i in range(1, len(points)):
            pt1 = tuple(map(int, points[i - 1]))
            pt2 = tuple(map(int, points[i]))

            cv2.line(image, pt1, pt2, color, 35)

    return image





def save_img(img):
    color_map = {0: (0, 0, 0),  # Black
                 1: (0, 0, 255),  # Red
                 2: (0, 255, 0),  # Green
                 3: (255, 0, 0)}  # Blue

    # Get the height, width and channels
    height, width = img.shape
    channels = 3

    # Create a blank RGB image
    image = np.zeros((height, width, channels), np.uint8)

    # Map each value in the array to a color
    for i in range(height):
        for j in range(width):
            value = img[i, j]
            image[i, j] = color_map[value]
    return image


def ssim(img1, img2):
    img1 = cv2.resize(img1, (224, 224))
    img2 = cv2.resize(img2, (224, 224))
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    ss = SSIM(img1, img2)
    return ss
