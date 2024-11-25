import torch
import torch.utils.data as data
import numpy as np
from imageio import imread
from path import Path
import random

from utils.util import normalize_image, normalize_image_numpy


def load_as_float(path):
    img = imread(path).astype(np.float32)
    valid_img = np.where(img > 1, img, 0 * img)
    # print("valid_img", valid_img)
    return valid_img


def load_depth_as_float(path):
    return imread(path).astype(np.float32)[:, :, 3]


class TrainFolder(data.Dataset):
    def __init__(self, root, sequence_length=3, transform=None, skip_frames=5):
        np.random.seed(0)
        random.seed(0)

        self.root = Path(root)
        self.img_list_path = self.root / 'train.txt'
        self.depth_list_path = self.root / 'depth.txt'
        self.transform = transform
        self.k = skip_frames
        self.crawl_folders(sequence_length)

    def crawl_folders(self, sequence_length):
        print(self.root)
        sequence_set = []
        imgs = []
        depths = []
        with open(self.img_list_path, encoding='utf-8') as file:
            imgs_content = file.readlines()
        for line in imgs_content:
            imgs.append(self.root / line[0:-1])
        with open(self.depth_list_path, encoding='utf-8') as file:
            depth_content = file.readlines()
        for line in depth_content:
            depths.append(self.root / line[0:-1])

        if len(imgs) < sequence_length:
            print("len(imgs) < sequence_length")
            return
        for i in range(len(imgs)):
            sample = {'img': imgs[i], 'depth': depths[i]}
            sequence_set.append(sample)
        self.samples = sequence_set

    def __getitem__(self, index):
        sample = self.samples[index]
        tgt_img = load_as_float(sample['img'])
        tgt_depth = load_depth_as_float(sample['depth'])
        return torch.tensor(normalize_image_numpy(tgt_img)).permute(2, 0, 1), torch.tensor(
            normalize_image_numpy(tgt_depth)).unsqueeze(0)

    def __len__(self):
        return len(self.samples)
