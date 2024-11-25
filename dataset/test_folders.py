import os
import torch
import torch.utils.data as data
import numpy as np
from imageio import imread
from path import Path
import random
from PIL import Image
from utils.util import normalize_image_numpy
from skimage.transform import resize


def load_as_float(path):
    img = resize(imread(path).astype(np.float32), (320, 320))
    valid_img = np.where(img > 1, img, 0 * img)
    t = valid_img[:, :, 0:3]
    return t

def load_depth_as_float(path):
    t = resize(imread(path).astype(np.float32)[0], (320, 320))
    return t

class TestFolder(torch.utils.data.Dataset):
    def __init__(self, root, sequence_length=3, transform=None, skip_frames=5):
        np.random.seed(0)
        random.seed(0)
        self.root = Path(root)
        self.img_list_path = [f for f in sorted(os.listdir(self.root)) if os.path.isfile(os.path.join(self.root, f))]
        # print(self.img_list_path)
        self.depth_root = Path("/mnt/disk1/toky/Dataset/endoslam_unity_colon/Depth/")
        self.transform = transform
        self.k = skip_frames
        self.crawl_folders(sequence_length)

    def crawl_folders(self, sequence_length):
        # print(self.root)
        sequence_set = []
        imgs = []
        depths = []
        img_names = []

        for line in self.img_list_path:
            imgs.append(self.root / line.strip())
            img_names.append(line.strip())

        for img_file in imgs:
            img_name = img_file.name
            # img_number = img_name.split('_')[1]
            depth_file_name = f"aov_{img_name}"
            depth_file_path = self.depth_root / depth_file_name
            if depth_file_path.exists():
                depths.append(depth_file_path)
            else:
                print(f"Depth file {depth_file_path} not found.")
                depths.append(None)

        if len(imgs) < sequence_length:
            print("len(imgs) < sequence_length")
            return

        for i in range(len(imgs)):
            sample = {'img': imgs[i], 'depth': depths[i] if depths[i] is not None else 'placeholder_path', 'img_name': img_names[i]}
            sequence_set.append(sample)
        self.samples = sequence_set

    def __getitem__(self, index):
        sample = self.samples[index]
        tgt_img = load_as_float(sample['img'])
        if sample['depth'] == 'placeholder_path':
            tgt_depth = np.zeros((320, 320), dtype=np.float32)
        else:
            tgt_depth = load_depth_as_float(sample['depth'])
        return torch.tensor(normalize_image_numpy(tgt_img)).permute(2, 0, 1), torch.tensor(normalize_image_numpy(tgt_depth)).unsqueeze(0), sample['img_name']


    def __len__(self):
        return len(self.samples)