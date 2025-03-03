import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import Dataset, random_split
class Dataset(Dataset):
    def __init__(self, lr_dir, hr_dir, scale_factor=4, patch_size=96):
        self.lr_dir = lr_dir
        self.hr_dir = hr_dir
        self.scale = scale_factor
        self.patch_size = patch_size
        self.filenames = os.listdir(lr_dir)

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        lr_path = os.path.join(self.lr_dir, self.filenames[idx])
        hr_path = os.path.join(self.hr_dir, self.filenames[idx])

        # 读取图像并转换为 RGB
        lr_img = cv2.imread(lr_path)
        hr_img = cv2.imread(hr_path)
        lr_img = cv2.cvtColor(lr_img, cv2.COLOR_BGR2RGB)
        hr_img = cv2.cvtColor(hr_img, cv2.COLOR_BGR2RGB)

        # # 随机裁剪
        # h, w = lr_img.shape[:2]
        # x = np.random.randint(0, w - self.patch_size)
        # y = np.random.randint(0, h - self.patch_size)
        # lr_patch = lr_img[y:y+self.patch_size, x:x+self.patch_size]
        # hr_patch = hr_img[y*self.scale:(y+self.patch_size)*self.scale,
        #                   x*self.scale:(x+self.patch_size)*self.scale]

        # 归一化并转为 Tensor
        lr_patch = lr_img.astype(np.float32) / 255.0
        hr_patch = hr_img.astype(np.float32) / 255.0
        lr_patch = torch.from_numpy(lr_patch).permute(2, 0, 1)
        hr_patch = torch.from_numpy(hr_patch).permute(2, 0, 1)

        return lr_patch, hr_patch

    def get_train_val_datasets(lr_dir, hr_dir, val_split=0.2):
        full_dataset = Dataset(lr_dir, hr_dir)
        val_size = int(len(full_dataset) * val_split)
        train_size = len(full_dataset) - val_size
        train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
        return train_dataset, val_dataset