import random

import cv2, time

from data import common
import numpy as np
import torch.utils.data as data
import os
from tqdm import tqdm


class dataloader(data.Dataset):
    def __init__(self, args):
        self.args = args
        self._set_filesystem()

        if self.args.store_in_ram:
            self.img_clean = []
            with tqdm(total=len(self.filepath_clean)) as pbar:
                for idx in range(len(self.filepath_clean)):
                    img_clean = cv2.cvtColor(cv2.imread(self.filepath_clean[idx]), cv2.COLOR_BGR2RGB)

                    h, w, c = img_clean.shape
                    if min(h, w) < self.args.patch_size:
                        img_clean = cv2.copyMakeBorder(img_clean, 0, max(self.args.patch_size - h, 0), 0,
                                                       max(self.args.patch_size - w, 0), cv2.BORDER_REFLECT)
                    self.img_clean.append(img_clean)

                    time.sleep(0.01)
                    pbar.update(1)
                    pbar.set_postfix(name=self.filepath_clean[idx].split('/')[-1])

    def _set_filesystem(self):
        self.filepath_clean = np.array([])
        for idx_dataset in range(len(self.args.data_train)):
            if self.args.n_train[idx_dataset] > 0:
                path = self.args.dir_data + 'Train/' + self.args.data_train[idx_dataset]
                names_clean = os.listdir(os.path.join(path, 'HR'))

                names_clean.sort()
                filepath_clean = np.array([])

                for idx_image in range(len(names_clean)):
                    filepath_clean = np.append(filepath_clean, os.path.join(path + '/HR', names_clean[idx_image]))

                data_length = len(filepath_clean)
                idx = np.arange(0, data_length)
                if self.args.n_train[idx_dataset] < data_length:
                    if self.args.shuffle:
                        idx = np.random.choice(idx, size=self.args.n_train[idx_dataset])
                    else:
                        idx = np.arange(0, self.args.n_train[idx_dataset])

                self.filepath_clean = np.append(self.filepath_clean, filepath_clean[idx])

    def __getitem__(self, idx):

        if self.args.store_in_ram:
            idx = idx % len(self.img_clean)
            img_clean = self.img_clean[idx]
        else:
            raise InterruptedError

        img_clean = common.set_channel(img_clean, self.args.n_colors)
        img_clean = common.get_patch(img_clean, self.args.patch_size, 1)
        flag_aug = random.randint(0, 7)
        img_clean = common.augment(img_clean, flag_aug)
        img_clean = common.np2Tensor(img_clean, self.args.value_range)

        return img_clean

    def __len__(self):
        return self.args.iter_epoch * self.args.batch_size

