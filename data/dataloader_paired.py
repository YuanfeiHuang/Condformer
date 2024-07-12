import random
import cv2, time
import torch
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
            self.img_clean, self.img_noisy, self.img_sigma = [], [], []
            with tqdm(total=len(self.filepath_clean)) as pbar:
                for idx in range(len(self.filepath_clean)):
                    img_clean, img_noisy = cv2.cvtColor(cv2.imread(self.filepath_clean[idx]), cv2.COLOR_BGR2RGB), \
                                           cv2.cvtColor(cv2.imread(self.filepath_noisy[idx]), cv2.COLOR_BGR2RGB)
                    if self.filepath_sigma[idx] is not None:
                        img_sigma = np.float32(np.load(self.filepath_sigma[idx]))
                    else:
                        img_sigma = None

                    self.img_clean.append(img_clean)
                    self.img_noisy.append(img_noisy)
                    self.img_sigma.append(img_sigma)
                    time.sleep(0.01)
                    pbar.update(1)
                    pbar.set_postfix(name=self.filepath_clean[idx].split('/')[-1],
                                     sigma='{:.4f}*I+{:.4f}'
                                     .format(self.img_sigma[idx][0], self.img_sigma[idx][1])
                                     if img_sigma is not None else '')


    def _set_filesystem(self):
        self.filepath_clean, self.filepath_noisy, self.filepath_sigma = np.array([]), np.array([]), np.array([])
        for idx_dataset in range(len(self.args.data_train)):
            if self.args.n_train[idx_dataset] > 0:
                path = self.args.dir_data + 'Train/' + self.args.data_train[idx_dataset]
                names_clean = os.listdir(os.path.join(path, 'Clean'))
                names_noisy = os.listdir(os.path.join(path, 'Noisy'))
                names_clean.sort()
                names_noisy.sort()
                filepath_clean, filepath_noisy = np.array([]), np.array([])
                filepath_sigma = np.array([])

                if os.path.exists(os.path.join(path, 'Sigma')):
                    names_sigma = os.listdir(os.path.join(path, 'Sigma'))
                    names_sigma.sort()
                    exist_sigma = True
                else:
                    exist_sigma = False

                for idx_image in range(len(names_clean)):
                    filepath_clean = np.append(filepath_clean,
                                               os.path.join(path + '/Clean', names_clean[idx_image]))
                    filepath_noisy = np.append(filepath_noisy,
                                               os.path.join(path + '/Noisy', names_noisy[idx_image]))

                    if exist_sigma:
                        filepath_sigma = np.append(filepath_sigma,
                                                   os.path.join(path + '/Sigma', names_sigma[idx_image]))
                    else:
                        filepath_sigma = np.append(filepath_sigma, None)

                data_length = len(filepath_clean)
                idx = np.arange(0, data_length)
                if self.args.n_train[idx_dataset] < data_length:
                    if self.args.shuffle:
                        idx = np.random.choice(idx, size=self.args.n_train[idx_dataset])
                    else:
                        idx = np.arange(0, self.args.n_train[idx_dataset])

                self.filepath_clean = np.append(self.filepath_clean, filepath_clean[idx])
                self.filepath_noisy = np.append(self.filepath_noisy, filepath_noisy[idx])
                self.filepath_sigma = np.append(self.filepath_sigma, filepath_sigma[idx])

    def __getitem__(self, idx):
        if self.args.store_in_ram:
            idx = idx % len(self.img_sigma)
            img_clean, img_noisy, img_sigma = self.img_clean[idx], self.img_noisy[idx], self.img_sigma[idx]
        else:
            raise InterruptedError

        img_clean, img_noisy = common.set_channel([img_clean, img_noisy], self.args.n_colors)
        img_clean, img_noisy = common.get_patch([img_clean, img_noisy], self.args.patch_size, 1)
        flag_aug = random.randint(0, 7)
        img_clean, img_noisy = common.augment(img_clean, flag_aug), common.augment(img_noisy, flag_aug)
        img_clean = common.np2Tensor(img_clean, self.args.value_range)
        img_noisy = common.np2Tensor(img_noisy, self.args.value_range)
        if img_sigma is not None:
            img_sigma = torch.from_numpy(img_sigma)
        else:
            img_sigma = torch.zeros((2))

        return img_clean, img_noisy, img_sigma

    def __len__(self):
        return self.args.iter_epoch * self.args.batch_size

