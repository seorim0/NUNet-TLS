import torch
from torch.utils.data import Dataset, DataLoader
import config as cfg
from tools import scan_directory, find_pair, addr2wav


def create_dataloader(mode):
    if mode == 'train':
        return DataLoader(
            dataset=Wave_Dataset(mode),
            batch_size=cfg.batch,
            shuffle=True,
            num_workers=0,
            pin_memory=True,
            drop_last=True,
            sampler=None
        )
    elif mode == 'valid':
        return DataLoader(
            dataset=Wave_Dataset(mode),
            batch_size=cfg.batch, shuffle=False, num_workers=0
        )


class Wave_Dataset(Dataset):
    def __init__(self, mode):
        # load data
        self.mode = mode

        if mode == 'train':
            print('<Training dataset>')
            print('Load the data...')
            # load the wav addr
            self.noisy_dirs = scan_directory(cfg.noisy_dirs_for_train)
            self.clean_dirs = find_pair(self.noisy_dirs)

        elif mode == 'valid':
            print('<Validation dataset>')
            print('Load the data...')
            # load the wav addr
            self.noisy_dirs = scan_directory(cfg.noisy_dirs_for_valid)
            self.clean_dirs = find_pair(self.noisy_dirs)

    def __len__(self):
        return len(self.noisy_dirs)

    def __getitem__(self, idx):
        # read the wav
        inputs = addr2wav(self.noisy_dirs[idx])
        targets = addr2wav(self.clean_dirs[idx])

        # transform to torch from numpy
        inputs = torch.from_numpy(inputs)
        targets = torch.from_numpy(targets)

        # (-1, 1)
        inputs = torch.clamp_(inputs, -1, 1)
        targets = torch.clamp_(targets, -1, 1)

        return inputs, targets
