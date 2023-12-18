"""
Tools for training, evaluating, ...
everything you need
"""
import os
import time
from pesq import pesq
import torch
import torch.nn as nn
import torch.nn.functional as F
from pystoi import stoi
from tensorboardX import SummaryWriter
import matplotlib
import soundfile
from scipy.signal import get_window
import numpy as np
import config as cfg


#######################################################################
#######################################################################
#                             Data Load                               #
#######################################################################
#######################################################################
def scan_directory(dir_name):
    if os.path.isdir(dir_name) is False:
        print("[Error] There is no directory '%s'." % dir_name)
        exit()

    addrs = []
    for subdir, dirs, files in os.walk(dir_name):
        for file in files:
            if file.endswith(".wav"):
                filepath = subdir + file
                addrs.append(filepath)
    return addrs


# TIMIT
def find_pair(noisy_dirs):
    clean_dirs = []
    for i in range(len(noisy_dirs)):
        addrs = noisy_dirs[i]
        if addrs.endswith(".wav"):
            addr_noisy = str(addrs)
            addr_clean = str(addrs).replace('noisy', 'clean')

            # 1st '_'
            idx_1st = addr_noisy.find('_')
            # 2nd '_'
            idx_2nd = addr_noisy[idx_1st + 1:].find('_')
            # 3rd '_'
            idx_3rd = addr_noisy[idx_1st + 1 + idx_2nd + 1:].find('_')
            # 4th '_'
            idx_4th = addr_noisy[idx_1st + 1 + idx_2nd + 1 + idx_3rd + 1:].find('_')

            addr_clean = addr_clean[ : idx_1st + idx_2nd + idx_3rd + idx_4th + 3] + '.wav'
            clean_dirs.append(addr_clean)
    return clean_dirs


def addr2wav(addr):
    wav, fs = soundfile.read(addr)
    return wav


#######################################################################
#######################################################################
#                         Feature extraction                          #
#######################################################################
#######################################################################
# this is from conv_stft https://github.com/huyanxin/DeepComplexCRN
def init_kernels(win_len, fft_len, win_type=None, invers=False):
    if win_type == 'None' or win_type is None:
        window = np.ones(win_len)
    else:
        window = get_window(win_type, win_len, fftbins=True)  # **0.5

    N = fft_len
    fourier_basis = np.fft.rfft(np.eye(N))[:win_len]
    real_kernel = np.real(fourier_basis)
    imag_kernel = np.imag(fourier_basis)
    kernel = np.concatenate([real_kernel, imag_kernel], 1).T

    if invers:
        kernel = np.linalg.pinv(kernel).T

    kernel = kernel * window
    kernel = kernel[:, None, :]
    return torch.from_numpy(kernel.astype(np.float32)), torch.from_numpy(window[None, :, None].astype(np.float32))


class ConvSTFT(nn.Module):

    def __init__(self, win_len, win_inc, fft_len=None, win_type='hamming', feature_type='real'):
        super(ConvSTFT, self).__init__()

        if fft_len is None:
            self.fft_len = np.int(2 ** np.ceil(np.log2(win_len)))
        else:
            self.fft_len = fft_len

        kernel, _ = init_kernels(win_len, self.fft_len, win_type)
        self.register_buffer('weight', kernel)
        self.feature_type = feature_type
        self.stride = win_inc
        self.win_len = win_len
        self.dim = self.fft_len

    def forward(self, inputs):
        if inputs.dim() == 2:
            inputs = torch.unsqueeze(inputs, 1)
        inputs = F.pad(inputs, [self.win_len - self.stride, self.win_len - self.stride])
        outputs = F.conv1d(inputs, self.weight, stride=self.stride)

        if self.feature_type == 'complex':
            return outputs
        else:
            dim = self.dim // 2 + 1
            real = outputs[:, :dim, :]
            imag = outputs[:, dim:, :]
            mags = torch.sqrt(real ** 2 + imag ** 2)
            phase = torch.atan2(imag, real)
            return mags, phase  # , real, imag


class ConviSTFT(nn.Module):

    def __init__(self, win_len, win_inc, fft_len=None, win_type='hamming', feature_type='real'):
        super(ConviSTFT, self).__init__()
        if fft_len is None:
            self.fft_len = np.int(2 ** np.ceil(np.log2(win_len)))
        else:
            self.fft_len = fft_len
        kernel, window = init_kernels(win_len, self.fft_len, win_type, invers=True)
        self.register_buffer('weight', kernel)
        self.feature_type = feature_type
        self.win_type = win_type
        self.win_len = win_len
        self.stride = win_inc
        self.dim = self.fft_len
        self.register_buffer('window', window)
        self.register_buffer('enframe', torch.eye(win_len)[:, None, :])

    def forward(self, inputs, phase=None):
        """
        inputs : [B, N+2, T] (complex spec) or [B, N//2+1, T] (mags)
        phase: [B, N//2+1, T] (if not none)
        """

        if phase is not None:
            real = inputs * torch.cos(phase)
            imag = inputs * torch.sin(phase)
            inputs = torch.cat([real, imag], 1)

        outputs = F.conv_transpose1d(inputs, self.weight, stride=self.stride)

        # this is from torch-stft: https://github.com/pseeth/torch-stft
        t = self.window.repeat(1, 1, inputs.size(-1)) ** 2
        coff = F.conv_transpose1d(t, self.enframe, stride=self.stride)

        outputs = outputs / (coff + 1e-8)

        outputs = outputs[..., self.win_len - self.stride:-(self.win_len - self.stride)]

        return outputs


#######################################################################
#######################################################################
#                             Get Score                               #
#######################################################################
#######################################################################
def cal_pesq(dirty_wavs, clean_wavs):
    avg_pesq_score = 0
    for i in range(len(dirty_wavs)):
        pesq_score = pesq(cfg.FS, clean_wavs[i], dirty_wavs[i], "wb")
        avg_pesq_score += pesq_score
    avg_pesq_score /= len(dirty_wavs)
    return avg_pesq_score


def cal_stoi(dirty_wavs, clean_wavs):
    avg_stoi_score = 0
    for i in range(len(dirty_wavs)):
        stoi_score = stoi(clean_wavs[i], dirty_wavs[i], cfg.FS, extended=False)
        avg_stoi_score += stoi_score
    avg_stoi_score /= len(dirty_wavs)
    return avg_stoi_score


def snr(dirty_wav, clean_wav, eps=1e-8):
    mean_signal = np.mean(clean_wav)
    signal_diff = clean_wav - mean_signal
    var_signal = np.sum(np.mean(signal_diff ** 2))  # variance of orignal data

    noise = dirty_wav - clean_wav
    mean_noise = np.mean(noise)
    noise_diff = noise - mean_noise
    var_noise = np.sum(np.mean(noise_diff ** 2))  # variance of noise

    if var_noise == 0:
        snr_score = 100  # clean
    else:
        snr_score = (np.log10(var_signal / var_noise + eps)) * 10
    return snr_score


def cal_snr(dirty_wavs, clean_wavs):
    avg_snr_score = 0
    for i in range(len(dirty_wavs)):
        snr_score = snr(dirty_wavs[i], clean_wavs[i])
        avg_snr_score += snr_score
    avg_snr_score /= len(dirty_wavs)
    return avg_snr_score


#######################################################################
#######################################################################
#                          Loss functions                             #
#######################################################################
#######################################################################
L1loss = nn.L1Loss()


#######################################################################
#######################################################################
#                             Interface                               #
#######################################################################
#######################################################################
#
class Bar(object):
    def __init__(self, dataloader):
        if not hasattr(dataloader, 'dataset'):
            raise ValueError('Attribute `dataset` not exists in dataloder.')
        if not hasattr(dataloader, 'batch_size'):
            raise ValueError('Attribute `batch_size` not exists in dataloder.')

        self.dataloader = dataloader
        self.iterator = iter(dataloader)
        self.dataset = dataloader.dataset
        self.batch_size = dataloader.batch_size
        self._idx = 0
        self._batch_idx = 0
        self._time = []
        self._DISPLAY_LENGTH = 50

    def __len__(self):
        return len(self.dataloader)

    def __iter__(self):
        return self

    def __next__(self):
        if len(self._time) < 2:
            self._time.append(time.time())

        self._batch_idx += self.batch_size
        if self._batch_idx > len(self.dataset):
            self._batch_idx = len(self.dataset)

        try:
            batch = next(self.iterator)
            self._display()
        except StopIteration:
            raise StopIteration()

        self._idx += 1
        if self._idx >= len(self.dataloader):
            self._reset()

        return batch

    def _display(self):
        if len(self._time) > 1:
            t = (self._time[-1] - self._time[-2])
            eta = t * (len(self.dataloader) - self._idx)
        else:
            eta = 0

        rate = self._idx / len(self.dataloader)
        len_bar = int(rate * self._DISPLAY_LENGTH)
        bar = ('=' * len_bar + '>').ljust(self._DISPLAY_LENGTH, '.')
        idx = str(self._batch_idx).rjust(len(str(len(self.dataset))), ' ')

        tmpl = '\r{}/{}: [{}] - ETA {:.1f}s'.format(
            idx,
            len(self.dataset),
            bar,
            eta
        )
        print(tmpl, end='')
        if self._batch_idx == len(self.dataset):
            print()

    def _reset(self):
        self._idx = 0
        self._batch_idx = 0
        self._time = []


# calculate the size of total network
def cal_total_params(our_model):
    total_parameters = 0
    for variable in our_model.parameters():
        shape = variable.size()
        variable_parameters = 1
        for dim in shape:
            variable_parameters *= dim
        total_parameters += variable_parameters

    return total_parameters


# write the configuration(settings)
def write_status(dir2log):
    fp = open(dir2log + '/config.txt', 'a')
    fp.write('- Feature\n')
    fp.write('FS: {} | WIN_LEN: {} | HOP_LEN: {} | FFT_LEN: {}\n\n'
             .format(cfg.FS, cfg.WIN_LEN, cfg.HOP_LEN, cfg.FFT_LEN))

    fp.write('- Train data directory: \n')
    fp.write('Clean: {} | Noisy: {}\n'.format(cfg.clean_dirs_for_train, cfg.noisy_dirs_for_train))
    fp.write('- Valid data directory: \n')
    fp.write('Clean: {} | Noisy: {}\n'.format(cfg.clean_dirs_for_valid, cfg.noisy_dirs_for_valid))

    fp.write('- Hyperparameters\n')
    fp.write('Learning rate: {} | Batch: {}\n\n'.format(cfg.learning_rate, cfg.batch))
    fp.close()


#######################################################################
#######################################################################
#                             Tensorboard                             #
#######################################################################
#######################################################################
class Writer(SummaryWriter):
    def __init__(self, logdir):
        super(Writer, self).__init__(logdir)
        # mask real/ imag
        cmap_custom = {
            'red': ((0.0, 0.0, 0.0),
                    (1 / 63, 0.0, 0.0),
                    (2 / 63, 0.0, 0.0),
                    (3 / 63, 0.0, 0.0),
                    (4 / 63, 0.0, 0.0),
                    (5 / 63, 0.0, 0.0),
                    (6 / 63, 0.0, 0.0),
                    (7 / 63, 0.0, 0.0),
                    (8 / 63, 0.0, 0.0),
                    (9 / 63, 0.0, 0.0),
                    (10 / 63, 0.0, 0.0),
                    (11 / 63, 0.0, 0.0),
                    (12 / 63, 0.0, 0.0),
                    (13 / 63, 0.0, 0.0),
                    (14 / 63, 0.0, 0.0),
                    (15 / 63, 0.0, 0.0),
                    (16 / 63, 0.0, 0.0),
                    (17 / 63, 0.0, 0.0),
                    (18 / 63, 0.0, 0.0),
                    (19 / 63, 0.0, 0.0),
                    (20 / 63, 0.0, 0.0),
                    (21 / 63, 0.0, 0.0),
                    (22 / 63, 0.0, 0.0),
                    (23 / 63, 0.0, 0.0),
                    (24 / 63, 0.5625, 0.5625),
                    (25 / 63, 0.6250, 0.6250),
                    (26 / 63, 0.6875, 0.6875),
                    (27 / 63, 0.7500, 0.7500),
                    (28 / 63, 0.8125, 0.8125),
                    (29 / 63, 0.8750, 0.8750),
                    (30 / 63, 0.9375, 0.9375),
                    (31 / 63, 1.0, 1.0),
                    (32 / 63, 1.0, 1.0),
                    (33 / 63, 1.0, 1.0),
                    (34 / 63, 1.0, 1.0),
                    (35 / 63, 1.0, 1.0),
                    (36 / 63, 1.0, 1.0),
                    (37 / 63, 1.0, 1.0),
                    (38 / 63, 1.0, 1.0),
                    (39 / 63, 1.0, 1.0),
                    (40 / 63, 1.0, 1.0),
                    (41 / 63, 1.0, 1.0),
                    (42 / 63, 1.0, 1.0),
                    (43 / 63, 1.0, 1.0),
                    (44 / 63, 1.0, 1.0),
                    (45 / 63, 1.0, 1.0),
                    (46 / 63, 1.0, 1.0),
                    (47 / 63, 1.0, 1.0),
                    (48 / 63, 1.0, 1.0),
                    (49 / 63, 1.0, 1.0),
                    (50 / 63, 1.0, 1.0),
                    (51 / 63, 1.0, 1.0),
                    (52 / 63, 1.0, 1.0),
                    (53 / 63, 1.0, 1.0),
                    (54 / 63, 1.0, 1.0),
                    (55 / 63, 1.0, 1.0),
                    (56 / 63, 0.9375, 0.9375),
                    (57 / 63, 0.8750, 0.8750),
                    (58 / 63, 0.8125, 0.8125),
                    (59 / 63, 0.7500, 0.7500),
                    (60 / 63, 0.6875, 0.6875),
                    (61 / 63, 0.6250, 0.6250),
                    (62 / 63, 0.5625, 0.5625),
                    (63 / 63, 0.5000, 0.5000)),
            'green': ((0.0, 0.0, 0.0),
                      (1 / 63, 0.0, 0.0),
                      (2 / 63, 0.0, 0.0),
                      (3 / 63, 0.0, 0.0),
                      (4 / 63, 0.0, 0.0),
                      (5 / 63, 0.0, 0.0),
                      (6 / 63, 0.0, 0.0),
                      (7 / 63, 0.0, 0.0),
                      (8 / 63, 0.0625, 0.0625),
                      (9 / 63, 0.1250, 0.1250),
                      (10 / 63, 0.1875, 0.1875),
                      (11 / 63, 0.2500, 0.2500),
                      (12 / 63, 0.3125, 0.3125),
                      (13 / 63, 0.3750, 0.3750),
                      (14 / 63, 0.4375, 0.4375),
                      (15 / 63, 0.5000, 0.5000),
                      (16 / 63, 0.5625, 0.5625),
                      (17 / 63, 0.6250, 0.6250),
                      (18 / 63, 0.6875, 0.6875),
                      (19 / 63, 0.7500, 0.7500),
                      (20 / 63, 0.8125, 0.8125),
                      (21 / 63, 0.8750, 0.8750),
                      (22 / 63, 0.9375, 0.9375),
                      (23 / 63, 1.0, 1.0),
                      (24 / 63, 1.0, 1.0),
                      (25 / 63, 1.0, 1.0),
                      (26 / 63, 1.0, 1.0),
                      (27 / 63, 1.0, 1.0),
                      (28 / 63, 1.0, 1.0),
                      (29 / 63, 1.0, 1.0),
                      (30 / 63, 1.0, 1.0),
                      (31 / 63, 1.0, 1.0),
                      (32 / 63, 1.0, 1.0),
                      (33 / 63, 1.0, 1.0),
                      (34 / 63, 1.0, 1.0),
                      (35 / 63, 1.0, 1.0),
                      (36 / 63, 1.0, 1.0),
                      (37 / 63, 1.0, 1.0),
                      (38 / 63, 1.0, 1.0),
                      (39 / 63, 1.0, 1.0),
                      (40 / 63, 0.9375, 0.9375),
                      (41 / 63, 0.8750, 0.8750),
                      (42 / 63, 0.8125, 0.8125),
                      (43 / 63, 0.7500, 0.7500),
                      (44 / 63, 0.6875, 0.6875),
                      (45 / 63, 0.6250, 0.6250),
                      (46 / 63, 0.5625, 0.5625),
                      (47 / 63, 0.5000, 0.5000),
                      (48 / 63, 0.4375, 0.4375),
                      (49 / 63, 0.3750, 0.3750),
                      (50 / 63, 0.3125, 0.3125),
                      (51 / 63, 0.2500, 0.2500),
                      (52 / 63, 0.1875, 0.1875),
                      (53 / 63, 0.1250, 0.1250),
                      (54 / 63, 0.0625, 0.0625),
                      (55 / 63, 0.0, 0.0),
                      (56 / 63, 0.0, 0.0),
                      (57 / 63, 0.0, 0.0),
                      (58 / 63, 0.0, 0.0),
                      (59 / 63, 0.0, 0.0),
                      (60 / 63, 0.0, 0.0),
                      (61 / 63, 0.0, 0.0),
                      (62 / 63, 0.0, 0.0),
                      (63 / 63, 0.0, 0.0)),
            'blue': ((0.0, 0.5625, 0.5625),
                     (1 / 63, 0.6250, 0.6250),
                     (2 / 63, 0.6875, 0.6875),
                     (3 / 63, 0.7500, 0.7500),
                     (4 / 63, 0.8125, 0.8125),
                     (5 / 63, 0.8750, 0.8750),
                     (6 / 63, 0.9375, 0.9375),
                     (7 / 63, 1.0, 1.0),
                     (8 / 63, 1.0, 1.0),
                     (9 / 63, 1.0, 1.0),
                     (10 / 63, 1.0, 1.0),
                     (11 / 63, 1.0, 1.0),
                     (12 / 63, 1.0, 1.0),
                     (13 / 63, 1.0, 1.0),
                     (14 / 63, 1.0, 1.0),
                     (15 / 63, 1.0, 1.0),
                     (16 / 63, 1.0, 1.0),
                     (17 / 63, 1.0, 1.0),
                     (18 / 63, 1.0, 1.0),
                     (19 / 63, 1.0, 1.0),
                     (20 / 63, 1.0, 1.0),
                     (21 / 63, 1.0, 1.0),
                     (22 / 63, 1.0, 1.0),
                     (23 / 63, 1.0, 1.0),
                     (24 / 63, 1.0, 1.0),
                     (25 / 63, 1.0, 1.0),
                     (26 / 63, 1.0, 1.0),
                     (27 / 63, 1.0, 1.0),
                     (28 / 63, 1.0, 1.0),
                     (29 / 63, 1.0, 1.0),
                     (30 / 63, 1.0, 1.0),
                     (31 / 63, 1.0, 1.0),
                     (32 / 63, 0.9375, 0.9375),
                     (33 / 63, 0.8750, 0.8750),
                     (34 / 63, 0.8125, 0.8125),
                     (35 / 63, 0.7500, 0.7500),
                     (36 / 63, 0.6875, 0.6875),
                     (37 / 63, 0.6250, 0.6250),
                     (38 / 63, 0.5625, 0.5625),
                     (39 / 63, 0.0, 0.0),
                     (40 / 63, 0.0, 0.0),
                     (41 / 63, 0.0, 0.0),
                     (42 / 63, 0.0, 0.0),
                     (43 / 63, 0.0, 0.0),
                     (44 / 63, 0.0, 0.0),
                     (45 / 63, 0.0, 0.0),
                     (46 / 63, 0.0, 0.0),
                     (47 / 63, 0.0, 0.0),
                     (48 / 63, 0.0, 0.0),
                     (49 / 63, 0.0, 0.0),
                     (50 / 63, 0.0, 0.0),
                     (51 / 63, 0.0, 0.0),
                     (52 / 63, 0.0, 0.0),
                     (53 / 63, 0.0, 0.0),
                     (54 / 63, 0.0, 0.0),
                     (55 / 63, 0.0, 0.0),
                     (56 / 63, 0.0, 0.0),
                     (57 / 63, 0.0, 0.0),
                     (58 / 63, 0.0, 0.0),
                     (59 / 63, 0.0, 0.0),
                     (60 / 63, 0.0, 0.0),
                     (61 / 63, 0.0, 0.0),
                     (62 / 63, 0.0, 0.0),
                     (63 / 63, 0.0, 0.0))
        }

        # mask magnitude
        cmap_custom2 = {
            'red': ((0.0, 1.0, 1.0),
                    (1 / 32, 1.0, 1.0),
                    (2 / 32, 1.0, 1.0),
                    (3 / 32, 1.0, 1.0),
                    (4 / 32, 1.0, 1.0),
                    (5 / 32, 1.0, 1.0),
                    (6 / 32, 1.0, 1.0),
                    (7 / 32, 1.0, 1.0),
                    (8 / 32, 1.0, 1.0),
                    (9 / 32, 1.0, 1.0),
                    (10 / 32, 1.0, 1.0),
                    (11 / 32, 1.0, 1.0),
                    (12 / 32, 1.0, 1.0),
                    (13 / 32, 1.0, 1.0),
                    (14 / 32, 1.0, 1.0),
                    (15 / 32, 1.0, 1.0),
                    (16 / 32, 1.0, 1.0),
                    (17 / 32, 1.0, 1.0),
                    (18 / 32, 1.0, 1.0),
                    (19 / 32, 1.0, 1.0),
                    (20 / 32, 1.0, 1.0),
                    (21 / 32, 1.0, 1.0),
                    (22 / 32, 1.0, 1.0),
                    (23 / 32, 1.0, 1.0),
                    (24 / 32, 1.0, 1.0),
                    (25 / 32, 0.9375, 0.9375),
                    (26 / 32, 0.8750, 0.8750),
                    (27 / 32, 0.8125, 0.8125),
                    (28 / 32, 0.7500, 0.7500),
                    (29 / 32, 0.6875, 0.6875),
                    (30 / 32, 0.6250, 0.6250),
                    (31 / 32, 0.5625, 0.5625),
                    (32 / 32, 0.5000, 0.5000)),
            'green': ((0.0, 1.0, 1.0),
                      (1 / 32, 1.0, 1.0),
                      (2 / 32, 1.0, 1.0),
                      (3 / 32, 1.0, 1.0),
                      (4 / 32, 1.0, 1.0),
                      (5 / 32, 1.0, 1.0),
                      (6 / 32, 1.0, 1.0),
                      (7 / 32, 1.0, 1.0),
                      (8 / 32, 1.0, 1.0),
                      (9 / 32, 0.9375, 0.9375),
                      (10 / 32, 0.8750, 0.8750),
                      (11 / 32, 0.8125, 0.8125),
                      (12 / 32, 0.7500, 0.7500),
                      (13 / 32, 0.6875, 0.6875),
                      (14 / 32, 0.6250, 0.6250),
                      (15 / 32, 0.5625, 0.5625),
                      (16 / 32, 0.5000, 0.5000),
                      (17 / 32, 0.4375, 0.4375),
                      (18 / 32, 0.3750, 0.3750),
                      (19 / 32, 0.3125, 0.3125),
                      (20 / 32, 0.2500, 0.2500),
                      (21 / 32, 0.1875, 0.1875),
                      (22 / 32, 0.1250, 0.1250),
                      (23 / 32, 0.0625, 0.0625),
                      (24 / 32, 0.0, 0.0),
                      (25 / 32, 0.0, 0.0),
                      (26 / 32, 0.0, 0.0),
                      (27 / 32, 0.0, 0.0),
                      (28 / 32, 0.0, 0.0),
                      (29 / 32, 0.0, 0.0),
                      (30 / 32, 0.0, 0.0),
                      (31 / 32, 0.0, 0.0),
                      (32 / 32, 0.0, 0.0)),
            'blue': ((0.0, 1.0, 1.0),
                     (1 / 32, 0.9375, 0.9375),
                     (2 / 32, 0.8750, 0.8750),
                     (3 / 32, 0.8125, 0.8125),
                     (4 / 32, 0.7500, 0.7500),
                     (5 / 32, 0.6875, 0.6875),
                     (6 / 32, 0.6250, 0.6250),
                     (7 / 32, 0.5625, 0.5625),
                     (8 / 32, 0.0, 0.0),
                     (9 / 32, 0.0, 0.0),
                     (10 / 32, 0.0, 0.0),
                     (11 / 32, 0.0, 0.0),
                     (12 / 32, 0.0, 0.0),
                     (13 / 32, 0.0, 0.0),
                     (14 / 32, 0.0, 0.0),
                     (15 / 32, 0.0, 0.0),
                     (16 / 32, 0.0, 0.0),
                     (17 / 32, 0.0, 0.0),
                     (18 / 32, 0.0, 0.0),
                     (19 / 32, 0.0, 0.0),
                     (20 / 32, 0.0, 0.0),
                     (21 / 32, 0.0, 0.0),
                     (22 / 32, 0.0, 0.0),
                     (23 / 32, 0.0, 0.0),
                     (24 / 32, 0.0, 0.0),
                     (25 / 32, 0.0, 0.0),
                     (26 / 32, 0.0, 0.0),
                     (27 / 32, 0.0, 0.0),
                     (28 / 32, 0.0, 0.0),
                     (29 / 32, 0.0, 0.0),
                     (30 / 32, 0.0, 0.0),
                     (31 / 32, 0.0, 0.0),
                     (32 / 32, 0.0, 0.0))
        }

        self.cmap_custom = matplotlib.colors.LinearSegmentedColormap('testCmap', segmentdata=cmap_custom, N=256)
        self.cmap_custom2 = matplotlib.colors.LinearSegmentedColormap('testCmap2', segmentdata=cmap_custom2, N=256)

    def log_train_loss(self, train_loss, step):
        self.add_scalar('train_loss', train_loss, step)

    def log_valid_loss(self, valid_loss, step):
        self.add_scalar('valid_loss', valid_loss, step)

    def log_train_joint_loss(self, train_main_loss, train_sub_loss, step):
        self.add_scalar('train_main_loss', train_main_loss, step)
        self.add_scalar('train_sub_loss', train_sub_loss, step)

    def log_valid_joint_loss(self, valid_main_loss, valid_sub_loss, step):
        self.add_scalar('valid_main_loss', valid_main_loss, step)
        self.add_scalar('valid_sub_loss', valid_sub_loss, step)

    def log_test_loss(self, test_loss, step):
        self.add_scalar('test_loss', test_loss, step)

    def log_scores(self, pesq, stoi, step):
        self.add_scalar('pesq', pesq, step)
        self.add_scalar('stoi', stoi, step)

    def log_per_pesq(self, snr, pesq_list):
        for i in range(len(pesq_list)):
            self.add_scalar('{}_test_pesq'.format(snr), pesq_list[i], i)

    def log_wav(self, noisy_wav, clean_wav, enhanced_wav, step):
        # <Audio>
        self.add_audio('noisy_wav', noisy_wav, step, cfg.FS)
        self.add_audio('clean_target_wav', clean_wav, step, cfg.FS)
        self.add_audio('enhanced_wav', enhanced_wav, step, cfg.FS)
