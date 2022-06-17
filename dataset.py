import os
import pickle
import librosa
import numpy as np
import matplotlib.pyplot as plt
import librosa.display
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import train_test_split
from torch.utils.data.dataset import Dataset

import warnings
warnings.filterwarnings(action='ignore', category=FutureWarning)

CLASSES = ['Hungry', 'Pain', 'Uncomfortable']


def mfcc(file_path, max_len=160000, sr=16000, n_mfcc=80, win_length=512, hop_length=256, n_fft=512):
    y, _ = librosa.load(file_path, sr=sr)
    if max_len is not None:
        while y.shape[0] < max_len:
            pad_len = min(max_len - y.shape[0], y.shape[0])
            y = np.concatenate([y, y[:pad_len]], axis=0)
        if y.shape[0] > max_len:
            y = y[:max_len]
    spec = librosa.feature.mfcc(y, sr, n_mfcc=n_mfcc, hop_length=hop_length, win_length=win_length, n_fft=n_fft)
    spec = librosa.power_to_db(spec, ref=np.max)
    return spec


def melspec(file_path, max_len=160000, sr=16000, n_mels=80, win_length=512, hop_length=256, n_fft=512):
    y, _ = librosa.load(file_path, sr=sr)
    if max_len is not None:
        while y.shape[0] < max_len:
            pad_len = min(max_len - y.shape[0], y.shape[0])
            y = np.concatenate([y, y[:pad_len]], axis=0)
        if y.shape[0] > max_len:
            y = y[:max_len]
    spec = librosa.feature.melspectrogram(y, sr, n_mels=n_mels, hop_length=hop_length, win_length=win_length, n_fft=n_fft)
    spec = librosa.power_to_db(spec, ref=np.max)
    return spec


def stft(file_path, max_len=160000, sr=16000, win_length=512, hop_length=256, n_fft=512):
    y, _ = librosa.load(file_path, sr=sr)
    if max_len is not None:
        while y.shape[0] < max_len:
            pad_len = min(max_len - y.shape[0], y.shape[0])
            y = np.concatenate([y, y[:pad_len]], axis=0)
        if y.shape[0] > max_len:
            y = y[:max_len]
    spec = librosa.stft(y, hop_length=hop_length, win_length=win_length, n_fft=n_fft)
    mags = np.abs(spec)
    log_spectrogram = librosa.amplitude_to_db(mags, ref=np.max)
    return log_spectrogram


def load_cache_data():
    dataset_path = 'data/Recordings'

    # load cache
    if os.path.exists('data/x_train.pickle'):
        with open('data/x_train.pickle', 'rb') as f:
            x_train = pickle.load(f)
        with open('data/x_test.pickle', 'rb') as f:
            x_test = pickle.load(f)
        with open('data/y_train.pickle', 'rb') as f:
            y_train = pickle.load(f)
        with open('data/y_test.pickle', 'rb') as f:
            y_test = pickle.load(f)
    # make cache
    else:
        files = []
        labels = []

        for d in os.listdir(dataset_path):
            dir_path = os.path.join(dataset_path, d)
            for file_name in os.listdir(dir_path):
                file_path = os.path.join(dir_path, file_name)
                if d in CLASSES:
                    files.append([file_path])
                    labels.append(CLASSES.index(d))

        # under sampling
        rus = RandomUnderSampler()
        files, labels = rus.fit_resample(files, labels)

        # transform mfcc
        data = []
        for file_path in files:
            data.append(melspec(file_path[0]))

        x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.25, stratify=labels, random_state=1)

        with open('data/x_train.pickle', 'wb') as f:
            pickle.dump(x_train, f)
        with open('data/x_test.pickle', 'wb') as f:
            pickle.dump(x_test, f)
        with open('data/y_train.pickle', 'wb') as f:
            pickle.dump(y_train, f)
        with open('data/y_test.pickle', 'wb') as f:
            pickle.dump(y_test, f)

    return np.array(x_train), np.array(x_test), np.array(y_train), np.array(y_test)


def load_cache_feature(feature_name):
    with open('data/f_train_{}.pickle'.format(feature_name), 'rb') as f:
        x_train = pickle.load(f)
    with open('data/f_test_{}.pickle'.format(feature_name), 'rb') as f:
        x_test = pickle.load(f)
    with open('data/y_train.pickle', 'rb') as f:
        y_train = pickle.load(f)
    with open('data/y_test.pickle', 'rb') as f:
        y_test = pickle.load(f)
    return np.array(x_train), np.array(x_test), np.array(y_train), np.array(y_test)


def to_normalize(spec):
    min_level_db = -100
    return np.clip((spec - min_level_db) / -min_level_db, 0, 1)


class CryingDataset(Dataset):
    def __init__(self, x, y, normalize=False):
        self.normalize = normalize
        self.x = x
        self.y = y

    def __getitem__(self, index):
        x, y = self.x[index], self.y[index]
        if self.normalize:
            x = to_normalize(x)
        return x, y

    def __len__(self):
        return len(self.x)


if __name__ == '__main__':
    data, _, label, _ = load_cache_data()
    spec = data[5]
    librosa.display.specshow(spec, sr=16000, hop_length=256)
    plt.title('Mel-Spectrogram')
    plt.xlabel('Time')
    plt.ylabel('Hz')
    plt.colorbar()
    plt.savefig('output/MelSpec.png')
