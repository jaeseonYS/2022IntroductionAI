import cv2
import random
import pickle
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
import torch
from torch.utils.data import DataLoader
from dataset import load_cache_data, CryingDataset, CLASSES
from models.mlp import MLP
from models.cnn import CNN
from models.lstm import LSTM
from models.lstmfcn import LSTMFCN
from models.attention import Attention
from models.resnet18 import Resnet18
from models.resnet50 import Resnet50

import warnings
warnings.filterwarnings(action='ignore', category=FutureWarning)
warnings.filterwarnings(action='ignore', category=UserWarning)

# seed
seed = 0
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)

# hyper-parameters
hid_channel = 128
batch_size = 32

device = torch.device('cpu')
if torch.cuda.is_available():
    device = torch.device(0)


def inference(model_name, weight_path):
    assert model_name in ['mlp', 'lstm', 'cnn', 'lstmfcn', 'attention', 'resnet18', 'resnet50']

    print('######################### [%s]' % model_name.upper())

    # dataset
    x_train, x_test, y_train, y_test = load_cache_data()
    train_dataset = CryingDataset(x_train, y_train, normalize=True)
    test_dataset = CryingDataset(x_test, y_test, normalize=True)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # model
    _, h, w = x_train.shape
    if model_name == 'mlp':
        model = MLP(in_channel=h, hid_channel=hid_channel, out_channel=len(CLASSES)).to(device)
    elif model_name == 'lstm':
        model = LSTM(in_channel=h, hid_channel=hid_channel, out_channel=len(CLASSES)).to(device)
    elif model_name == 'cnn':
        model = CNN(in_channel=1, hid_channel=hid_channel, out_channel=len(CLASSES)).to(device)
    elif model_name == 'lstmfcn':
        model = LSTMFCN(in_channel=h, hid_channel=hid_channel, out_channel=len(CLASSES)).to(device)
    elif model_name == 'attention':
        model = Attention(in_channel=h, hid_channel=hid_channel, out_channel=len(CLASSES)).to(device)
    elif model_name == 'resnet18':
        model = Resnet18(out_channel=len(CLASSES)).to(device)
    elif model_name == 'resnet50':
        model = Resnet50(out_channel=len(CLASSES)).to(device)
    model.load_state_dict(torch.load(weight_path))

    # inference
    model.eval()
    with torch.no_grad():
        features = []
        y_pred = np.array([])
        y_true = np.array([])
        for i, (x, y) in enumerate(train_loader):
            x = x.float().to(device)
            y = y.long().to(device)
            feature, pred = model(x)
            y_pred = np.concatenate([y_pred, torch.argmax(pred, dim=1).cpu().detach().numpy()])
            y_true = np.concatenate([y_true, y.cpu().detach().numpy()])
            for f in feature.squeeze().cpu().detach().numpy():
                features.append(f)
        acc = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, average='macro')
        matrix = confusion_matrix(y_true, y_pred)
        print('# [Train]')
        print('# Accuracy: %.4f' % acc)
        print('# F1-Score: %.4f' % f1)
        print('# Confusion Matrix')
        print(matrix)
        with open('data/f_train_{}.pickle'.format(model_name), 'wb') as f:
            pickle.dump(features, f)

        print()

        features = []
        y_pred = np.array([])
        y_true = np.array([])
        xs = []
        for i, (x, y) in enumerate(test_loader):
            x = x.float().to(device)
            y = y.long().to(device)
            feature, pred = model(x)
            y_pred = np.concatenate([y_pred, torch.argmax(pred, dim=1).cpu().detach().numpy()])
            y_true = np.concatenate([y_true, y.cpu().detach().numpy()])
            for f in feature.squeeze().cpu().detach().numpy():
                features.append(f)
            for xx in x.cpu().detach().numpy():
                xs.append(xx)

        # outlier check
        for i, x in enumerate(xs):
            if not y_true[i] == y_pred[i]:
                cv2.imwrite('output/outlier/{}/{}_p({})_t({}).jpg'.format(model_name, str(i), str(int(y_pred[i])), str(int(y_true[i]))), x * 255)

        acc = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, average='macro')
        matrix = confusion_matrix(y_true, y_pred)
        print('# [Test]')
        print('# Accuracy: %.4f' % acc)
        print('# F1-Score: %.4f' % f1)
        print('# Confusion Matrix')
        print(matrix)
        with open('data/f_test_{}.pickle'.format(model_name), 'wb') as f:
            pickle.dump(features, f)

    print('#########################\n\n')


if __name__ == '__main__':
    inference('mlp', 'logs/mlp/20220528-113852/best.pth')
    inference('lstm', 'logs/lstm/20220528-113930/best.pth')
    inference('cnn', 'logs/cnn/20220528-114757/best.pth')
    inference('lstmfcn', 'logs/lstmfcn/20220528-114956/best.pth')
    inference('attention', 'logs/attention/20220528-134534/best.pth')
    inference('resnet18', 'logs/resnet18/20220529-185449/best.pth')
