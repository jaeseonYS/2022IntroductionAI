import os
import random
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
import torch
from dataset import load_cache_feature, CLASSES
from models.gcn import GCN

import warnings
warnings.filterwarnings(action='ignore', category=FutureWarning)
warnings.filterwarnings(action='ignore', category=UserWarning)

# seed
seed = 2
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)

# hyper-parameters
epochs = 100

k = 5
in_channel = 128
hid_channel = 128

learning_rate = 0.001
weight_decay = 0.01
eps = 1e-08
gamma = 0.95

device = torch.device('cpu')
if torch.cuda.is_available():
    device = torch.device(0)


def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * (np.linalg.norm(b)))


def adjacency_matrix(features, test_size):
    size = len(features)
    train_size = size - test_size
    adj = np.zeros((size, size))
    for i in range(size):
        if i > train_size:
            for j in range(0, train_size):
                adj[i][j] = cosine_similarity(features[i], features[j])
        else:
            for j in range(size):
                adj[i][j] = cosine_similarity(features[i], features[j])
        idx = adj[i].argsort()[-k:][::-1]
        adj[i] = 0
        adj[i][idx] = 1
    return adj


def inference(feature_name, weight_path):
    model_name = 'gcn'

    print('######################### [GCN - %s]' % feature_name.upper())

    # dataset
    x_train, x_test, y_train, y_test = load_cache_feature(feature_name)

    test_size = len(y_test)
    features = np.concatenate([x_train, x_test], axis=0)
    adj = adjacency_matrix(features, len(x_test))
    y = np.concatenate([y_train, y_test], axis=0)

    # model
    if model_name == 'gcn':
        model = GCN(in_channel, hid_channel, out_channel=len(CLASSES)).to(device)
    model.load_state_dict(torch.load(weight_path))

    # eval
    model.eval()
    x = torch.tensor(features).float().to(device)
    a = torch.tensor(adj).float().to(device)
    y = torch.tensor(y).long().to(device)
    with torch.no_grad():
        pred = model(x, a)
        y_pred = torch.argmax(pred[-test_size:], dim=1).cpu().detach().numpy()
        y_true = y[-test_size:].cpu().detach().numpy()
        acc = accuracy_score(y_pred, y_true)
        matrix = confusion_matrix(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, average='macro')
        print('# Accuracy: %.4f' % acc)
        print('# F1-Score: %.4f' % f1)
        print('# Confusion Matrix')
        print(matrix)
        print('#########################\n\n')


if __name__ == '__main__':
    inference('mlp', 'logs_gcn/gcn/mlp/20220528-152424/best.pth')
    inference('lstm', 'logs_gcn/gcn/lstm/20220528-152427/best.pth')
    inference('cnn', 'logs_gcn/gcn/cnn/20220528-152428/best.pth')
    inference('lstmfcn', 'logs_gcn/gcn/lstmfcn/20220528-152429/best.pth')
    inference('attention', 'logs_gcn/gcn/attention/20220528-152430/best.pth')
    inference('resnet18', 'logs_gcn/gcn/resnet18/20220529-193116/best.pth')
