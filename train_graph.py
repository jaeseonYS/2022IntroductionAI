import os
import random
import numpy as np
from datetime import datetime
from sklearn.metrics import accuracy_score, confusion_matrix
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
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


def train(feature_name):
    model_name = 'gcn'

    print('######################### [GCN - %s]' % feature_name.upper())

    # logs
    current_time = datetime.now().strftime('%Y%m%d-%H%M%S')
    log_dir = os.path.join('logs_gcn', model_name, feature_name, current_time)
    writer = SummaryWriter(log_dir)

    # dataset
    x_train, x_test, y_train, y_test = load_cache_feature(feature_name)

    test_size = len(y_test)
    features = np.concatenate([x_train, x_test], axis=0)
    adj = adjacency_matrix(features, len(x_test))
    y = np.concatenate([y_train, y_test], axis=0)

    # model
    if model_name == 'gcn':
        model = GCN(in_channel, hid_channel, out_channel=len(CLASSES)).to(device)

    # loss function
    criterion = F.nll_loss

    # optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, eps=eps, weight_decay=weight_decay)

    # scheduler
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)

    # train
    max_score = 0
    for epoch in range(1, epochs+1):
        model.train()
        lr = optimizer.param_groups[0]['lr']
        writer.add_scalar('train/lr', lr, epoch)
        optimizer.zero_grad()

        x = torch.tensor(features).float().to(device)
        a = torch.tensor(adj).float().to(device)
        y = torch.tensor(y).long().to(device)

        pred = model(x, a)

        loss = criterion(pred[:-test_size], y[:-test_size])
        loss.backward()
        optimizer.step()
        scheduler.step()
        # print('[Epoch %d] loss: %.4f' % (epoch, loss))
        writer.add_scalar('train/loss', loss.item(), epoch)

        # eval
        model.eval()
        with torch.no_grad():
            pred = model(x, a)
            loss = criterion(pred[-test_size:], y[-test_size:])
            y_pred = torch.argmax(pred[-test_size:], dim=1).cpu().detach().numpy()
            y_true = y[-test_size:].cpu().detach().numpy()
            score = accuracy_score(y_pred, y_true)
            # print('[Epoch %d] loss: %.4f, score: %.4f' % (epoch, loss, score))
            writer.add_scalar('eval/loss', loss.item(), epoch)
            writer.add_scalar('eval/accuracy', score, epoch)

            # save
            if max_score < score:
                max_score = score
                torch.save(model.state_dict(), os.path.join(log_dir, 'best.pth'))
    print('# Accuracy: %.4f' % max_score)
    matrix = confusion_matrix(y_true, y_pred)
    print('# Confusion Matrix')
    print(matrix)
    print('#########################\n\n')


if __name__ == '__main__':
    train('mlp')
    train('lstm')
    train('cnn')
    train('lstmfcn')
    train('attention')
    train('resnet18')
