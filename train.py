import os
import random
import numpy as np
from datetime import datetime
from sklearn.metrics import accuracy_score
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
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
seed = 1
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)

# hyper-parameters
batch_size = 32
epochs = 100
hid_channel = 128

learning_rate = 0.001
weight_decay = 0.01
eps = 1e-08
gamma = 0.95

device = torch.device('cpu')
if torch.cuda.is_available():
    device = torch.device(0)


def load(model, weight_path):
    model.load_state_dict(torch.load(weight_path))
    # for param in model.parameters():
    #     param.requires_grad = False
    return model


def train(model_name):
    assert model_name in ['mlp', 'lstm', 'cnn', 'lstmfcn', 'attention', 'resnet18', 'resnet50']

    # logs
    current_time = datetime.now().strftime('%Y%m%d-%H%M%S')
    log_dir = os.path.join('logs', model_name, current_time)
    writer = SummaryWriter(log_dir)

    # dataset
    x_train, x_test, y_train, y_test = load_cache_data()
    train_dataset = CryingDataset(x_train, y_train, normalize=True)
    test_dataset = CryingDataset(x_test, y_test, normalize=True)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
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
        model.mlp = load(model.mlp, 'logs/mlp/20220528-113852/best.pth')
        model.lstm = load(model.lstm, 'logs/lstm/20220528-113930/best.pth')
        model.cnn = load(model.cnn, 'logs/cnn/20220528-114757/best.pth')
        model.lstmfcn = load(model.lstmfcn, 'logs/lstmfcn/20220528-114956/best.pth')
    elif model_name == 'resnet18':
        model = Resnet18(out_channel=len(CLASSES)).to(device)
        model.base = load(model.base, 'weights/crisscross.pth')
    elif model_name == 'resnet50':
        model = Resnet50(out_channel=len(CLASSES)).to(device)

    # loss function
    criterion = nn.CrossEntropyLoss()

    # optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, eps=eps, weight_decay=weight_decay)

    # scheduler
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)

    # train
    max_score = 0
    for epoch in range(1, epochs+1):
        model.train()
        avg_loss = 0
        lr = optimizer.param_groups[0]['lr']
        writer.add_scalar('train/lr', lr, epoch)
        for i, (x, y) in enumerate(train_loader):
            optimizer.zero_grad()
            x = x.float().to(device)
            y = y.long().to(device)
            _, pred = model(x)
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()
            avg_loss += loss.item()
        scheduler.step()
        avg_loss = avg_loss / len(train_loader)
        # print('[Epoch %d] loss: %.4f' % (epoch, avg_loss))
        writer.add_scalar('train/loss', avg_loss, epoch)

        # eval
        model.eval()
        with torch.no_grad():
            y_pred = np.array([])
            y_true = np.array([])
            avg_loss = 0
            for i, (x, y) in enumerate(test_loader):
                x = x.float().to(device)
                y = y.long().to(device)
                _, pred = model(x)
                loss = criterion(pred, y)
                avg_loss += loss.item()
                y_pred = np.concatenate([y_pred, torch.argmax(pred, dim=1).cpu().detach().numpy()])
                y_true = np.concatenate([y_true, y.cpu().detach().numpy()])
            avg_loss = avg_loss / len(test_loader)
            score = accuracy_score(y_true, y_pred)
            print('[Epoch %d] loss: %.4f, score: %.4f' % (epoch, avg_loss, score))
            writer.add_scalar('eval/loss', avg_loss, epoch)
            writer.add_scalar('eval/accuracy', score, epoch)

            # save
            if max_score < score:
                max_score = score
                torch.save(model.state_dict(), os.path.join(log_dir, 'best.pth'))
    print('# Model: %s\t | Accuracy: %.4f' % (model_name, max_score))


if __name__ == '__main__':
    print('START !!!')
    train('mlp')
    train('lstm')
    train('cnn')
    train('lstmfcn')
    train('attention')
    print('END !!!')
