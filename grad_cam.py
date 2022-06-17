import cv2
import random
import numpy as np

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

from pytorch_grad_cam import GradCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, FullGrad
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
from torchvision.models import resnet50

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
batch_size = 1

device = torch.device('cpu')
if torch.cuda.is_available():
    device = torch.device(0)


def gradcam(model_name, weight_path):
    assert model_name in ['mlp', 'lstm', 'cnn', 'lstmfcn', 'attention']

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

    print(model)

    model.load_state_dict(torch.load(weight_path))

    # GradCAM
    target_layers = [model.layer]
    cam = GradCAM(model=model, target_layers=target_layers, use_cuda=True)
    targets = [ClassifierOutputTarget(3)]
    for i, (x, y) in enumerate(train_loader):
        print('#', i)
        if i == 20:
            break
        grayscale_cam = cam(input_tensor=x, targets=targets)
        grayscale_cam = (grayscale_cam[0, :] * -1) + 1
        x = x[0].detach().numpy()
        y = int(y.numpy())
        cv2.imwrite('output/gradcam/{}/{}_{}.jpg'.format(model_name, str(i), str(y)), x*255)
        img = cv2.cvtColor(x, cv2.COLOR_GRAY2RGB)
        visualization = show_cam_on_image(img, grayscale_cam, use_rgb=True)
        cv2.imwrite('output/gradcam/{}/{}_{}_cam.jpg'.format(model_name, str(i), str(y)), visualization)


if __name__ == '__main__':
    # gradcam('cnn', 'logs/cnn/20220528-114757/best.pth')
    # gradcam('lstm', 'logs/lstm/20220528-113930/best.pth')
    gradcam('lstmfcn', 'logs/lstmfcn/20220528-114956/best.pth')
    # gradcam('attention', 'logs/attention/20220528-134534/best.pth')
