import random
import numpy as np
import torch
from torch.utils.data import DataLoader
from dataset import load_cache_data, CryingDataset, CLASSES
from models.attention import Attention
from matplotlib import pyplot as plt

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


def inference(weight_path):
    # dataset
    x_train, x_test, y_train, y_test = load_cache_data()
    train_dataset = CryingDataset(x_train, y_train, normalize=True)
    test_dataset = CryingDataset(x_test, y_test, normalize=True)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # model
    _, h, w = x_train.shape
    model = Attention(in_channel=h, hid_channel=hid_channel, out_channel=len(CLASSES)).to(device)
    model.load_state_dict(torch.load(weight_path))

    # inference
    model.eval()
    with torch.no_grad():
        for i, (x, y) in enumerate(train_loader):
            x = x.float().to(device)

            z, x, s = model(x, True)
            z = z.cpu().detach().numpy()[10][:, :16]
            x = x.cpu().detach().numpy()[10][np.newaxis, :16]
            s = s.cpu().detach().numpy()[10]

            plt.title('Attention Input')
            plt.imshow(z)
            plt.axis('off')
            plt.savefig('output/attention_in.png')
            plt.close()

            plt.title('Attention Output')
            plt.imshow(x)
            plt.axis('off')
            plt.savefig('output/attention_out.png')
            plt.close()

            plt.title('Attention Score')
            plt.imshow(s)
            plt.colorbar()
            plt.axis('off')
            plt.savefig('output/attention_score.png')
            plt.close()

            exit()


if __name__ == '__main__':
    inference('logs/attention/20220528-134534/best.pth')
