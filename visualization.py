import numpy as np
from dataset import load_cache_feature
from sklearn.manifold import TSNE
from matplotlib import pyplot as plt


def visualization(model_name):
    x_train, x_test, y_train, y_test = load_cache_feature(model_name)
    # train
    model = TSNE(n_components=2)
    rst = model.fit_transform(x_train)

    plt.title(model_name.upper())
    class_0 = np.where(y_train == 0)
    class_1 = np.where(y_train == 1)
    class_2 = np.where(y_train == 2)
    plt.scatter(rst[class_0, 0], rst[class_0, 1], color='red', label=0)
    plt.scatter(rst[class_1, 0], rst[class_1, 1], color='blue', label=1)
    plt.scatter(rst[class_2, 0], rst[class_2, 1], color='green', label=2)
    plt.legend()
    plt.savefig('output/train_{}.png'.format(model_name.upper()))
    plt.close()

    # test
    model = TSNE(n_components=2)
    rst = model.fit_transform(x_test)

    plt.title(model_name.upper())
    class_0 = np.where(y_test == 0)
    class_1 = np.where(y_test == 1)
    class_2 = np.where(y_test == 2)
    plt.scatter(rst[class_0, 0], rst[class_0, 1], color='red', label=0)
    plt.scatter(rst[class_1, 0], rst[class_1, 1], color='blue', label=1)
    plt.scatter(rst[class_2, 0], rst[class_2, 1], color='green', label=2)
    plt.legend()
    plt.savefig('output/test_{}.png'.format(model_name.upper()))
    plt.close()


visualization('mlp')
visualization('lstm')
visualization('cnn')
visualization('lstmfcn')
visualization('attention')
