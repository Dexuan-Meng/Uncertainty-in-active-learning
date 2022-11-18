import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def get_dataset(dataset, datapath, args=None):

    if dataset == 'MNIST':

        dst_train = datasets.MNIST(datapath, train=True, download=True, transform=transforms.ToTensor())
        dst_test = datasets.MNIST(datapath, train=False, download=True, transform=transforms.ToTensor())
        train_dataloader = DataLoader(dst_train, shuffle=True, batch_size=60000)
        test_dataloader = DataLoader(dst_test, shuffle=True, batch_size=10000)

    return dst_train, dst_test, train_dataloader, test_dataloader


def img_reshape(list):
    re_list = []
    for X in list:
        re_list.append(X.reshape(X.shape[0], -1))
    return re_list


def img_processing(X_train, y_train, X_test, y_test):

    X_train = X_train.detach().cpu().numpy()
    y_train = y_train.detach().cpu().numpy()
    X_test = X_test.detach().cpu().numpy()
    y_test = y_test.detach().cpu().numpy()

    X_train = X_train.reshape(60000, 1, 28, 28)
    X_test = X_test.reshape(10000, 1, 28, 28)

    return X_train, y_train, X_test, y_test


def data_init(X_train, y_train, args=None):
    """
    generate the initial training dataset, with arg.ipc images in each class
    :param X_train: X_train
    :param y_train: y_train
    :param args: args
    :return: initial training image and target, Pool image and target
    """

    initial_idx = np.array([], dtype=np.int)

    for i in range(10):
        idx = np.where(y_train == i)[0][0:args.ipc]
        initial_idx = np.concatenate((initial_idx, idx))

    X_initial = X_train[initial_idx]
    y_initial = y_train[initial_idx]

    X_pool = np.delete(X_train, initial_idx, axis=0)
    y_pool = np.delete(y_train, initial_idx, axis=0)

    return X_initial, y_initial, X_pool, y_pool


