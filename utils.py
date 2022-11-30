import numpy as np
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from modAL.models import ActiveLearner
from networks import CNN
from sklearn.ensemble import RandomForestClassifier
from skorch import NeuralNetClassifier


def get_dataset(dataset, datapath, args=None):

    if dataset == 'MNIST':

        dst_train = datasets.MNIST(datapath, train=True, download=True, transform=transforms.ToTensor())
        dst_test = datasets.MNIST(datapath, train=False, download=True, transform=transforms.ToTensor())
        train_dataloader = DataLoader(dst_train, shuffle=True, batch_size=60000)
        test_dataloader = DataLoader(dst_test, shuffle=True, batch_size=10000)

    return dst_train, dst_test, train_dataloader, test_dataloader


def get_network(model, args):
    if model == 'CNN':
        net = NeuralNetClassifier(CNN,
                                  max_epochs=50,
                                  batch_size=128,
                                  lr=0.001,
                                  optimizer=torch.optim.Adam,
                                  criterion=torch.nn.CrossEntropyLoss,
                                  train_split=None,
                                  verbose=0,
                                  device=args.device)
    elif model == 'RFC':
        net = RandomForestClassifier()
    elif model == 'RFC':

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


######## Query Strategies ########
def uniform(learner, X, n_instances=1):
    """
    Uniform acquisition function. Sample the query_index uniformly from pool dataset.
    :param learner: modAL.model.ActiveLearner Class
    :param X: the pool dataset to be sampled
    :param n_instances: number of instances for each query
    :return: query_index and query data
    """
    query_idx = np.random.choice(range(len(X)), size=n_instances, replace=False)
    return query_idx, X[query_idx]

def max_entropy(learner, x, n_instances=1, T=100):
    """
    Max_entropy acquisition function. Query the data point with the max entropy (max uncertainty).
    :param learner: modAL.model.ActiveLearner Class
    :param X: the pool dataset to be sampled
    :param n_instances: number of selected instances for each query
    :param T: number of samples of the weight of BNN
    :return: query_index and query data
    """
    random_subset = np.random.choice(range(len(x)), size=2000, replace=False)
    with torch.no_grad():
        outputs = np.stack([torch.softmax(learner.estimator.forward(x[random_subset], training=True),dim=-1).cpu().numpy()
                            for t in range(T)])
    pc = outputs.mean(axis=0)
    acquisition = (-pc*np.log(pc + 1e-10)).sum(axis=-1)
    idx = (-acquisition).argsort()[:n_instances]
    query_idx = random_subset[idx]
    return query_idx, x[query_idx]


def bald(learner, x, n_instances=1, T=100):
    """
    BALD acquisition function.
    :param learner: modAL.model.ActiveLearner Class
    :param X: the pool dataset to be sampled
    :param n_instances: number of selected instances for each query
    :param T: number of samples of the weight of BNN
    :return: query_index and query data
    """
    random_subset = np.random.choice(range(len(x)), size=2000, replace=False)
    with torch.no_grad():
        outputs = np.stack([torch.softmax(torch.tensor(learner.estimator.predict(x[random_subset])).float(),dim=-1).cpu().numpy()
                            for t in range(T)])
    pc = outputs.mean(axis=0)
    h = (-pc*np.log(pc + 1e-10)).sum(axis=-1)
    e_h = - np.mean(np.sum(outputs * np.log(outputs + 1e-10), axis=-1), axis=0)  # [batch size]
    acquisition = h - e_h
    idx = (-acquisition).argsort()[:n_instances]
    query_idx = random_subset[idx]
    return query_idx, x[query_idx]


def active_learning_procedure(query_strategy,
                              X_test,
                              y_test,
                              X_pool,
                              y_pool,
                              X_initial,
                              y_initial,
                              estimator,
                              n_queries=100,
                              n_instances=10):

    learner = ActiveLearner(estimator=estimator, X_training=X_initial, y_training=y_initial,
                            query_strategy=query_strategy, )

    perf_hist = [learner.score(X_test, y_test)]
    # print(X_pool.shape)
    for index in range(n_queries):
        query_idx, query_instance = learner.query(X_pool, n_instances)
        learner.teach(X_pool[query_idx], y_pool[query_idx])
        X_pool = np.delete(X_pool, query_idx, axis=0)
        y_pool = np.delete(y_pool, query_idx, axis=0)
        model_accuracy = learner.score(X_test, y_test)
        print('Accuracy after query {n}: {acc:0.4f}'.format(n=index + 1, acc=model_accuracy))
        perf_hist.append(model_accuracy)
    # print(X_pool.shape)

    return perf_hist

