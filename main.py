import torch
import numpy as np
import os
import argparse
from utils import get_dataset, img_processing, data_init, img_reshape


def main():

    dst_train, dst_test, traindataloader, testdataloader = get_dataset(args.dataset, args.data_path)
    X_train, y_train = next(iter(traindataloader))
    X_test, y_test = next(iter(testdataloader))
    X_train, y_train, X_test, y_test = img_processing(X_train, y_train, X_test, y_test)
    X_initial, y_initial, X_pool, y_pool = data_init(X_train, y_train, args)
    X_list = [X_test, X_pool, X_initial]
    X_test_re, X_pool_re, X_initial_re = img_reshape(X_list)

    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'





if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Parameter Processing')
    parser.add_argument('--dataset', type=str, default='MNIST', help='dataset')
    parser.add_argument('--model', type=str, default='ConvNet', help='model')
    parser.add_argument('--data_path', type=str, default='data', help='dataset path')


    args = parser.parse_args()
    main()
