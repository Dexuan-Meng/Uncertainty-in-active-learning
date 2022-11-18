import torch
import numpy as np
import os
import argparse
from utils import get_dataset, img_processing, data_init


def main():

    dst_train, dst_test, traindataloader, testdataloader = get_dataset(args.dataset, args.data_path)
    X_train, y_train = next(iter(traindataloader))
    X_test, y_test = next(iter(testdataloader))
    X_train, y_train, X_test, y_test = img_processing(X_train, y_train, X_test, y_test)
    X_initial, y_initial, X_pool, y_pool = data_init(X_train, y_train, args)




if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Parameter Processing')
    parser.add_argument('--dataset', type=str, default='MNIST', help='dataset')
    parser.add_argument('--model', type=str, default='ConvNet', help='model')
    parser.add_argument('--data_path', type=str, default='data', help='dataset path')
    parser.add_argument('--ipc', type=int, default=3, help='image(s) per class')

    args = parser.parse_args()
    main()
