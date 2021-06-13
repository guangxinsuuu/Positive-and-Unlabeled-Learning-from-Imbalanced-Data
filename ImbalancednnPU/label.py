import numpy as np
import urllib.request
import os
import tarfile
import pickle

import torch
from sklearn.datasets import fetch_openml

def get_mnist():
    mnist = fetch_openml('mnist_784', data_home=".")

    x = mnist.data
    y = mnist.target
    # reshape to (#data, #channel, width, height)
    x = np.reshape(x, (x.shape[0], 1, 28, 28)) / 255.
    x_tr = np.asarray(x[:60000], dtype=np.float32)
    y_tr = np.asarray(y[:60000], dtype=np.int32)
    x_te = np.asarray(x[60000:], dtype=np.float32)
    y_te = np.asarray(y[60000:], dtype=np.int32)


    return (x_tr, y_tr), (x_te, y_te)

#
# def binarize_mnist_class(y_train, y_test):
#     y_train_bin = -np.ones(len(y_train), dtype=np.int32)
#     y_train_bin[y_train % 2 == 1] = 1
#     y_test_bin = -np.ones(len(y_test), dtype=np.int32)
#     y_test_bin[y_test % 2 == 1] = 1
#     return y_train_bin, y_test_bin
#
# 更改class：imbalance数据，1为正样本，其余为负样本
# 更改标签
def binarize_mnist_class(y_train, y_test):
    y_train_bin = np.ones(len(y_train), dtype=np.int32)
    y_train_bin[(y_train == 2) | (y_train == 6) | (y_train == 4) | (y_train == 5) | (y_train == 1)
                | (y_train == 7) | (y_train == 8) | (y_train == 9)| (y_train == 0)] = -1
    y_test_bin = np.ones(len(y_test), dtype=np.int32)
    y_test_bin[(y_test == 2) | (y_test == 6) | (y_test == 4) | (y_test == 5) | (y_test == 1) | (y_test == 7) | (y_test == 8) | (y_test == 9)| (y_test == 0)] = -1
    return y_train_bin, y_test_bin

def unpickle(file):
    fo = open(file, 'rb')
    dictionary = pickle.load(fo, encoding='latin1')
    fo.close()
    return dictionary


def conv_data2image(data):
    return np.rollaxis(data.reshape((3, 32, 32)), 0, 3)


def get_cifar10(path="./mldata"):
    if not os.path.isdir(path):
        os.mkdir(path)
    url = "http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
    file_name = os.path.basename(url)
    full_path = os.path.join(path, file_name)
    folder = os.path.join(path, "cifar-10-batches-py")
    # if cifar-10-batches-py folder doesn't exists, download from website
    if not os.path.isdir(folder):
        print("download the dataset from {} to {}".format(url, path))
        urllib.request.urlretrieve(url, full_path)
        with tarfile.open(full_path) as f:
            f.extractall(path=path)
        urllib.request.urlcleanup()

    x_tr = np.empty((0, 32 * 32 * 3))
    y_tr = np.empty(1)
    for i in range(1, 6):
        fname = os.path.join(folder, "%s%d" % ("data_batch_", i))
        data_dict = unpickle(fname)
        if i == 1:
            x_tr = data_dict['data']
            y_tr = data_dict['labels']
        else:
            x_tr = np.vstack((x_tr, data_dict['data']))
            y_tr = np.hstack((y_tr, data_dict['labels']))

    data_dict = unpickle(os.path.join(folder, 'test_batch'))
    x_te = data_dict['data']
    y_te = np.array(data_dict['labels'])

    bm = unpickle(os.path.join(folder, 'batches.meta'))
    # label_names = bm['label_names']
    # rehape to (#data, #channel, width, height)
    x_tr = np.reshape(x_tr, (np.shape(x_tr)[0], 3, 32, 32)).astype(np.float32)
    x_te = np.reshape(x_te, (np.shape(x_te)[0], 3, 32, 32)).astype(np.float32)
    # normalize
    x_tr /= 255.
    x_te /= 255.
    return (x_tr, y_tr), (x_te, y_te)  # , label_names


def binarize_cifar10_class(y_train, y_test,label_num):
    #更改标签
    y_train_bin = -np.ones(len(y_train), dtype=np.int32)
    y_train_bin[(y_train == label_num )] = 1
    y_test_bin = -np.ones(len(y_test), dtype=np.int32)
    y_test_bin[(y_test == label_num) ] = 1
    return y_train_bin, y_test_bin


def make_dataset(dataset, n_labeled, n_unlabeled, seed):
    def make_pu_dataset_from_binary_dataset(x, y, seed,labeled=n_labeled, unlabeled=n_unlabeled):
        labels = np.unique(y)
        positive, negative = labels[1], labels[0]
        x, y = np.asarray(x, dtype=np.float32), np.asarray(y, dtype=np.int32)
        assert(len(x) == len(y))
        perm = np.random.permutation(len(y))
        #为了将标签打乱
        x, y = x[perm], y[perm]
        #各数据数量
        n_p = (y == positive).sum()
        #人工定义的数量，用在positive部分
        n_lp = labeled
        n_n = (y == negative).sum()
        n_u = unlabeled
        if labeled + unlabeled == len(x):
            n_up = n_p - n_lp
        elif unlabeled == len(x):
            n_up = n_p
        else:
            raise ValueError("Only support |P|+|U|=|X| or |U|=|X|.")
        _prior = float(n_up) / float(n_u)
        #数据集划分
        xlp = x[y == positive][:n_lp]
        xup = np.concatenate((x[y == positive][n_lp:], xlp), axis=0)[:n_up]
        xun = x[y == negative]
        #pos+u_pos+u_n
        x_positive = xlp
        x_unlabel = np.asarray(np.concatenate((xup, xun), axis=0), dtype=np.float32)


        #pos+un: label
        y_positive = np.asarray((np.ones(n_lp)), dtype=np.int32)
        y_unlabel = np.asarray( -np.ones(n_u), dtype=np.int32)
        #y = np.asarray(np.concatenate((np.ones(n_lp), -np.ones(n_u))), dtype=np.int32)
        perm1 = np.random.permutation(len(y_positive))
        x_positive, y_positive = x_positive[perm1], y_positive[perm1]
        perm2 = np.random.permutation(len(y_unlabel))
        x_unlabel ,y_unlabel = x_unlabel[perm2], y_unlabel[perm2]
        return x_positive, x_unlabel, y_positive, y_unlabel, _prior

    def make_pn_dataset_from_binary_dataset(x, y):
        labels = np.unique(y)
        positive, negative = labels[1], labels[0]
        X, Y = np.asarray(x, dtype=np.float32), np.asarray(y, dtype=np.int32)
        n_p = (Y == positive).sum()
        n_n = (Y == negative).sum()
        Xp = X[Y == positive][:n_p]
        Xn = X[Y == negative][:n_n]
        X_pos = np.asarray(Xp, dtype=np.float32)
        X_neg = np.asarray(Xn, dtype=np.float32)
        Y_pos = np.asarray(np.ones(n_p), dtype=np.int32)
        Y_neg = np.asarray((-np.ones(n_n)), dtype=np.int32)
        perm1 = np.random.permutation(len(Y_pos))
        X_pos, Y_pos = X_pos[perm1], Y_pos[perm1]
        perm2 = np.random.permutation(len(Y_neg))
        X_neg, Y_neg = X_neg[perm2], Y_neg[perm2]
        return X_pos, X_neg, Y_pos, Y_neg
#看这块
    (x_train, y_train), (x_test, y_test) = dataset
    x_positive, x_unlabel, y_positive, y_unlabel, prior = make_pu_dataset_from_binary_dataset(x_train, y_train, seed)
    X_pos, X_neg, Y_pos, Y_neg = make_pn_dataset_from_binary_dataset(x_test, y_test)
    print("training:{}".format(x_train.shape))
    print("test:{}".format(x_test.shape))
    return list(zip(x_positive, y_positive)), list(zip(x_unlabel, y_unlabel)),list(zip(X_pos, Y_pos)),list(zip(X_neg, Y_neg)), prior


def load_dataset(dataset_name, n_labeled, n_unlabeled,seed,label_num):
    if dataset_name == "mnist":
        (x_train, y_train), (x_test, y_test) = get_mnist()
        y_train, y_test = binarize_mnist_class(y_train, y_test)
    elif dataset_name == "cifar10":
        (x_train, y_train), (x_test, y_test) = get_cifar10()
        y_train, y_test = binarize_cifar10_class(y_train, y_test,label_num)
    else:
        raise ValueError("dataset name {} is unknown.".format(dataset_name))
    xy_pos_train, xy_unlabel_train, xy_pos_test, xy_neg_test, prior = make_dataset(((x_train, y_train), (x_test, y_test)), n_labeled, n_unlabeled,seed)
    return xy_pos_train, xy_unlabel_train, xy_pos_test, xy_neg_test, prior
#test如果需要更改，和上边一样的思路

