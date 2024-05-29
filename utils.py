import gzip

import numpy as np
from tinygrad import Tensor
from tinygrad.helpers import fetch


def filter_data_by_class(X, Y, class_label):
    class_indices = Y == class_label
    return X[class_indices], Y[class_indices]


def get_mislabeled_counts(y, y_pred, n_output) -> dict[int, float]:
    mislabeled_counts_dict = {}
    for cls in range(n_output):
        class_mask = y == cls
        total_predictions = np.sum(class_mask)
        if total_predictions > 0:
            incorrect_predictions = np.sum(class_mask & (y != y_pred))
            mislabeled_counts_dict[cls] = incorrect_predictions
        else:
            mislabeled_counts_dict[cls] = -np.inf
    return mislabeled_counts_dict


def pretty_print_mislabeled_counts(mislabeled_counts: dict[int, float]) -> None:
    for cls, count in mislabeled_counts.items():
        print(f"Class {cls}: Missing {count}")


def fetch_fashion_mnist(tensors=False):
    parse = lambda file: np.frombuffer(gzip.open(file).read(), dtype=np.uint8).copy()
    BASE_URL = "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/"
    X_train = (
        parse(fetch(f"{BASE_URL}train-images-idx3-ubyte.gz"))[0x10:]
        .reshape((-1, 28 * 28))
        .astype(np.float32)
    )
    Y_train = parse(fetch(f"{BASE_URL}train-labels-idx1-ubyte.gz"))[8:].astype(np.int8)
    X_test = (
        parse(fetch(f"{BASE_URL}t10k-images-idx3-ubyte.gz"))[0x10:]
        .reshape((-1, 28 * 28))
        .astype(np.float32)
    )
    Y_test = parse(fetch(f"{BASE_URL}t10k-labels-idx1-ubyte.gz"))[8:].astype(np.int8)
    if tensors:
        return (
            Tensor(X_train).reshape(-1, 1, 28, 28),
            Tensor(Y_train),
            Tensor(X_test).reshape(-1, 1, 28, 28),
            Tensor(Y_test),
        )
    else:
        return X_train, Y_train, X_test, Y_test


def fetch_mnist(tensors=False):
    parse = lambda file: np.frombuffer(gzip.open(file).read(), dtype=np.uint8).copy()
    BASE_URL = "https://storage.googleapis.com/cvdf-datasets/mnist/"  # http://yann.lecun.com/exdb/mnist/ lacks https
    X_train = (
        parse(fetch(f"{BASE_URL}train-images-idx3-ubyte.gz"))[0x10:]
        .reshape((-1, 28 * 28))
        .astype(np.float32)
    )
    Y_train = parse(fetch(f"{BASE_URL}train-labels-idx1-ubyte.gz"))[8:].astype(np.int8)
    X_test = (
        parse(fetch(f"{BASE_URL}t10k-images-idx3-ubyte.gz"))[0x10:]
        .reshape((-1, 28 * 28))
        .astype(np.float32)
    )
    Y_test = parse(fetch(f"{BASE_URL}t10k-labels-idx1-ubyte.gz"))[8:].astype(np.int8)
    if tensors:
        return (
            Tensor(X_train).reshape(-1, 1, 28, 28),
            Tensor(Y_train),
            Tensor(X_test).reshape(-1, 1, 28, 28),
            Tensor(Y_test),
        )
    else:
        return X_train, Y_train, X_test, Y_test
