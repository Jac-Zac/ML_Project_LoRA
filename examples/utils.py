import gzip
import random

import numpy as np
from tinygrad import Tensor
from tinygrad.engine.jit import TinyJit
from tinygrad.helpers import CI, fetch
from tqdm import trange


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


def mix_old_and_new_data(X, Y, worst_class, ratio):
    """Mixes a certain percentage of the original data with the worst performing class data."""

    # Calculate the number of samples for the original dataset and worst class
    num_original_samples = int(len(Y) * ratio)
    num_worst_class_samples = len(Y) - num_original_samples

    # Randomly sample from the original dataset
    sampled_indices = random.sample(range(len(Y)), num_original_samples)
    original_x_sampled = [X[i] for i in sampled_indices]
    original_y_sampled = [Y[i] for i in sampled_indices]

    # Separate the worst performing class samples
    worst_class_x, worst_class_y = filter_data_by_class(X, Y, worst_class)

    # Ensure there are enough samples from the worst class
    num_worst_class_samples = min(num_worst_class_samples, len(worst_class_y))

    # Randomly sample from the worst class
    worst_sampled_indices = random.sample(
        range(len(worst_class_y)), num_worst_class_samples
    )
    worst_class_x_sampled = [worst_class_x[i] for i in worst_sampled_indices]
    worst_class_y_sampled = [worst_class_y[i] for i in worst_sampled_indices]

    # Combine the original samples with the worst class samples
    mixed_X = np.concatenate((original_x_sampled, worst_class_x_sampled), axis=0)
    mixed_Y = np.concatenate((original_y_sampled, worst_class_y_sampled), axis=0)

    return mixed_X, mixed_Y


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


def train(
    model,
    X_train,
    Y_train,
    optim,
    steps,
    BS=128,
    lossfn=lambda out, y: out.sparse_categorical_crossentropy(y),
    transform=lambda x: x,
    target_transform=lambda x: x,
    noloss=False,
    allow_jit=True,
):

    def train_step(x, y):
        # network
        out = model.forward(x) if hasattr(model, "forward") else model(x)
        loss = lossfn(out, y)
        optim.zero_grad()
        loss.backward()
        if noloss:
            del loss
        optim.step()
        if noloss:
            return (None, None)
        cat = out.argmax(axis=-1)
        accuracy = (cat == y).mean()
        return loss.realize(), accuracy.realize()

    if allow_jit:
        train_step = TinyJit(train_step)

    with Tensor.train():
        losses, accuracies = [], []
        for i in (t := trange(steps, disable=CI)):
            samp = np.random.randint(0, X_train.shape[0], size=(BS))
            x = Tensor(transform(X_train[samp]), requires_grad=False)
            y = Tensor(target_transform(Y_train[samp]))
            loss, accuracy = train_step(x, y)
            # printing
            if not noloss:
                loss, accuracy = loss.numpy(), accuracy.numpy()
                losses.append(loss)
                accuracies.append(accuracy)
                t.set_description("loss %.2f accuracy %.2f" % (loss, accuracy))
    return [losses, accuracies]


def evaluate(
    model,
    X_test,
    Y_test,
    num_classes=None,
    BS=128,
    return_predict=False,
    transform=lambda x: x,
    target_transform=lambda y: y,
):
    Tensor.training = False

    def numpy_eval(Y_test, num_classes):
        Y_test_preds_out = np.zeros(list(Y_test.shape) + [num_classes])
        for i in trange((len(Y_test) - 1) // BS + 1, disable=CI):
            x = Tensor(transform(X_test[i * BS : (i + 1) * BS]))
            out = model.forward(x) if hasattr(model, "forward") else model(x)
            Y_test_preds_out[i * BS : (i + 1) * BS] = out.numpy()
        Y_test_preds = np.argmax(Y_test_preds_out, axis=-1)
        Y_test = target_transform(Y_test)
        return (Y_test == Y_test_preds).mean(), Y_test_preds

    if num_classes is None:
        num_classes = Y_test.max().astype(int) + 1
    acc, Y_test_pred = numpy_eval(Y_test, num_classes)
    print("test set accuracy is %f" % acc)
    return (acc, Y_test_pred) if return_predict else acc
