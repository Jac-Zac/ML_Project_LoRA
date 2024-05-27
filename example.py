#!/usr/bin/env python3

import numpy as np
from tinygrad import Tensor, nn

from extra.datasets import fetch_mnist
from extra.training import evaluate, train
from lora_tinygrad import LoRA


class TinyNet:
    def __init__(self):
        self.l1 = nn.Linear(784, 128, bias=False)
        self.l2 = nn.Linear(128, 10, bias=False)

    def __call__(self, x):
        x = self.l1(x)
        x = x.leakyrelu()
        x = self.l2(x)
        return x


def filter_data_by_class(X, Y, class_label):
    class_indices = np.where(Y == class_label)[0]

    filtered_X = X[class_indices]
    filtered_Y = Y[class_indices]

    return filtered_X, filtered_Y


def _get_mislabeled_counts(y, y_pred) -> dict[int, float]:
    mislabeled_counts_dict: dict[int, float] = {cls: -np.inf for cls in range(10)}
    for cls in range(10):
        total_predictions = np.sum(y == cls)
        incorrect_predictions = np.sum((y == cls) & (y != y_pred))
        if total_predictions > 0:
            mislabeled_count = incorrect_predictions
        else:
            mislabeled_count = -np.inf
        mislabeled_counts_dict[cls] = mislabeled_count
    return mislabeled_counts_dict


def _pretty_print_mislabeled_counts(mislabeled_counts: dict[int, float]) -> None:
    for cls in mislabeled_counts.keys():
        print(f"Class {cls}: Missing {mislabeled_counts[cls]}")


if __name__ == "__main__":
    print("Simulating a pre-trained model, with one epoch..")
    lrs = [1e-3]
    epochss = [1]
    BS = 128

    X_train, Y_train, X_test, Y_test = fetch_mnist()

    steps = len(X_train) // BS
    x = Tensor.randn(1, 28, 28).reshape(-1)

    model = TinyNet()

    for lr, epochs in zip(lrs, epochss):
        optimizer = nn.optim.Adam(nn.state.get_parameters(model), lr=lr)
        for epoch in range(1, epochs + 1):
            train(
                model,
                X_train,
                Y_train,
                optimizer,
                steps=steps,
                lossfn=Tensor.sparse_categorical_crossentropy,
                BS=BS,
            )

    print("After pre-training our model..")
    accuracy, Y_test_pred = evaluate(model, X_test, Y_test, BS=BS, return_predict=True)

    print(accuracy)

    mislabeled_counts = _get_mislabeled_counts(Y_test, Y_test_pred)
    _pretty_print_mislabeled_counts(mislabeled_counts)

    worst_class = max(mislabeled_counts, key=lambda k: mislabeled_counts[k])
    print(f"Worst class: {worst_class}")

    print("Lora-izing the model..")

    # Getting the Lora model from the original model without modifying the original one
    lora_model = LoRA.from_module(model, rank=15, inplace=False)

    print(f"Fine-tuning the worst class, {worst_class}..")
    lrs = [1e-5]
    epochss = [1]
    BS = 128

    X_train, Y_train = filter_data_by_class(X_train, Y_train, worst_class)
    for lr, epochs in zip(lrs, epochss):
        optimizer = nn.optim.Adam(lora_model.parameters(), lr=lr)
        for epoch in range(1, epochs + 1):
            train(
                lora_model,
                X_train,
                Y_train,
                optimizer,
                steps=200,
                lossfn=Tensor.sparse_categorical_crossentropy,
                BS=BS,
            )

    print("Here's your fine-tuned model..")
    accuracy, Y_test_pred = evaluate(
        lora_model, X_test, Y_test, BS=BS, return_predict=True
    )

    # checkpoint_file_path = f"examples/checkpoint{accuracy * 1e6:.0f}"
    # lora_model.save(checkpoint_file_path)
    mislabeled_counts = _get_mislabeled_counts(Y_test, Y_test_pred)
    _pretty_print_mislabeled_counts(mislabeled_counts)

    print(lora_model.parameters())

    # Getting a random example to test the model
    x = Tensor.randn(1, 28, 28).reshape(-1)

    # Assert if the values are not all the same and thus I have done something
    assert not np.allclose(
        model(x).numpy(), lora_model(x).numpy()
    ), "The outputs are too close!"
    lora_model.disable_lora()

    # Assert if the values are the same and thus I haven't changed the original model
    assert np.allclose(
        model(x).numpy(), lora_model(x).numpy()
    ), "The outputs are too close!"
    print("Everything works as expected")
