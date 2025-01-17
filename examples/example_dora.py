#!/usr/bin/env python3

import os
import sys

import numpy as np
from tinygrad import Tensor, nn
from utils import (evaluate, fetch_fashion_mnist, get_mislabeled_counts,
                   mix_old_and_new_data, pretty_print_mislabeled_counts, train)

# Get the path of the parent directory
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Add the parent directory to the system path
sys.path.append(parent_dir)

# Now you can import the DoRA module
from dora_tinygrad import DoRA


class TinyNet:
    def __init__(self):
        self.l1 = nn.Linear(784, 128, bias=False)
        self.l2 = nn.Linear(128, 10, bias=False)

    def __call__(self, x):
        x = self.l1(x)
        x = x.leakyrelu()
        x = self.l2(x)
        return x


if __name__ == "__main__":
    # Setting up a random seed
    print("Simulating a pre-trained model, with one epoch..")
    lr = 1e-3
    epochss = 5
    BS = 128
    n_outputs = 10

    X_train, Y_train, X_test, Y_test = fetch_fashion_mnist()
    steps = len(X_train) // BS

    # Define the model
    model = TinyNet()

    # Define loss function
    lossfn = Tensor.sparse_categorical_crossentropy

    # Pre-training the model
    for _ in range(epochss):
        optimizer = nn.optim.Adam(nn.state.get_parameters(model), lr=lr)
        train(model, X_train, Y_train, optimizer, lossfn=lossfn, steps=steps, BS=BS)
        # accuracy, Y_test_pred = evaluate(model, X_test, Y_test, return_predict=True)
        lr /= 1.2
        # print(f"reducing lr to {lr:.7f}")

    print("After pre-training our model..")
    accuracy, Y_test_pred = evaluate(model, X_test, Y_test, BS=BS, return_predict=True)

    print(f"Test set accuracy: {accuracy}")

    mislabeled_counts = get_mislabeled_counts(Y_test, Y_test_pred, n_output=n_outputs)
    pretty_print_mislabeled_counts(mislabeled_counts)

    worst_class = max(mislabeled_counts, key=lambda k: mislabeled_counts[k])
    print(f"Worst class: {worst_class}")

    print("Lora-izing the model..")
    # Getting the Lora model from the original model without modifying the original one
    dora_model = DoRA.from_module(model, rank=8, inplace=False)

    print(f"Fine-tuning the worst class, {worst_class}..")
    lrs = 1e-7
    epochss = 1
    BS = 64

    # Filter to only train on the worst class and make a mixture with the old dataset
    X_train, Y_train = mix_old_and_new_data(X_train, Y_train, worst_class, ratio=0.6)
    steps = len(X_train) // BS

    # Pre-training the model
    for _ in range(epochss):
        optimizer = nn.optim.Adam(dora_model.parameters(), lr=lr)
        # Default loss function is sparse_categorical_crossentropy
        train(dora_model, X_train, Y_train, optimizer, steps=steps, BS=BS)
        accuracy, Y_test_pred = evaluate(
            dora_model, X_test, Y_test, return_predict=True
        )
        lr /= 1.2
        # print(f"reducing lr to {lr:.7f}")

    print("Here's your fine-tuned model..")
    accuracy, Y_test_pred = evaluate(
        dora_model, X_test, Y_test, BS=BS, return_predict=True
    )

    print(f"Test set accuracy after finetuning: {accuracy}")

    # checkpoint_file_path = f"examples/checkpoint{accuracy * 1e6:.0f}"
    # lora_model.save(checkpoint_file_path)
    mislabeled_counts = get_mislabeled_counts(Y_test, Y_test_pred, n_output=n_outputs)
    pretty_print_mislabeled_counts(mislabeled_counts)

    print(dora_model.parameters())

    # Getting a random example to test the model
    x = Tensor.randn(1, 28, 28).reshape(-1)

    # Assert if the values are not all the same and thus I have done something
    assert not np.allclose(
        model(x).numpy(), dora_model(x).numpy()
    ), "The outputs are too close!"
