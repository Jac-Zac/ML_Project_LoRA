#!/usr/bin/env python3
import numpy as np
from tinygrad import Tensor, nn

from dora_tinygrad import DoRA
from examples.utils import evaluate, fetch_mnist, train

# FIX:: Sequential doesn't work because the projection layers are placed after the list

# class TinyNet:
#     def __init__(self):
#         self.layers: List[Callable[[Tensor], Tensor]] = [
#             nn.Linear(784, 784 * 4, bias=True),
#             Tensor.leakyrelu,
#             nn.Linear(784 * 4, 784, bias=False),
#             Tensor.leakyrelu,
#             nn.Linear(784, 128, bias=False),
#             Tensor.leakyrelu,
#             nn.Linear(128, 10, bias=False),
#         ]
#
#     def __call__(self, x: Tensor) -> Tensor:
#         return x.sequential(self.layers)


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
    print("Simulating a pre-trained model, with one epoch..")
    lr = 1e-3
    epochs = 1
    BS = 128

    X_train, Y_train, X_test, Y_test = fetch_mnist()

    steps = len(X_train) // BS
    model = TinyNet()

    # Print the model output to realize it at least once
    x = Tensor.randn(1, 28, 28).reshape(-1)
    print("\033[92mModel predictions when not modified\033[0m")
    print(model(x).numpy())

    dora_model = DoRA.from_module(model, rank=8, inplace=False)
    print(f"\nPrinting model: {nn.state.get_state_dict(model)}\n")
    print(f"\nPrinting dora_model: {nn.state.get_state_dict(dora_model)}\n")

    original_parameters = sum(p.numel() for p in nn.state.get_parameters(model))
    dora_parameters = sum(p.numel() for p in dora_model.parameters())

    print(
        f"We can see that: {original_parameters = }, dora_params = {dora_parameters = }, thus the model needs to update {(dora_parameters / original_parameters) * 100:.2f}% of the original parameters in this example.\n"
    )

    for _ in range(epochs):
        optimizer = nn.optim.Adam(dora_model.parameters(), lr=lr)
        train(dora_model, X_train, Y_train, optimizer, steps=steps, BS=BS)
        accuracy, Y_test_pred = evaluate(
            dora_model, X_test, Y_test, return_predict=True
        )
        lr /= 1.2

    # Testing things out
    # Defining the random tensor to pass through the models

    # NOTE: If I only compute the output of the model here it never realized
    print("\033[92mModel predictions when not modified\033[0m")
    print(model(x).numpy())

    # Get predictions for the dora model
    print("\033[94mdora model predictions:\033[0m")
    print(dora_model(x).numpy())

    print("\033[92mModel predictions after I disable dora\033[0m")
    dora_model.disable_dora()
    print(dora_model(x).numpy())

    print("\033[93mModel predictions after I re-enable dora\033[0m")
    dora_model.enable_dora()
    print(dora_model(x).numpy())

    # Merge dora
    new_model = dora_model.merge_dora(inplace=False)

    print("\033[91mModel prediction after I merged dora\033[0m")
    print(new_model(x).numpy())

    print(
        f"\nState dict after I have removed dora: {nn.state.get_state_dict(new_model)}"
    )

    # Remove model
    # original_model = dora_model.remove_dora(inplace=False)
    #
    # print("\033[91mModel prediction after I removed dora\033[0m")
    # print(original_model(x).numpy())
    #
    # print(
    #     f"\nState dict after I have removed dora: {nn.state.get_state_dict(original_model)}"
    # )

    #
    # NOTE: new_model has the same type as the original model!  Inference is just as fast as in the original model.
    # assert isinstance(new_model, TinyNet)
    # """
