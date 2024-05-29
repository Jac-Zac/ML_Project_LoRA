#!/usr/bin/env python3
from collections.abc import Callable
from typing import List

import numpy as np
from tinygrad import Tensor, nn

from extra.datasets import fetch_mnist
from extra.training import evaluate, train
from lora_tinygrad import LoRA

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
        self.l1 = nn.Linear(784, 784 * 3, bias=False)
        self.l2 = nn.Linear(784 * 3, 784, bias=False)
        self.l3 = nn.Linear(784, 128, bias=False)
        self.l4 = nn.Linear(128, 10, bias=False)

    def __call__(self, x):
        x = self.l1(x).leakyrelu()
        x = self.l2(x).leakyrelu()
        x = self.l3(x).leakyrelu()
        x = self.l4(x)
        return x


# class TinyNet:
#     def __init__(self):
#         self.l1 = nn.Linear(784, 128, bias=False)
#         self.l2 = nn.Linear(128, 10, bias=False)
#
#     def __call__(self, x):
#         x = self.l1(x)
#         x = x.leakyrelu()
#         x = self.l2(x)
#         return x
#

if __name__ == "__main__":
    print("Simulating a pre-trained model, with one epoch..")
    lr = 1e-3
    epochss = 1
    BS = 128

    X_train, Y_train, X_test, Y_test = fetch_mnist()

    steps = len(X_train) // BS
    x = Tensor.randn(1, 28, 28).reshape(-1)

    model = TinyNet()
    print(model(x).numpy())

    lora_model = LoRA.from_module(model, rank=8, inplace=False)
    print(f"\nPrinting model: {nn.state.get_state_dict(model)}\n")
    print(f"\nPrinting lora_model: {nn.state.get_state_dict(lora_model)}\n")

    original_parameters = sum(p.numel() for p in nn.state.get_parameters(model))
    lora_parameters = sum(p.numel() for p in lora_model.parameters())

    print(
        f"We can see that: {original_parameters = }, lora_params = {lora_parameters = }, thus the model needs to update {(lora_parameters / original_parameters) * 100:.2f}% of the original parameters in this example.\n"
    )

    # original_mode = lora_model.remove_lora(inplace=False)
    # # print(original_mode(x).numpy)
    #
    # print(nn.state.get_state_dict(lora_model))

    for _ in range(epochss):
        optimizer = nn.optim.Adam(lora_model.parameters(), lr=lr)
        train(lora_model, X_train, Y_train, optimizer, steps=steps, BS=BS)
        accuracy, Y_test_pred = evaluate(
            lora_model, X_test, Y_test, return_predict=True
        )
        lr /= 1.2

    # Get predictions for the lora model
    print("Lora model predictions:")
    print(lora_model(x).numpy())
    # Remove LoRA weights
    # original_model = lora_model.remove_lora(inplace=False)  # default: inplace=False
    # lora_model.remove_lora(inplace=True)  # default: inplace=False

    print("Model predictions after I disable LoRA")
    lora_model.disable_lora()
    print(lora_model(x).numpy())

    original_mode = lora_model.remove_lora(inplace=False)
    print(original_mode(x).numpy)

    print(nn.state.get_state_dict(lora_model))
    # assert isinstance(original_mode, TinyNet)
    #
    # # Get predictions for the original model
    # print(original_model(x).numpy())
    #
    # print(nn.state.get_state_dict(original_model))

    """

    # Merge LoRA weights into the original model.
    new_model = lora_model.merge_lora(inplace=False)  # default: inplace=False

    # print(f"\nPrinting merged model: {nn.state.get_state_dict(new_model)}\n")

    print("New merged model predictions")
    print(new_model(x).numpy())

    # NOTE: new_model has the same type as the original model!  Inference is just as fast as in the original model.
    assert isinstance(new_model, TinyNet)
    """
