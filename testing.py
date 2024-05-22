#!/usr/bin/env python3

import numpy as np
from tinygrad import Device, Tensor, nn

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


# if __name__ == "__main__":
#     print("Simulating a pre-trained model, with one epoch..")
#     lrs = [1e-3]
#     epochss = [1]
#     BS = 128
#
#     X_train, Y_train, X_test, Y_test = fetch_mnist()
#
#     steps = len(X_train) // BS
#     lossfn = lambda out, y: out.sparse_categorical_crossentropy(y)
#
#     model = TinyNet()

# for lr, epochs in zip(lrs, epochss):
#     optimizer = nn.optim.Adam(nn.state.get_parameters(model), lr=lr)
#     for epoch in range(1, epochs + 1):
#         train(model, X_train, Y_train, optimizer, steps=steps, lossfn=lossfn, BS=BS)
#
#     print("After pre-training our model..")
#     accuracy, Y_test_pred = evaluate(
#         model, X_test, Y_test, BS=BS, return_predict=True
#     )
#
#     print(accuracy)


# Super simple linear model
model = nn.Linear(784, 2, bias=False)

# model = TinyNet()

# Apply LoRA weighst to the model
lora_model = LoRA.from_module(model, rank=5)

print(f"\nPrinting model: {nn.state.get_state_dict(model)}\n")
print(f"Printing lora model: {nn.state.get_state_dict(lora_model)}\n")

# Train or predict as usual.
x = Tensor.randn(1, 28, 28).reshape(-1)

y = model(x)
# Print the predcitions
print(f"Printing model Output: {y.numpy() = }")

y_lora = lora_model(x)
# Print the predcitions
print(f"Printing model Output: {y_lora.numpy() = }")

# # compute loss, backprop, etc...

# # Merge LoRA weights into the original model.
# new_model = lora_model.merge_lora(inplace=False)  # default: inplace=False

# NOTE: new_model has the same type as the original model!  Inference is just as fast as in the original model.
# assert isinstance(new_model, ResNet)
