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


if __name__ == "__main__":
    print("Simulating a pre-trained model, with one epoch..")
    lrs = [1e-3]
    epochss = [1]
    BS = 128

    X_train, Y_train, X_test, Y_test = fetch_mnist()

    steps = len(X_train) // BS
    lossfn = lambda out, y: out.sparse_categorical_crossentropy(y)
    x = Tensor.randn(1, 28, 28).reshape(-1)

    model = TinyNet()
    print(model(x).numpy())

    print(f"\nPrinting model: {nn.state.get_state_dict(model)}\n")
    lora_model = LoRA.from_module(model, rank=5)

    print(f"\nPrinting lora_model: {nn.state.get_state_dict(model)}\n")

    y = lora_model(x)
    print(lora_model(x).numpy())
    #
    lora_model.disable_lora()
    print(lora_model(x).numpy())
    # print(y.requires_grad)
    #
    # # # Re-enable
    # lora_model.enable_lora()
    # y = lora_model(x)
    # print(y.requires_grad)

    #
    # for lr, epochs in zip(lrs, epochss):
    #     # optimizer = nn.optim.Adam(nn.state.get_parameters(model), lr=lr)
    #     optimizer = nn.optim.Adam(
    #         [
    #             model.l1.lora_module.in_proj,
    #             model.l1.lora_module.out_proj,
    #             model.l2.lora_module.in_proj,
    #             model.l2.lora_module.out_proj,
    #         ],
    #         lr=lr,
    #     )
    #     for epoch in range(1, epochs + 1):
    #         train(model, X_train, Y_train, optimizer, steps=steps, lossfn=lossfn, BS=BS)
    #
    #     print("After pre-training our model..")
    #     accuracy, Y_test_pred = evaluate(
    #         model, X_test, Y_test, BS=BS, return_predict=True
    #     )
    #
    #     print(accuracy)
    #
    # # Get predictions for the lora model
    # print(lora_model(x).numpy())
    # Remove LoRA weights
    # original_model = lora_model.remove_lora(inplace=False)  # default: inplace=False
    # lora_model.remove_lora(inplace=True)  # default: inplace=False
    #
    # print(lora_model(x).numpy())
    #
    # print(nn.state.get_state_dict(lora_model))
    #
    # # # Get predictions for the original model
    # print(original_model(x).numpy())
    #
    # print(nn.state.get_state_dict(original_model))

# with Tensor.train():
#     for step in range(1000):
#         # random sample a batch
#         samp = np.random.randint(0, X_train.shape[0], size=(64))
#         batch = Tensor(X_train[samp], requires_grad=False)
#         # get the corresponding labels
#         labels = Tensor(Y_train[samp])
#
#         # forward pass
#         out = lora_model(batch)
#
#         # compute loss
#         loss = Tensor.sparse_categorical_crossentropy(out, labels)
#
#         # zero gradients
#         opt.zero_grad()
#
#         # backward pass
#         loss.backward()
#
#         # update parameters
#         opt.step()
#
#         # calculate accuracy
#         pred = out.argmax(axis=-1)
#         acc = (pred == labels).mean()
#
#         if step % 100 == 0:
# print(f"Step {step+1} | Loss: {loss.numpy()} | Accuracy: {acc.numpy()}")


# Super simple linear model
# model = nn.Linear(784, 2, bias=False)

# model = TinyNet()

# print(f"\nPrinting model: {nn.state.get_state_dict(model)}\n")
#
# # Apply LoRA weighst to the model
# lora_model = LoRA.from_module(model, rank=5)
#
# print(f"Printing lora model: {nn.state.get_state_dict(lora_model)}\n")
#
# # Train or predict as usual.
# x = Tensor.randn(1, 28, 28).reshape(-1)
#
# y = model(x)
# # Print the predcitions
# print(f"Printing model Output: {y.numpy() = }")
#
# y_lora = lora_model(x)
# # Print the predcitions
# print(f"Printing model Output: {y_lora.numpy() = }")
#
# # compute loss, backprop, etc...

# # Merge LoRA weights into the original model.
# new_model = lora_model.merge_lora(inplace=False)  # default: inplace=False

# NOTE: new_model has the same type as the original model!  Inference is just as fast as in the original model.
# assert isinstance(new_model, ResNet)
