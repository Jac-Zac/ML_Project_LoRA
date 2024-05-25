#!/usr/bin/env python3

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


if __name__ == "__main__":
    print("Simulating a pre-trained model, with one epoch..")
    lrs = [1e-3]
    epochss = [1]
    BS = 128

    X_train, Y_train, X_test, Y_test = fetch_mnist()

    steps = len(X_train) // BS
    x = Tensor.randn(1, 28, 28).reshape(-1)

    model = TinyNet()
    print(model(x).numpy())

    lora_model = LoRA.from_module(model, rank=5, inplace=True)

    for lr, epochs in zip(lrs, epochss):
        optimizer = nn.optim.Adam(lora_model.parameters(), lr=lr)
        for epoch in range(1, epochs + 1):
            train(
                lora_model,
                X_train,
                Y_train,
                optimizer,
                steps=steps,
                lossfn=Tensor.sparse_categorical_crossentropy,
                BS=BS,
            )

        print("After pre-training our model..")
        accuracy, Y_test_pred = evaluate(
            model, X_test, Y_test, BS=BS, return_predict=True
        )

        print(accuracy)

    # Get predictions for the lora model
    print("Lora model predictions:")
    print(lora_model(x).numpy())
    # Remove LoRA weights
    # original_model = lora_model.remove_lora(inplace=False)  # default: inplace=False
    # lora_model.remove_lora(inplace=True)  # default: inplace=False

    print("Model predictions after I disable LoRA")
    lora_model.disable_lora()
    print(lora_model(x).numpy())
    #
    # print(nn.state.get_state_dict(lora_model))
    #
    # # Get predictions for the original model
    # print(original_model(x).numpy())
    #
    # print(nn.state.get_state_dict(original_model))
