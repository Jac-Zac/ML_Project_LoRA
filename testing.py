#!/usr/bin/env python3

from tinygrad import nn
from tinygrad.tensor import Tensor

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


# Wrap your model with LoRA
model = TinyNet()
# print(nn.state.get_parameters(model))
# print(nn.state.get_state_dict(model))
lora_model = LoRA.from_module(model, rank=5)

# print(model)
# LoRA(
#   (module): ResNet(
#     (conv1): LoRA(
#       (module): Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
#       (lora_module): Conv2dLoRAModule(
#         (in_conv): Conv2d(3, 5, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
#         (out_conv): Conv2d(5, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
#         (dropout): Dropout(p=0.0, inplace=False)
#       )
#     )
#     (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#     (relu): ReLU(inplace=True)
# ...

# Train or predict as usual.
x = Tensor.randn(1, 28, 28).reshape(-1)

y = model(x)
# Print the predcitions
print(f"Printing model Output: {y.numpy() = }")

y = lora_model(x)


# # compute loss, backprop, etc...
#
# # Merge LoRA weights into the original model.
# new_model = lora_model.merge_lora(inplace=False)  # default: inplace=False

# NOTE: new_model has the same type as the original model!  Inference is just as fast as in the original model.
# assert isinstance(new_model, ResNet)
