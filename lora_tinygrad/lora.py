from __future__ import annotations

import copy
from typing import Optional, Type, Union

from tinygrad import Tensor, nn

from .modules.base import BaseLoRAModule
from .modules.linear import LinearLoRAModule


def get_nested_attr(obj, attr):
    """Recursively get the nested attribute."""
    for attribute in attr:
        obj = getattr(obj, attribute)
    return obj


class LoRA:
    def __init__(
        self,
        module,
        lora_module: Optional[BaseLoRAModule],
        enabled: bool = True,
    ):
        super().__init__()
        self.module = module
        self.lora_module = lora_module
        self.enabled = enabled and lora_module is not None

        if not enabled:
            self.disable_lora()
        else:
            self.enable_lora()

    def __call__(self, x: Tensor, *args, **kwargs) -> Tensor:
        # Disable gradient if LORA is not present and gradients are enabled, otherwise enable gradients
        Tensor.no_grad = not (self.lora_module is None and not Tensor.no_grad)
        y = self.module(x, *args, **kwargs)
        # Enable the gradient again
        Tensor.no_grad = False

        # IF lora is enable also add the lora
        if self.enabled and self.lora_module is not None:
            y = y + self.lora_module(x)

        return y

    # I have to only return the lora parameters
    def parameters(self):
        parameters = []

        for name in nn.state.get_state_dict(self):
            # Split the attributes
            name = name.split(".")
            # Layers is all of the attribute but the last one
            layer = get_nested_attr(self, name[:-1])

            if isinstance(layer, BaseLoRAModule):
                # NOTE: name[-1] for linear layer will be the out_proj and in_proj for example
                parameters.append(getattr(layer, name[-1]))

        return parameters

    def enable_lora(self) -> None:
        return enable_lora(self)  # type: ignore

    def disable_lora(self) -> None:
        return disable_lora(self)  # type: ignore

    def remove_lora(self, inplace: bool = False):
        """Remove all LoRA modules from the model."""
        return remove_lora(self, inplace=inplace)  # type: ignore

    def merge_lora(self: LoRA, inplace: bool = False):
        return merge_lora(self, inplace=inplace)  # type: ignore

    @classmethod
    def _from_linear(cls, module: nn.Linear, rank: int) -> LoRA:
        # Get the input and output size
        out_size, in_size = module.weight.shape

        # Initialize a new LoRA layer
        lora_module = LinearLoRAModule(in_size, out_size, rank=rank)
        return LoRA(module, lora_module)

    # Potentially you could implement something like this
    # @classmethod
    # def _from_multihead_attention(cls, module: nn.MultiheadAttention, rank: int) -> MultiheadAttentionLoRA:

    # Actual implementation of from module
    @classmethod
    def from_module(
        cls,
        module,
        rank: int,
        inplace: bool = True,
        enabled: bool = True,
    ):
        # If not inplace, create a deepcopy of the module to avoid modifying the original
        out = module if inplace else copy.deepcopy(module)

        # Get all of the layers
        for name in nn.state.get_state_dict(out):
            name = name.split(".")[:-1]
            layer = get_nested_attr(out, name)

            if isinstance(layer, nn.Linear):
                # when you encounter a known layer that can be made into a LoRA layer do it
                setattr(out, name[0], LoRA._from_linear(layer, rank))

            # Here you could potentially add options for other layers such as for MultiheadAttention

        # Return the new (or modified) module
        return LoRA(out, None, enabled=enabled)

    @property
    def weight(self) -> Tensor:
        if not hasattr(self.module, "weight"):
            raise AttributeError("Module has no attribute 'weight'")

        if self.enabled and self.lora_module is not None:
            assert hasattr(self.lora_module, "weight")
            return self.module.weight + self.lora_module.weight
        else:
            return self.module.weight

    @property
    def bias(self) -> Optional[Tensor]:
        if not hasattr(self.module, "bias"):
            return None

        if (
            self.enabled
            and self.lora_module is not None
            and hasattr(self.lora_module, "bias")
        ):
            return self.module.bias + self.lora_module.bias
        else:
            return self.module.bias


def enable_lora(module: Union[Type, LoRA]) -> None:
    # Enable only the Lora layers through all the layers
    for name in nn.state.get_state_dict(module):
        layer = get_nested_attr(module, name.split(".")[:-2])

        if isinstance(layer, LoRA):
            layer.enabled = True


def disable_lora(module: Union[Type, LoRA]) -> None:
    # Enable only the Lora layers through all the layers
    for name in nn.state.get_state_dict(module):
        layer = get_nested_attr(module, name.split(".")[:-2])

        if isinstance(layer, LoRA):
            layer.enabled = False


def merge_lora(module: Union[Type, "LoRA"], inplace: bool = False):
    out = module if inplace else copy.deepcopy(module)
    out = out.module

    for name in nn.state.get_state_dict(out):
        name = name.split(".")
        # Get layer and parent_layer
        parent_layer = get_nested_attr(out, name[:-2])

        if hasattr(parent_layer, name[-2]):
            layer = getattr(parent_layer, name[-2])
        else:
            # Since it has already been removed
            continue

        if isinstance(layer, BaseLoRAModule):
            if parent_layer.lora_module is not None:
                # Check if already merged
                parent_layer.module = parent_layer.lora_module.merge(
                    parent_layer.module, inplace=inplace
                )

                setattr(out, name[0], parent_layer.module)

    return out


def remove_lora(module: Union[Type, LoRA], inplace: bool = False):
    """Remove all LORA modules from the model."""
    out = module if inplace else copy.deepcopy(module)

    # Remove LoRA parent module
    out = out.module

    # List to save the attributes to modify
    attributes_to_modify = []

    for name in nn.state.get_state_dict(out):
        # Get the layer based on the name recursively getting the correct attribuet
        layer = get_nested_attr(out, name.split(".")[:-1])
        if isinstance(layer, nn.Linear):
            # Add the attribute name to the list to be modified
            attributes_to_modify.append((name.split(".module")[0], layer))

    # Modify the attributes outside the loop to avoid issues with getattr
    for name, layer in attributes_to_modify:
        # NOTE: This is an example:
        # And we set it equal to the current layer:
        # out.l1 = out.l1.module
        setattr(out, name, layer)

    return out
