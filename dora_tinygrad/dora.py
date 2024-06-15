from __future__ import annotations

import copy
from typing import Optional, Type, Union

from tinygrad import Tensor, nn

from .modules.base import BaseDoRAModule
from .modules.linear import LinearDoRAModule


def get_nested_attr(obj, attr):
    """Recursively get the nested attribute."""
    for attribute in attr:
        obj = getattr(obj, attribute)
    return obj


class DoRA:
    def __init__(
        self,
        module,
        dora_module: Optional[BaseDoRAModule],
        enabled: bool = True,
    ):
        super().__init__()
        self.module = module
        self.dora_module = dora_module
        self.enabled = enabled and dora_module is not None

        if not enabled:
            self.disable_dora()
        else:
            self.enable_dora()

    def __call__(self, x: Tensor, *args, **kwargs) -> Tensor:
        # Disable gradient if DoRA is not present and gradients are enabled, otherwise enable gradients
        Tensor.no_grad = not (self.dora_module is None and not Tensor.no_grad)
        y = self.module(x, *args, **kwargs)
        # Enable the gradient again
        Tensor.no_grad = False

        # If DoRA is enable also add the DoRA
        if self.enabled and self.dora_module is not None:
            y = y + self.dora_module(x)

        return y

    # I have to only return the dora parameters
    def parameters(self):
        parameters = []

        for name in nn.state.get_state_dict(self):
            # Split the attributes
            name = name.split(".")
            # Layers is all of the attribute but the last one
            layer = get_nested_attr(self, name[:-1])

            if isinstance(layer, BaseDoRAModule):
                # NOTE: name[-1] for linear layer will be the m, out_proj, in_proj for example
                parameters.append(getattr(layer, name[-1]))

        return parameters

    def enable_dora(self) -> None:
        return enable_dora(self)  # type: ignore

    def disable_dora(self) -> None:
        return disable_dora(self)  # type: ignore

    def remove_dora(self, inplace: bool = False):
        """Remove all DoRA modules from the model."""
        return remove_dora(self, inplace=inplace)  # type: ignore

    def merge_dora(self: DoRA, inplace: bool = False):
        return merge_dora(self, inplace=inplace)  # type: ignore

    @classmethod
    def _from_linear(cls, module: nn.Linear, rank: int) -> DoRA:
        # Get the input and output size
        out_size, in_size = module.weight.shape

        # Do not track the gradient for the creation of this magnitude vector
        # Tensor.no_grad = True
        # magnitude = Tensor.sqrt(Tensor.sum(module.weight**2, axis=0, keepdim=True))
        # This reqiures grad
        # HACK: This requires grad but should not be connected with the weights
        magnitude = Tensor.sqrt(Tensor.sum(module.weight**2, axis=1, keepdim=False))
        # Tensor.no_grad = False

        # Initialize a new DoRA layer
        dora_module = LinearDoRAModule(magnitude, in_size, out_size, rank=rank)
        return DoRA(module, dora_module)

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
                # when you encounter a known layer that can be made into a DoRA layer do it
                setattr(out, name[0], DoRA._from_linear(layer, rank))

            # Here you could potentially add options for other layers such as for MultiheadAttention

        # Return the new (or modified) module
        return DoRA(out, None, enabled=enabled)

    @property
    def weight(self) -> Tensor:
        if not hasattr(self.module, "weight"):
            raise AttributeError("Module has no attribute 'weight'")

        if self.enabled and self.dora_module is not None:
            assert hasattr(self.dora_module, "weight")
            return self.module.weight + self.dora_module.weight
        else:
            return self.module.weight

    @property
    def bias(self) -> Optional[Tensor]:
        if not hasattr(self.module, "bias"):
            return None

        if (
            self.enabled
            and self.dora_module is not None
            and hasattr(self.dora_module, "bias")
        ):
            return self.module.bias + self.dora_module.bias
        else:
            return self.module.bias


def enable_dora(module: Union[Type, DoRA]) -> None:
    # Enable only the Dora layers through all the layers
    for name in nn.state.get_state_dict(module):
        layer = get_nested_attr(module, name.split(".")[:-2])

        if isinstance(layer, DoRA):
            layer.enabled = True


def disable_dora(module: Union[Type, DoRA]) -> None:
    # Enable only the Dora layers through all the layers
    for name in nn.state.get_state_dict(module):
        layer = get_nested_attr(module, name.split(".")[:-2])

        if isinstance(layer, DoRA):
            layer.enabled = False


def merge_dora(module: Union[Type, "DoRA"], inplace: bool = False):
    return NotImplemented


def remove_dora(module: Union[Type, DoRA], inplace: bool = False):
    return NotImplemented
