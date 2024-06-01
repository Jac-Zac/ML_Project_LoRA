from __future__ import annotations

import copy
from collections import OrderedDict
from typing import (Any, Dict, Generic, Literal, Optional, Tuple, Type, Union,
                    overload)

from tinygrad import Tensor, nn

from .modules.base import BaseLoRAModule
from .modules.linear import LinearLoRAModule

# from lora_tinygrad.modules.attention import MultiheadAttentionLoRAModule

# from tinygrad import Tensor, nn
#                                        Conv3dLoRAModule, ConvType)
# from lora_tinygrad.modules.embedding import EmbeddingLoRAModule
# from lora_tinygrad.modules.linear import LinearLoRAModule


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

    # def remove_lora(self, inplace: bool = False) -> ModuleType:
    def remove_lora(self, inplace: bool = False):
        """Remove all LoRA modules from the model."""
        return remove_lora(self, inplace=inplace)  # type: ignore

    # def merge_lora(self: LoRA, inplace: bool = False) -> ModuleType:
    def merge_lora(self: LoRA, inplace: bool = False):
        return merge_lora(self, inplace=inplace)  # type: ignore

    @classmethod
    def _from_linear(cls, module: nn.Linear, rank: int) -> LoRA:
        # Get the input and output size
        out_size, in_size = module.weight.shape

        # Initialize a new LoRA layer
        lora_module = LinearLoRAModule(in_size, out_size, rank=rank)
        return LoRA(module, lora_module)

    # @classmethod
    # def _from_embedding(cls, module: nn.Embedding, rank: int) -> LoRA[nn.Embedding]:
    #     num_embeddings, embedding_dim = module.weight.shape
    #     device = module.weight.device
    #     dtype = module.weight.dtype
    #     lora_module = EmbeddingLoRAModule(
    #         num_embeddings=num_embeddings,
    #         embedding_dim=embedding_dim,
    #         rank=rank,
    #         device=device,
    #         dtype=dtype,
    #     )
    #     return LoRA(module, lora_module)
    #
    # @classmethod
    # def _from_multihead_attention(
    #     cls, module: nn.MultiheadAttention, rank: int
    # ) -> MultiheadAttentionLoRA:
    #     device = module.out_proj.weight.device
    #     dtype = module.out_proj.weight.dtype
    #     lora_module = MultiheadAttentionLoRAModule(
    #         embed_dim=module.embed_dim,
    #         num_heads=module.num_heads,
    #         rank=rank,
    #         bias=False,  # TODO: support bias
    #         kdim=module.kdim,
    #         vdim=module.vdim,
    #         device=device,
    #         dtype=dtype,
    #     )
    #     return MultiheadAttentionLoRA(module, lora_module)

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

            # elif isinstance(sub_layer, nn.Embedding):
            #     setattr(target_module, name, LoRA._from_embedding(sub_layer, rank))
            # elif isinstance(sub_layer, nn.MultiheadAttention):
            #     setattr(target_module, name, LoRA._from_multihead_attention(sub_layer, rank))  # type: ignore

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


# TODO: needs to be checked
def merge_lora(module: Union[Type, LoRA], inplace: bool = False):
    import numpy as np

    """Merge all LoRA modules from the model."""
    out = module if inplace else copy.deepcopy(module)

    # Remove LoRA parent module
    out = out.module

    # List to save the attributes to modify
    attributes_to_modify = []

    for name in nn.state.get_state_dict(out):
        layer = get_nested_attr(out, name.split(".")[:-1])
        if isinstance(layer, BaseLoRAModule):
            layer = get_nested_attr(out, name.split(".")[:-2])
            # print(layer)
            if layer.lora_module is None:
                # print("Got here")
                # attributes_to_modify.append(
                #     (
                #         name.split(".module")[0],
                #         get_nested_attr(out, name.split(".")[:-1]),
                #     )
                # )
                pass
            else:
                print(name.split(".")[:-2])
                # layer.module = layer.lora_module.merge(layer.module, inplace=inplace)
                layer.module = layer.lora_module.merge(layer.module, inplace=inplace)

                # NOTE: After this I have to remove the lora_module

    # for name in nn.state.get_state_dict(out):
    #     layer = get_nested_attr(out, name.split(".")[:-1])
    #     print(name.split(".")[:-1])
    #
    #     if isinstance(layer, nn.Linear):
    #         print(f"Prining weight: {layer.weight.numpy()}")
    #         # Add the attribute name to the list to be modified
    #         attributes_to_modify.append((name.split(".module")[0], layer))
    #         # break

    for name, layer in attributes_to_modify:
        # NOTE: This is an example:
        # And we set it equal to the current layer:
        # out.l1 = out.l1.module
        setattr(out, name, layer)

    # end = out.l1.weight.numpy()
    # assert not np.allclose(start, end)
    return out


def remove_lora(module: Union[Type, LoRA], inplace: bool = False):
    """Remove all LoRA modules from the model."""
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
