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


def get_layers_dict(obj, layer_name=""):
    """
    Return the layers of a network in a dictionary with layer_name and layer
    """

    # NOTE: custom function added for ease of use
    layers = {}

    # layers[layer_name] = obj  # Store the current object with its layer name as key
    layers[layer_name.lstrip(".")] = (
        obj  # Store the current object with its layer name as key without the leading "."
    )

    if hasattr(obj, "_asdict"):
        for key, value in obj._asdict().items():
            layers.update(get_layers_dict(value, f"{layer_name}.{key}"))
    elif isinstance(obj, OrderedDict):
        for key, value in obj.items():
            layers.update(get_layers_dict(value, f"{layer_name}.{key}"))
    elif hasattr(obj, "__dict__"):
        for key, value in obj.__dict__.items():
            layers.update(get_layers_dict(value, f"{layer_name}.{key}"))
    elif isinstance(obj, (list, tuple)):
        for i, x in enumerate(obj):
            layers.update(get_layers_dict(x, f"{layer_name}.{i}"))
    elif isinstance(obj, dict):
        for k, v in obj.items():
            layers.update(get_layers_dict(v, f"{layer_name}.{k}"))

    return layers


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

    # TODO: Improve code readability
    def __call__(self, x: Tensor, *args, **kwargs) -> Tensor:

        # Run the operation without gradient if there is no lora and gradient is enable else run it with the gradient
        Tensor.no_grad = not ((self.lora_module is None) and (Tensor.no_grad == False))
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
        layers_dict = get_layers_dict(self)

        for layer in layers_dict.values():
            if isinstance(layer, BaseLoRAModule):
                parameters.append(getattr(layer, "in_proj"))
                parameters.append(getattr(layer, "out_proj"))

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
        out_size, in_size = nn.state.get_state_dict(module)["weight"].shape

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
        for name, layer in get_layers_dict(out).items():

            if isinstance(layer, nn.Linear):
                # when you encounter a known layer that can be made into a LoRA layer do it
                setattr(out, name, LoRA._from_linear(layer, rank))

        # for layer in get_layers(target_module, nn.Linear):
        #     # when you encounter a known layer that can be made into a LoRA layer do it
        #     target_module.layer = LoRA._from_linear(layer, rank)  # type: ignore

        # elif isinstance(sub_layer, nn.Embedding):
        #     setattr(target_module, name, LoRA._from_embedding(sub_layer, rank))
        # elif isinstance(sub_layer, nn.MultiheadAttention):
        #     setattr(target_module, name, LoRA._from_multihead_attention(sub_layer, rank))  # type: ignore

        # Return the new (or modified) module
        return LoRA(out, None, enabled=enabled)

    # @property
    # def weight(self) -> Tensor:
    #     if not hasattr(self.module, "weight"):
    #         raise AttributeError("Module has no attribute 'weight'")
    #
    #     if self.enabled and self.lora_module is not None:
    #         assert hasattr(self.lora_module, "weight")
    #         return self.module.weight + self.lora_module.weight
    #     else:
    #         return self.module.weight

    # @property
    # def bias(self) -> Optional[Tensor]:
    #     if not hasattr(self.module, "bias"):
    #         return None
    #
    #     if (
    #         self.enabled
    #         and self.lora_module is not None
    #         and hasattr(self.lora_module, "bias")
    #     ):
    #         return self.module.bias + self.lora_module.bias
    #     else:
    #         return self.module.bias


def enable_lora(module: Union[Type, LoRA]) -> None:
    # Enable only the Lora layers through all the layers
    for layer in get_layers_dict(module).values():
        if isinstance(layer, LoRA):
            layer.enabled = True


def disable_lora(module: Union[Type, LoRA]) -> None:
    # Disable only the Lora layers through all the layers
    for layer in get_layers_dict(module).values():
        if isinstance(layer, LoRA):
            layer.enabled = False


# TODO: needs to be checked
def merge_lora(module: Union[Type, LoRA], inplace: bool = False):
    """Merge all LoRA modules from the model."""
    out = module if inplace else copy.deepcopy(module)

    for name, layer in get_layers_dict(out).items():
        if isinstance(layer, LoRA):
            print(name)
            if layer.lora_module is None:
                setattr(layer, name, layer.module)
            else:
                setattr(
                    layer, name, layer.lora_module.merge(layer.module, inplace=inplace)
                )

    return out


def remove_lora(module: Union[Type, LoRA], inplace: bool = False):
    """Remove all LoRA modules from the model."""
    out = module if inplace else copy.deepcopy(module)

    for name, layer in get_layers_dict(out).items():

        # if isinstance(layer, BaseLoRAModule):
        #     print("Deliting")
        #     print(getattr(layer, "in_proj"))
        #     delattr(layer, "in_proj")
        #     delattr(layer, "out_proj")

        if isinstance(layer, LoRA):
            sub_module = out
            # # Get the correct attribute
            for attr in name.split(".")[:-1]:
                sub_module = getattr(sub_module, attr)

            sub_module = layer.module
        else:
            sub_module = out
            # # Get the correct attribute
            for attr in name.split(".")[:-1]:
                sub_module = getattr(sub_module, attr)

            sub_module = layer

    return out
