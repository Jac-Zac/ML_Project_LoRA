from __future__ import annotations

from copy import deepcopy
from itertools import chain
from typing import (Generic, Iterable, Literal, Optional, Tuple, Type, Union,
                    overload)

from tinygrad import Tensor, nn

from .modules.base import BaseLoRAModule
from .modules.linear import LinearLoRAModule


def get_state_layers_names(model):
    # Get everything before .weight
    # return [name.split(".weight")[0] for name in nn.state.get_state_dict(model)]
    return [name.split(".")[0] for name in nn.state.get_state_dict(model)]


# from lora_tinygrad.modules.attention import MultiheadAttentionLoRAModule

# from tinygrad import Tensor, nn
#                                        Conv3dLoRAModule, ConvType)
# from lora_tinygrad.modules.embedding import EmbeddingLoRAModule
# from lora_tinygrad.modules.linear import LinearLoRAModule


class LoRA:
    def __init__(
        self,
        module,
        lora_module: Optional[BaseLoRAModule],
        enabled: bool = True,
    ):
        super().__init__()
        # self.module = module.eval()
        self.module = module
        self.lora_module = lora_module
        self.enabled = enabled and lora_module is not None

        # if not enabled:
        #     self.disable_lora()
        # else:
        #     self.enable_lora()

    def __call__(self, x: Tensor, *args, **kwargs) -> Tensor:

        # Run the operation without gradient if there is no lora and gradient is enable else run it with the gradient
        Tensor.no_grad = not ((self.lora_module is None) and (Tensor.no_grad == False))
        y = self.module(x, *args, **kwargs)
        # Enable the gradient again
        Tensor.no_grad = False

        # IF lora is enable also add the lora
        if self.enabled and self.lora_module is not None:
            print("Lora is enabled")
            y = y + self.lora_module(x)

        return y

    # I have to only return the lora parameters

    # def parameters(self) -> Iterable[nn.Parameter]:  # type: ignore[override]
    #     def _get_lora_parameters(module: nn.Module):
    #         parameters = chain(
    #             *[_get_lora_parameters(child) for child in module.children()]
    #         )
    #         if isinstance(module, LoRA) and module.lora_module is not None:
    #             parameters = chain(parameters, module.lora_module.parameters())
    #
    #         return parameters
    #
    #     return _get_lora_parameters(self)

    # Possibly get lora
    # get_state_layers_names similar but search where in name = 'lora_module'

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

    # Abstract implemenataion of from module
    # _______________________________________
    # @overload
    # @classmethod
    # def from_module(
    #     cls,
    #     module,
    #     rank: int,
    #     enabled: bool = True,
    #     is_root: Literal[True] = True,
    # ) -> LoRA: ...
    #
    # @overload
    # @classmethod
    # def from_module(
    #     cls,
    #     module,
    #     rank: int,
    #     enabled: bool = True,
    #     is_root: Literal[False] = False,
    # ) -> Union[LoRA, Type]: ...

    # Actual implementation of from module
    @classmethod
    def from_module(cls, module, rank: int, inplace=True):
        # Get all of the layers
        for name in get_state_layers_names(module):
            # Single layer
            sub_module = getattr(module, name)

            # when you encounter a known layer that can be made into a LoRA layer do it
            if isinstance(sub_module, nn.Linear):
                setattr(module, name, LoRA._from_linear(sub_module, rank))  # type: ignore
            # elif isinstance(module, nn.Embedding):
            #     return LoRA._from_embedding(module, rank)
            # elif isinstance(module, nn.MultiheadAttention):
            #     return LoRA._from_multihead_attention(module, rank)  # type: ignore

        # Return the new modified module
        return module

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


# Enable the LoRA parameters
def enable_lora(module: Union[Type, LoRA]) -> None:
    for name in get_state_layers_names(module):
        sub_module = getattr(module, name)

        if isinstance(sub_module, LoRA):
            sub_module.enabled = False


# Disable the LoRA parameters
def disable_lora(module: Union[Type, LoRA]) -> None:
    # For all the layers
    for name in get_state_layers_names(module):
        sub_module = getattr(module, name)

        if isinstance(sub_module, LoRA):
            sub_module.enabled = False

        # for child in module.children():
        #     disable_lora(child)
        # if isinstance(module, LoRA):
        #     module.enabled = False


# def merge_lora(module: Union[Type, LoRA], inplace: bool = False) -> Type:
#     out = module if inplace else deepcopy(module)
#     for name, child in out.named_children():
#         out._modules[name] = merge_lora(child, inplace=inplace)
#
#     if isinstance(out, LoRA):
#         if out.lora_module is None:
#             return out.module
#         else:
#             return out.lora_module.merge(out.module, inplace=inplace)
#     else:
#         return out
#


# def remove_lora(module: Union[Type, LoRA], inplace: bool = False) -> Type:
#     """Remove all LoRA modules from the model."""
#     out = module if inplace else deepcopy(module)
#
#     # print(out)
#     for name in get_state_layers_names(out):
#         setattr(out, name, remove_lora(getattr(out, name), inplace=inplace))
#         # out._modules[name] = remove_lora(child, inplace=inplace)
#
#     if isinstance(out, LoRA):
#         return out.module
#     else:
#         return out


def remove_lora(module: Union[Type, LoRA], inplace: bool = False) -> Type:
    """Remove all LoRA modules from the model."""
    out = module if inplace else deepcopy(module)

    layer_names = get_state_layers_names(out)

    for name in layer_names:
        sub_module = getattr(out, name, None)
        if sub_module is not None and isinstance(sub_module, LoRA):
            setattr(out, name, sub_module.module)
        else:
            setattr(out, name, sub_module)

    return out
