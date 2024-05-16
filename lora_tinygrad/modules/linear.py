from __future__ import annotations

from typing import Optional

from tinygrad import Tensor, nn

from .base import BaseLoRAModule


class LinearLoRAModule(BaseLoRAModule):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int,
        # TODO: Add support for bias
        # bias: bool = False,
        alpha: float = 1.0,
        dropout: float = 0.0,
        device: Optional[Tensor.device] = None,
        dtype: Optional[Tensor.dtype] = None,
    ):
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank

        self.in_proj = nn.Linear(
            in_features=in_features, out_features=rank, device=device, dtype=dtype
        )
        self.out_proj = nn.Linear(
            in_features=rank, out_features=out_features, device=device, dtype=dtype
        )

        # Set the droput probability
        self.dropout_prob = dropout

        # NOTE: The original LoRA paper recommends multiplying the output of 'in_proj'
        # by (alpha / rank).  This adds more computation to the forward pass, and it's
        # mathematically equivalent to scaling 'in_proj' by (alpha / rank) ahead of
        # time.  I have chosen the second option for simplicity.
        # nn.init.kaiming_uniform_(self.in_proj, alpha / rank)
        # nn.init.zeros_(self.out_proj)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(in_features={self.in_features}, "
            f"out_features={self.out_features}, rank={self.rank})"
        )

    def __call__(self, x: Tensor) -> Tensor:
        # Multiply x by the in projection
        x = x @ self.in_proj

        # Apply dropout
        x = Tensor.dropout(x, p=self.dropout_prob)

        # Multiply by output projection layer
        return x @ self.out_proj

    def merge(self, module: nn.Linear, inplace: bool = False) -> nn.Linear:
        # einstein notation:
        # - i: input features
        # - o: output features
        # - r: rank
        with Tensor.no_grad():
            lora_weight = Tensor.einsum("i r, r o -> o i", self.in_proj, self.out_proj)

            if inplace:
                module.weight.data += lora_weight
                return module

            out = nn.Linear(
                in_features=module.in_features,
                out_features=module.out_features,
                bias=module.bias is not None,
                device=module.weight.device,
                dtype=module.weight.dtype,
            )
            out.weight.data = module.weight.data + lora_weight
            out.bias = module.bias
            return out

    @property
    def weight(self) -> Tensor:
        return torch.einsum("i r, r o -> o i", self.in_proj, self.out_proj)
