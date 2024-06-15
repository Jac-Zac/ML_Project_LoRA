from __future__ import annotations

from tinygrad import Tensor, nn

from .base import BaseDoRAModule


class LinearDoRAModule(BaseDoRAModule):
    def __init__(
        self,
        m: Tensor,
        in_features: int,
        out_features: int,
        rank: int,
        linear_layer_weights: Tensor,
        # TODO: Add support for bias
        # bias: bool = False,
        alpha: float = 1.0,
        dropout: float = 0.0,
    ):

        self.m = m
        # Testing option
        # self.m = Tensor.ones(1, out_features, requires_grad=True)

        self.linear_layer_weights = linear_layer_weights

        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank

        # Define Linear projections for DoRA layers
        # NOTE: The original DoRA paper recommends multiplying the output of 'in_proj'
        # by (alpha / rank).  This adds more computation to the forward pass, and it's
        # mathematically equivalent to scaling 'in_proj' by (alpha / rank) ahead of
        self.in_proj = Tensor.kaiming_uniform(in_features, rank, requires_grad=True) * (
            alpha / rank
        )
        self.out_proj = Tensor.zeros(rank, out_features, requires_grad=True)

        # Set the droput probability
        self.dropout_prob = dropout

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

        # Output of the LoRA layer
        lora_output = x @ self.out_proj

        # TODO: Think about this to compute the norm correctly
        # The lora weights
        # print(self.linear_layer_weights.shape)
        # print((self.in_proj @ self.out_proj).T.shape)

        model_weights = self.linear_layer_weights + ((self.in_proj @ self.out_proj).T)

        # Normalization in relation to the Lora weights
        lora_output_norm = lora_output / Tensor.sqrt(
            Tensor.sum(model_weights, axis=1, keepdim=False)
        )

        # Rescaled output with the magnitude
        dora_output = self.m * lora_output_norm

        return dora_output

    def merge(self, module: nn.Linear, inplace: bool = False) -> nn.Linear:
        return NotImplemented

    @property
    def weight(self) -> Tensor:
        # return torch.einsum("i r, r o -> o i", self.in_proj, self.out_proj)
        return (self.in_proj @ self.out_proj).transpose(1, 0)
