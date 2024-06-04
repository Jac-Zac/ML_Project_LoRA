from __future__ import annotations

from tinygrad import Tensor, nn

from .base import BaseDoRAModule


class LinearDoRAModule(BaseDoRAModule):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int,
        # TODO: Add support for bias
        # bias: bool = False,
        alpha: float = 1.0,
        dropout: float = 0.0,
    ):
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

        # Multiply by output projection layer
        return x @ self.out_proj

    def merge(self, module: nn.Linear, inplace: bool = False) -> nn.Linear:
        # einstein notation:
        # - i: input features
        # - o: output features
        # - r: rank

        # Get the shape of the Linear Layer
        out_features, in_features = module.weight.shape

        # Avoid tracking the gradient since there is no need for it in the merging
        with Tensor.train():
            # dora_weight = Tensor.einsum("i r, r o -> o i", self.in_proj, self.out_proj)
            dora_weight = (self.in_proj @ self.out_proj).transpose(1, 0)

            # Update the weights in place
            if inplace:
                module.weight += dora_weight
                return module

            # Update the weights of a new Layer
            out = nn.Linear(
                in_features=in_features,
                out_features=out_features,
                bias=module.bias is not None,
            )
            out.weight = module.weight + dora_weight
            out.bias = module.bias
            return out

    @property
    def weight(self) -> Tensor:
        # return torch.einsum("i r, r o -> o i", self.in_proj, self.out_proj)
        return (self.in_proj @ self.out_proj).transpose(1, 0)
