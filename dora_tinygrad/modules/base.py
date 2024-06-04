from __future__ import annotations

from abc import abstractmethod

from tinygrad import Tensor


class BaseDoRAModule:
    @abstractmethod
    def forward(self, x: Tensor) -> Tensor: ...

    @abstractmethod
    def merge(self, module, inplace: bool = False): ...
