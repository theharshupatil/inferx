"""A correctness-first dynamic KV cache. Not optimized for production."""
from dataclasses import dataclass
import torch

@dataclass
class LayerCache:
    key: torch.Tensor | None = None
    value: torch.Tensor | None = None

    def append(self, key: torch.Tensor, value: torch.Tensor) -> None:
        if key.shape != value.shape:
            raise ValueError("key and value shapes must match")
        if self.key is None:
            self.key, self.value = key, value
        else:
            if key.shape[:-2] != self.key.shape[:-2] or key.shape[-1] != self.key.shape[-1]:
                raise ValueError("incompatible cache shape")
            self.key = torch.cat((self.key, key), dim=-2)
            self.value = torch.cat((self.value, value), dim=-2)

    @property
    def sequence_length(self) -> int:
        return 0 if self.key is None else self.key.shape[-2]
