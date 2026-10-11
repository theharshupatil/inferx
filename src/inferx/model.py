import torch
from torch import nn

class TinyModel(nn.Module):
    """Tiny compute workload for validating the benchmark harness."""
    def __init__(self, vocab_size=512, hidden=128):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden)
        self.proj = nn.Linear(hidden, vocab_size)

    def forward(self, tokens):
        return self.proj(self.embedding(tokens))
