import torch
from inferx.kv_cache import LayerCache

def test_cache_appends_on_sequence_axis():
    cache = LayerCache()
    cache.append(torch.ones(2, 4, 1, 8), torch.ones(2, 4, 1, 8))
    cache.append(torch.zeros(2, 4, 1, 8), torch.zeros(2, 4, 1, 8))
    assert cache.sequence_length == 2
    assert cache.key.shape == (2, 4, 2, 8)

def test_cache_rejects_key_value_mismatch():
    cache = LayerCache()
    try:
        cache.append(torch.ones(1, 1, 1, 4), torch.ones(1, 1, 1, 5))
    except ValueError:
        return
    raise AssertionError("Expected shape mismatch to raise")
