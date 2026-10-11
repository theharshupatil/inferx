import argparse
import time
import torch
from .model import TinyModel

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=50)
    args = parser.parse_args()
    device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    model = TinyModel().to(device).eval()
    x = torch.randint(0, 512, (args.batch_size, args.tokens), device=device)
    with torch.no_grad():
        for _ in range(args.warmup):
            model(x)
        if device == "mps":
            torch.mps.synchronize()
        elif device == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(args.iterations):
            model(x)
        if device == "mps":
            torch.mps.synchronize()
        elif device == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
    print(f"device={device}")
    print(f"batch_size={args.batch_size} sequence_length={args.tokens} iterations={args.iterations}")
    print(f"mean_forward_ms={1000*elapsed/args.iterations:.3f}")
    print("Note: this is a forward-pass microbenchmark, not autoregressive token latency.")

if __name__ == "__main__":
    main()
