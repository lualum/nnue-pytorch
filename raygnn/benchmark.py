"""One-shot reference inference benchmark including feature construction."""

import argparse
import time

import chess
import torch

from .encoding import boards_to_batch
from .model import RayGNN


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--iterations", type=int, default=20)
    args = parser.parse_args()
    if args.iterations < 1 or args.batch_size < 1:
        parser.error("iterations and batch-size must be positive")
    batch = boards_to_batch([chess.Board()] * args.batch_size, args.device)
    model = RayGNN().to(args.device).eval()

    def synchronize():
        if args.device.startswith("cuda"):
            torch.cuda.synchronize()

    with torch.inference_mode():
        for _ in range(3):
            model(batch)
        synchronize()
        start = time.perf_counter()
        for _ in range(args.iterations):
            model.geometry.rays(batch.piece)
        synchronize()
        geometry_ms = 1000 * (time.perf_counter() - start) / args.iterations
        start = time.perf_counter()
        for _ in range(args.iterations):
            model(batch)
        synchronize()
        full_ms = 1000 * (time.perf_counter() - start) / args.iterations
    parameters = sum(p.numel() for p in model.parameters())
    print(f"parameters={parameters} batch={args.batch_size} geometry_ms={geometry_ms:.3f} "
          f"full_ms={full_ms:.3f} positions_per_second={1000 * args.batch_size / full_ms:.1f}")
    if args.device.startswith("cuda"):
        print(f"peak_cuda_mib={torch.cuda.max_memory_allocated() / 2**20:.1f}")


if __name__ == "__main__":
    main()
