#!/usr/bin/env python3
"""Run the first matched OrderedRay2 screen on exactly two CUDA GPUs."""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path


BASELINE_FEATURES = "Full_Threats+PP_3Wide+HalfKAv2_hm^"
ORDERED_RAY_FEATURES = BASELINE_FEATURES + "+OrderedRay2"


def run(
    command: list[str],
    *,
    cwd: Path,
    log_path: Path | None = None,
    env: dict[str, str] | None = None,
) -> None:
    print("+", " ".join(command), flush=True)
    if log_path is None:
        subprocess.run(command, cwd=cwd, check=True, env=env)
        return

    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            command,
            cwd=cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=env,
        )
        assert process.stdout is not None
        for line in process.stdout:
            sys.stdout.write(line)
            log.write(line)
        return_code = process.wait()
    if return_code != 0:
        raise subprocess.CalledProcessError(return_code, command)


def final_metrics(metrics_path: Path) -> dict[str, float]:
    last: dict[str, float] = {}
    with metrics_path.open(newline="", encoding="utf-8") as metrics_file:
        for row in csv.DictReader(metrics_file):
            for name in ("train_loss_epoch", "val_loss_epoch"):
                value = row.get(name)
                if value:
                    last[name] = float(value)
    if "val_loss_epoch" not in last:
        raise RuntimeError(f"No validation loss found in {metrics_path}")
    return last


def train(
    repo: Path,
    output: Path,
    name: str,
    features: str,
    gpu: int,
    args: argparse.Namespace,
) -> dict[str, object]:
    root = output / name
    command = [
        sys.executable,
        "ddp_launcher.py",
        "train.py",
        str(args.dataset),
        "--validation-datasets",
        str(args.dataset),
        "--accelerator=cuda",
        "--gpus=0",
        "--no-affinity",
        f"--threads={args.threads}",
        f"--num-workers={args.workers}",
        f"--batch-size={args.batch_size}",
        f"--epoch-size={args.epoch_size}",
        f"--validation-size={args.validation_size}",
        f"--max-epochs={args.epochs}",
        "--network-save-period=1000000",
        "--save-last-network=False",
        f"--default-root-dir={root}",
        f"--features={features}",
        f"--seed={args.seed}",
        "--optimizer-name=adamw",
    ]
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    env["TORCHINDUCTOR_CACHE_DIR"] = str(output / f"torch_cache_{name}")
    started = time.monotonic()
    run(command, cwd=repo, log_path=output / f"{name}.log", env=env)
    elapsed = time.monotonic() - started
    metrics_path = root / "training_logs" / "version_0" / "metrics.csv"
    return {
        "features": features,
        "elapsed_seconds": elapsed,
        **final_metrics(metrics_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=Path(".pgo/small.binpack"))
    parser.add_argument("--output", type=Path, default=Path("ordered_ray_first_test"))
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--epoch-size", type=int, default=262_144)
    parser.add_argument("--validation-size", type=int, default=65_536)
    parser.add_argument("--batch-size", type=int, default=8_192)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20261002)
    args = parser.parse_args()

    repo = Path(__file__).resolve().parents[1]
    args.dataset = (repo / args.dataset).resolve() if not args.dataset.is_absolute() else args.dataset
    args.output = (repo / args.output).resolve() if not args.output.is_absolute() else args.output
    args.output.mkdir(parents=True, exist_ok=True)

    import torch

    gpu_count = torch.cuda.device_count()
    if gpu_count != 2:
        raise RuntimeError(f"This experiment requires exactly two CUDA GPUs; found {gpu_count}")

    hardware = {
        "gpu_count": gpu_count,
        "gpus": [torch.cuda.get_device_name(i) for i in range(gpu_count)],
        "cuda": torch.version.cuda,
        "torch": torch.__version__,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "cpu_count": os.cpu_count(),
    }
    print(json.dumps(hardware, indent=2), flush=True)

    run(
        [
            "cmake",
            "-S",
            "data_loader/cpp",
            "-B",
            "build",
            "-DCMAKE_BUILD_TYPE=Release",
            f"-DLIB_COPY_DIR={repo}",
        ],
        cwd=repo,
    )
    run(["cmake", "--build", "build", "-j2"], cwd=repo)
    run(
        [sys.executable, "-m", "pytest", "-q", "tests/test_ordered_ray_2.py"],
        cwd=repo,
    )

    # The bundled PGO sample is a single tiny binpack and cannot be sharded
    # safely across DDP ranks. Run the matched models concurrently instead:
    # one model per T4, with identical data exposure and seeds.
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        baseline_future = executor.submit(
            train, repo, args.output, "baseline", BASELINE_FEATURES, 0, args
        )
        ordered_ray_future = executor.submit(
            train, repo, args.output, "ordered_ray", ORDERED_RAY_FEATURES, 1, args
        )
        baseline = baseline_future.result()
        ordered_ray = ordered_ray_future.result()
    baseline_loss = float(baseline["val_loss_epoch"])
    ordered_ray_loss = float(ordered_ray["val_loss_epoch"])

    results = {
        "experiment": "OrderedRay2 first matched screen",
        "comparison": "Current Stockfish feature baseline vs additive OrderedRay2",
        "commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repo, text=True
        ).strip(),
        "dataset": str(args.dataset),
        "dataset_bytes": args.dataset.stat().st_size,
        "config": {
            "epochs": args.epochs,
            "epoch_size": args.epoch_size,
            "validation_size": args.validation_size,
            "execution": "parallel matched runs, one model per GPU",
            "batch_size_per_model": args.batch_size,
            "threads_per_model": args.threads,
            "workers_per_model": args.workers,
            "seed": args.seed,
        },
        "hardware": hardware,
        "baseline": baseline,
        "ordered_ray": ordered_ray,
        "validation_loss_delta": ordered_ray_loss - baseline_loss,
        "validation_loss_relative_percent": 100.0
        * (ordered_ray_loss - baseline_loss)
        / baseline_loss,
        "limitations": [
            "Smoke-screen budget over the repository's small cyclic PGO dataset.",
            "Training and validation use the same source file; this is not a strength estimate.",
            "No engine-side OrderedRay2 inference exists yet, so no Elo match is reported.",
        ],
    }
    result_path = args.output / "results.json"
    result_path.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(results, indent=2), flush=True)
    print(f"RESULTS_PATH={result_path}", flush=True)


if __name__ == "__main__":
    main()
