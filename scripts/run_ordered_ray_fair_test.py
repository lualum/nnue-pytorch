#!/usr/bin/env python3
"""Matched fine-tune, engine benchmark, and paired games for OrderedRay2."""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import json
import math
import os
import platform
import random
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path


BASELINE_FEATURES = "Full_Threats+PP_3Wide+HalfKAv2_hm^"
CANDIDATE_FEATURES = "Full_Threats+PP_3Wide+OrderedRay2+HalfKAv2_hm^"
STOCKFISH_COMMIT = "49ea5ded38315cff8e67f4a677a9e7811612fbf6"
OFFICIAL_NET = "nn-252f33942263.nnue"


def run(command, *, cwd, env=None, log=None, stdin=None):
    print("+", " ".join(map(str, command)), flush=True)
    result = subprocess.run(
        list(map(str, command)), cwd=cwd, env=env, input=stdin, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False,
    )
    if log:
        Path(log).write_text(result.stdout, encoding="utf-8")
    print(result.stdout[-4000:], flush=True)
    if result.returncode:
        raise subprocess.CalledProcessError(result.returncode, command, output=result.stdout)
    return result.stdout


def final_metrics(path):
    final = {}
    with path.open(newline="", encoding="utf-8") as source:
        for row in csv.DictReader(source):
            for key in ("train_loss_epoch", "val_loss_epoch"):
                if row.get(key):
                    final[key] = float(row[key])
    return final


def train(repo, output, name, features, model, gpu, args):
    root = output / name
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    env["TORCHINDUCTOR_CACHE_DIR"] = str(output / f"torch_cache_{name}")
    cmd = [
        sys.executable, "ddp_launcher.py", "train.py", args.train_data,
        "--validation-datasets", args.validation_data,
        "--accelerator=cuda", "--gpus=0", "--no-affinity",
        f"--threads={args.threads}", f"--num-workers={args.workers}",
        f"--batch-size={args.batch_size}", f"--epoch-size={args.epoch_size}",
        f"--validation-size={args.validation_size}", f"--max-epochs={args.epochs}",
        "--network-save-period=1000000", "--save-last-network=True",
        f"--default-root-dir={root}", f"--features={features}",
        f"--resume-from-model={model}", f"--seed={args.seed}",
        "--optimizer-name=adamw",
    ]
    started = time.monotonic()
    run(cmd, cwd=repo, env=env, log=output / f"{name}-training.log")
    elapsed = time.monotonic() - started
    version = root / "training_logs" / "version_0"
    return {
        "elapsed_seconds": elapsed,
        **final_metrics(version / "metrics.csv"),
        "checkpoint": str(version / "checkpoints" / "last.ckpt"),
    }


def parse_bench(text):
    nodes = int(re.findall(r"Nodes searched\s*:\s*(\d+)", text)[-1])
    nps = int(re.findall(r"Nodes/second\s*:\s*(\d+)", text)[-1])
    return {"nodes": nodes, "nps": nps}


def parse_match(pgn, candidate="OrderedRay2"):
    headers = []
    current = {}
    for line in pgn.read_text(encoding="utf-8").splitlines():
        match = re.match(r'^\[(\w+) "(.*)"\]$', line)
        if match:
            current[match.group(1)] = match.group(2)
        elif not line and current.get("Result"):
            headers.append(current)
            current = {}
    if current.get("Result"):
        headers.append(current)

    scores = []
    for game in headers:
        result = game["Result"]
        if result == "1/2-1/2":
            scores.append(0.5)
        else:
            candidate_white = game.get("White") == candidate
            scores.append(float((result == "1-0") == candidate_white))
    wins = sum(s == 1 for s in scores)
    draws = sum(s == 0.5 for s in scores)
    losses = sum(s == 0 for s in scores)
    score = sum(scores) / len(scores)
    elo = 400 * math.log10(score / (1 - score)) if 0 < score < 1 else math.copysign(math.inf, score - .5)

    # Opening-pair bootstrap preserves the correlation induced by color swaps.
    pairs = [scores[i:i + 2] for i in range(0, len(scores) - 1, 2)]
    rng = random.Random(20261002)
    samples = []
    for _ in range(20000):
        total = sum(sum(pairs[rng.randrange(len(pairs))]) for _ in pairs)
        s = min(max(total / (2 * len(pairs)), 1e-9), 1 - 1e-9)
        samples.append(400 * math.log10(s / (1 - s)))
    samples.sort()
    return {
        "games": len(scores), "wins": wins, "draws": draws, "losses": losses,
        "score_percent": 100 * score, "elo": elo,
        "elo_95ci": [samples[500], samples[19499]],
    }


def play(c_chess, baseline_engine, candidate_engine, baseline_net, candidate_net,
         book, output, games, mode):
    pgn = output / f"{mode}.pgn"
    common = [
        c_chess, "-each", "option.Hash=16", "option.Threads=1",
        "-engine", f"cmd={baseline_engine}", "name=Baseline",
        f"option.EvalFile={baseline_net}",
        "-engine", f"cmd={candidate_engine}", "name=OrderedRay2",
        f"option.EvalFile={candidate_net}",
        "-games", str(games), "-concurrency", "2", "-openings", f"file={book}",
        "order=random", "srand=20261002", "-repeat", "-resign", "count=3",
        "score=700", "-draw", "number=40", "count=8", "score=10",
        "-pgn", pgn, "0",
    ]
    if mode == "equal_nodes":
        common[2:2] = ["nodes=10000"]
    else:
        common[2:2] = ["tc=1+0.01"]
    run(common, cwd=output, log=output / f"{mode}.log")
    return parse_match(pgn)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-data", required=True)
    parser.add_argument("--validation-data", required=True)
    parser.add_argument("--output", type=Path, default=Path("ordered_ray_fair_test"))
    parser.add_argument("--engine-patch", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--epoch-size", type=int, default=20_000_000)
    parser.add_argument("--validation-size", type=int, default=1_000_000)
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--games", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20261002)
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[1]
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)

    import torch
    if torch.cuda.device_count() != 2:
        raise RuntimeError(f"Expected exactly two CUDA GPUs, got {torch.cuda.device_count()}")

    run(["cmake", "-S", "data_loader/cpp", "-B", "build", "-DCMAKE_BUILD_TYPE=Release", f"-DLIB_COPY_DIR={repo}"], cwd=repo)
    run(["cmake", "--build", "build", "-j2"], cwd=repo)
    run([sys.executable, "-m", "pytest", "-q", "tests/test_ordered_ray_2.py"], cwd=repo)

    official_net = output / OFFICIAL_NET
    run(["curl", "-L", "-o", official_net, f"https://tests.stockfishchess.org/api/nn/{OFFICIAL_NET}"], cwd=repo)
    init_baseline = output / "baseline-init.pt"
    init_candidate = output / "candidate-init.pt"
    run([sys.executable, "scripts/initialize_ordered_ray_from_stockfish.py", official_net, init_baseline, init_candidate], cwd=repo)

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        b = pool.submit(train, repo, output, "baseline", BASELINE_FEATURES, init_baseline, 0, args)
        c = pool.submit(train, repo, output, "candidate", CANDIDATE_FEATURES, init_candidate, 1, args)
        baseline = b.result()
        candidate = c.result()

    baseline_net = output / "baseline.nnue"
    candidate_net = output / "candidate.nnue"
    run([sys.executable, "serialize.py", baseline["checkpoint"], baseline_net, "--features", BASELINE_FEATURES], cwd=repo)
    run([sys.executable, "serialize.py", candidate["checkpoint"], candidate_net, "--features", CANDIDATE_FEATURES], cwd=repo)

    sf = output / "Stockfish"
    run(["git", "clone", "https://github.com/official-stockfish/Stockfish.git", sf], cwd=output)
    run(["git", "checkout", STOCKFISH_COMMIT], cwd=sf)
    run(["make", "-j2", "build", "ARCH=x86-64-avx2"], cwd=sf / "src")
    baseline_engine = output / "stockfish-baseline"
    shutil.copy2(sf / "src" / "stockfish", baseline_engine)
    run(["git", "apply", args.engine_patch.resolve()], cwd=sf)
    run(["make", "clean"], cwd=sf / "src")
    run(["make", "-j2", "build", "ARCH=x86-64-avx2"], cwd=sf / "src")
    candidate_engine = output / "stockfish-ordered-ray"
    shutil.copy2(sf / "src" / "stockfish", candidate_engine)

    ccli = output / "c-chess-cli"
    run(["git", "clone", "--depth", "1", "https://github.com/lucasart/c-chess-cli.git", ccli], cwd=output)
    run([sys.executable, "make.py"], cwd=ccli)
    book_zip = output / "UHO_Lichess_4852_v1.epd.zip"
    run(["curl", "-L", "-o", book_zip, "https://github.com/official-stockfish/books/raw/master/UHO_Lichess_4852_v1.epd.zip"], cwd=output)
    run(["unzip", "-o", book_zip, "-d", output], cwd=output)
    book = next(output.glob("UHO_Lichess_4852_v1*.epd"))

    bench_input = "setoption name EvalFile value {}\nisready\nbench 16 1 13 default depth\nquit\n"
    baseline_bench = parse_bench(run([baseline_engine], cwd=output, stdin=bench_input.format(baseline_net)))
    candidate_bench = parse_bench(run([candidate_engine], cwd=output, stdin=bench_input.format(candidate_net)))
    matches = {
        "equal_nodes": play(ccli / "c-chess-cli", baseline_engine, candidate_engine, baseline_net, candidate_net, book, output, args.games, "equal_nodes"),
        "equal_time": play(ccli / "c-chess-cli", baseline_engine, candidate_engine, baseline_net, candidate_net, book, output, args.games, "equal_time"),
    }

    results = {
        "experiment": "OrderedRay2 matched Stockfish ablation",
        "stockfish_commit": STOCKFISH_COMMIT,
        "initial_network": OFFICIAL_NET,
        "hardware": {"gpus": [torch.cuda.get_device_name(i) for i in range(2)], "cpu_count": os.cpu_count(), "platform": platform.platform()},
        "config": vars(args) | {"output": str(output), "engine_patch": str(args.engine_patch)},
        "baseline": baseline,
        "candidate": candidate,
        "validation_loss_delta": candidate["val_loss_epoch"] - baseline["val_loss_epoch"],
        "bench": {"baseline": baseline_bench, "candidate": candidate_bench, "nps_delta_percent": 100 * (candidate_bench["nps"] / baseline_bench["nps"] - 1)},
        "matches": matches,
        "method": [
            "Both nets start from the same official Stockfish network; OrderedRay2 starts at zero.",
            "Both see the same training positions, optimizer, seed, batch count, and separate held-out validation source.",
            "Openings are repeated with colors swapped; equal-node isolates net quality and equal-time includes feature overhead.",
        ],
    }
    (output / "results.json").write_text(json.dumps(results, indent=2, default=str) + "\n", encoding="utf-8")
    print(json.dumps(results, indent=2, default=str), flush=True)


if __name__ == "__main__":
    main()
