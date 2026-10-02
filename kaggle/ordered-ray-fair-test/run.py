#!/usr/bin/env python3
"""Kaggle entry point pinned to the public OrderedRay2 experiment commit."""

import hashlib
import subprocess
import sys
from pathlib import Path


REPOSITORY = "https://github.com/lualum/nnue-pytorch.git"
EXPERIMENT_COMMIT = "bf363cd3ee77345e73fc30900ee4414836104340"


def run(command, cwd=None):
    print("+", " ".join(map(str, command)), flush=True)
    subprocess.run(list(map(str, command)), cwd=cwd, check=True)


def find_input(filename):
    matches = list(Path("/kaggle/input").rglob(filename))
    if len(matches) != 1:
        available = [str(path) for path in Path("/kaggle/input").rglob("*.binpack")]
        raise RuntimeError(
            f"Expected one mounted {filename}, found {matches}; binpacks={available}"
        )
    print(f"Resolved {filename} to {matches[0]}", flush=True)
    return matches[0]


working = Path("/kaggle/working")
repo = working / "nnue-pytorch"
if repo.exists():
    raise RuntimeError(f"Refusing to overwrite existing {repo}")

run(["git", "clone", REPOSITORY, repo])
run(["git", "checkout", "--detach", EXPERIMENT_COMMIT], cwd=repo)
run(["git", "rev-parse", "HEAD"], cwd=repo)
run(["git", "status", "--short"], cwd=repo)

for relative in (
    "model/modules/features/ordered_ray_2.py",
    "scripts/run_ordered_ray_fair_test.py",
    "kaggle/ordered-ray-fair-test/stockfish-ordered-ray.patch",
):
    data = (repo / relative).read_bytes()
    print(f"sha256 {hashlib.sha256(data).hexdigest()}  {relative}", flush=True)

run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"], cwd=repo)

train_data = find_input("wrongIsRight_nodes5000pv2.binpack")
validation_data = find_input("dfrc_n5000_piece_0.binpack")
run([
    sys.executable,
    "scripts/run_ordered_ray_fair_test.py",
    f"--train-data={train_data}",
    f"--validation-data={validation_data}",
    "--engine-patch=kaggle/ordered-ray-fair-test/stockfish-ordered-ray.patch",
    "--output=/kaggle/working/ordered_ray_fair_test",
    "--threads=2",
    "--workers=2",
    "--epoch-size=20000000",
    "--validation-size=1000000",
    "--games=1000",
], cwd=repo)
