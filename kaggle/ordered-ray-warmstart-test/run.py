#!/usr/bin/env python3
"""Warm-start OrderedRay2 from the trained baseline produced by the prior run."""

import hashlib
import subprocess
import sys
from pathlib import Path


REPOSITORY = "https://github.com/lualum/nnue-pytorch.git"
EXPERIMENT_COMMIT = "9f4ef1889e873154f6cb22e12155c183c0bb1921"
PARENT_SHA256 = "e0d2a2956c39264b135bd78d4fc19a33f75edfb8d13f806c4daeaabd8ab38457"


def run(command, cwd=None):
    print("+", " ".join(map(str, command)), flush=True)
    subprocess.run(list(map(str, command)), cwd=cwd, check=True)


def find_input(filename):
    matches = list(Path("/kaggle/input").rglob(filename))
    if len(matches) != 1:
        raise RuntimeError(f"Expected one mounted {filename}, found {matches}")
    print(f"Resolved {filename} to {matches[0]}", flush=True)
    return matches[0]


working = Path("/kaggle/working")
repo = working / "nnue-pytorch"
run(["git", "clone", REPOSITORY, repo])
run(["git", "checkout", "--detach", EXPERIMENT_COMMIT], cwd=repo)
run(["git", "rev-parse", "HEAD"], cwd=repo)
run(["git", "status", "--short"], cwd=repo)
run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"], cwd=repo)
# TensorBoard falls back to its bundled TensorFlow compatibility stub when the
# optional TensorFlow stack is absent. Kaggle's Python 3.13 image currently
# ships JAX/TensorFlow builds that require NumPy 2, while nnue-pytorch pins
# NumPy 1.x, so remove those unused optional packages before training.
run([
    sys.executable, "-m", "pip", "uninstall", "-y",
    "tensorflow", "jax", "jaxlib",
], cwd=repo)

parent = find_input("baseline.nnue")
parent_hash = hashlib.sha256(parent.read_bytes()).hexdigest()
print(f"parent sha256={parent_hash}", flush=True)
if parent_hash != PARENT_SHA256:
    raise RuntimeError(f"Unexpected parent network hash: {parent_hash}")

run([
    sys.executable,
    "scripts/run_ordered_ray_fair_test.py",
    f"--train-data={find_input('wrongIsRight_nodes5000pv2.binpack')}",
    f"--validation-data={find_input('dfrc_n5000_piece_0.binpack')}",
    f"--initial-network={parent}",
    "--engine-patch=kaggle/ordered-ray-fair-test/stockfish-ordered-ray.patch",
    "--output=/kaggle/working/ordered_ray_warmstart_test",
    "--threads=2",
    "--workers=2",
    "--epochs=4",
    "--epoch-size=5000000",
    "--validation-size=1000000",
    "--learning-rate=0.00004375",
    "--ray-warmup-size=5000000",
    "--ray-warmup-learning-rate=0.000875",
    "--games=1000",
], cwd=repo)
