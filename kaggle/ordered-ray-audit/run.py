#!/usr/bin/env python3
import subprocess
import sys
from pathlib import Path


COMMIT = "f3e1f7de12556d926b629fde354076d39c32ca90"


def run(command, cwd=None):
    print("+", " ".join(map(str, command)), flush=True)
    subprocess.run(list(map(str, command)), cwd=cwd, check=True)


def one(filename):
    matches = list(Path("/kaggle/input").rglob(filename))
    if len(matches) != 1:
        raise RuntimeError(f"Expected one {filename}, found {matches}")
    return matches[0]


def one_suffix(suffix):
    matches = [path for path in Path("/kaggle/input").rglob("*") if str(path).endswith(suffix)]
    if len(matches) != 1:
        raise RuntimeError(f"Expected one path ending in {suffix}, found {matches}")
    return matches[0]


repo = Path("/kaggle/working/nnue-pytorch")
run(["git", "clone", "https://github.com/lualum/nnue-pytorch.git", repo])
run(["git", "checkout", "--detach", COMMIT], cwd=repo)
run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"], cwd=repo)
run([sys.executable, "-m", "pip", "uninstall", "-y", "tensorflow", "jax", "jaxlib"], cwd=repo)
run([
    "cmake", "-S", "data_loader/cpp", "-B", "build",
    "-DCMAKE_BUILD_TYPE=Release", f"-DLIB_COPY_DIR={repo}",
], cwd=repo)
run(["cmake", "--build", "build", "-j2"], cwd=repo)
run([
    sys.executable, "scripts/audit_ordered_ray_checkpoint.py",
    "--checkpoint", one_suffix("candidate/training_logs/version_0/checkpoints/epoch=3-step=2440.ckpt"),
    "--reference-model", one("candidate-ray-warmup-init.pt"),
    "--validation-data", one("dfrc_n5000_piece_0.binpack"),
    "--output=/kaggle/working/ordered-ray-audit.json",
], cwd=repo)
