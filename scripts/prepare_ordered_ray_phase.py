#!/usr/bin/env python3
"""Freeze a gated candidate or thaw a phase-one checkpoint for fine-tuning."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("template", type=Path, help="Candidate .pt model")
    parser.add_argument("output", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--freeze-inherited", action="store_true")
    args = parser.parse_args()

    model = torch.load(args.template, map_location="cpu", weights_only=False)
    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        model.load_state_dict(checkpoint["state_dict"])

    for parameter in model.parameters():
        parameter.requires_grad_(not args.freeze_inherited)

    ray = model.model.input.features[2]
    ray.weight.requires_grad_(True)
    ray.gate.requires_grad_(True)

    trainable = [(name, p.numel()) for name, p in model.named_parameters() if p.requires_grad]
    print(f"OrderedRay2 gate={ray.gate.item():.9g}")
    print(f"Trainable parameters={sum(size for _, size in trainable)}")
    print("Trainable tensors=" + ", ".join(name for name, _ in trainable))
    if args.freeze_inherited:
        assert {name for name, _ in trainable} == {
            "model.input.features.2.weight",
            "model.input.features.2.gate",
        }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model, args.output)


if __name__ == "__main__":
    main()
